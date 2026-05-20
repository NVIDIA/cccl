// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Single-kernel cluster-based batched top-k dispatch.
//!
//! Prototype that launches a single grid of thread block clusters to compute a
//! segmented top-k. Each cluster processes one segment end-to-end: private
//! histograms are reduced into the leader block via DSMEM atomics, then every
//! block reads the merged histogram back through DSMEM, locally identifies the
//! k-th bucket, and refines its in-register key set across radix passes.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_batched_topk_cluster.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/util_device.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda_runtime.h>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk_cluster
{
// -----------------------------------------------------------------------------
// Default tuning
// -----------------------------------------------------------------------------
// Defaults chosen so a single cluster tile holds the largest segment we expect
// in the existing test and benchmark coverage (8 KiB items). The leader
// block's BlockScan uses 1 bin per thread, so threads_per_block (256) must be
// >= num_buckets (1 << bits_per_pass = 256). Cluster size is 8 — the largest
// portable cluster size — so all eight blocks of the cluster participate in
// the histogram merge.
inline constexpr int default_cluster_size      = 8;
inline constexpr int default_threads_per_block = 256;
inline constexpr int default_items_per_thread  = 4;
inline constexpr int default_bits_per_pass     = 8;

// -----------------------------------------------------------------------------
// Kernel entry point
// -----------------------------------------------------------------------------
template <int ClusterSize,
          int ThreadsPerBlock,
          int ItemsPerThread,
          int BitsPerPass,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
__launch_bounds__(ThreadsPerBlock) __cluster_dims__(ClusterSize, 1, 1) _CCCL_KERNEL_ATTRIBUTES void
  device_segmented_topk_cluster_kernel(
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k_param,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments)
{
  using agent_t = agent_batched_topk_cluster<
    ClusterSize,
    ThreadsPerBlock,
    ItemsPerThread,
    BitsPerPass,
    KeyInputItItT,
    KeyOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT>;

  __shared__ typename agent_t::TempStorage temp_storage;

  agent_t agent(temp_storage, d_key_segments_it, d_key_segments_out_it, segment_sizes, k_param, select_directions, num_segments);

  agent.Process();
}

// -----------------------------------------------------------------------------
// Dispatch
// -----------------------------------------------------------------------------
// The dispatch is intentionally narrow: it currently only supports the
// keys-only path and assumes the maximum segment size fits in a single cluster
// tile (`ClusterSize * ThreadsPerBlock * ItemsPerThread`). Both restrictions
// match the small/decode-style segmented top-k workloads we are prototyping for.
template <int ClusterSize        = default_cluster_size,
          int ThreadsPerBlock    = default_threads_per_block,
          int ItemsPerThread     = default_items_per_thread,
          int BitsPerPass        = default_bits_per_pass,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename TotalNumItemsGuaranteeT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k_param,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments,
  [[maybe_unused]] TotalNumItemsGuaranteeT total_num_items_guarantee,
  cudaStream_t stream = nullptr)
{
  static_assert(ClusterSize >= 1 && ClusterSize <= 8, "ClusterSize must be in [1, 8]");

  constexpr int cluster_tile = ClusterSize * ThreadsPerBlock * ItemsPerThread;

  static_assert(detail::params::static_max_value_v<SegmentSizeParameterT> <= cluster_tile,
                "Cluster top-k prototype only supports segments that fit in a single cluster tile.");

  // The harness expects temp_storage_bytes > 0. Allocate a single byte placeholder.
  size_t allocation_sizes[1] = {1};
  void* allocations[1]       = {};
  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  {
    return cudaSuccess;
  }

  static_assert(!detail::params::is_per_segment_param_v<NumSegmentsParameterT>,
                "Number of segments must be resolved on the host.");

  using num_segments_val_t  = typename NumSegmentsParameterT::value_type;
  const auto num_seg_val    = num_segments.get_param(num_segments_val_t{0});
  const auto num_seg_unsigned = static_cast<unsigned long long>(num_seg_val);
  if (num_seg_unsigned == 0)
  {
    return cudaSuccess;
  }

  // Cluster launches require compute capability 9.0+.
  int sm_version = 0;
  if (const auto error = CubDebug(SmVersionUncached(sm_version)))
  {
    return error;
  }
  if (sm_version < 900)
  {
    return cudaErrorNotSupported;
  }

  const auto grid_blocks = static_cast<unsigned long long>(num_seg_unsigned) * static_cast<unsigned long long>(ClusterSize);
  if (grid_blocks > static_cast<unsigned long long>(::cuda::std::numeric_limits<int>::max()))
  {
    return cudaErrorInvalidValue;
  }

  auto kernel = device_segmented_topk_cluster_kernel<
    ClusterSize,
    ThreadsPerBlock,
    ItemsPerThread,
    BitsPerPass,
    KeyInputItItT,
    KeyOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT>;

  // The kernel declares `__cluster_dims__(ClusterSize, 1, 1)`, so the
  // cluster dimension is fixed at compile time. Launch via the standard
  // triple-chevron syntax; the cluster setup is applied automatically.
  if (const auto error =
        CubDebug(THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
                   dim3(static_cast<unsigned int>(grid_blocks), 1, 1),
                   dim3(static_cast<unsigned int>(ThreadsPerBlock), 1, 1),
                   0,
                   stream)
                   .doit(
                     kernel,
                     d_key_segments_it,
                     d_key_segments_out_it,
                     segment_sizes,
                     k_param,
                     select_directions,
                     num_segments)))
  {
    return error;
  }

  // Synchronously surface any launch-time errors back to the caller. The
  // prototype is keyed on `cudaLaunchKernelEx`, which may return success even
  // when the cluster launch fails on the device (e.g. insufficient resources),
  // so we explicitly check `cudaDeviceSynchronize` here while bringing up the
  // implementation.
  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  return CubDebug(detail::DebugSyncStream(stream));
}

} // namespace detail::batched_topk_cluster

CUB_NAMESPACE_END
