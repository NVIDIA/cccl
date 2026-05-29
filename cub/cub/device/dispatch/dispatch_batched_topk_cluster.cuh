// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Cluster-based batched top-k dispatch.
//!
//! Prototype that launches a grid of thread block clusters to compute a
//! segmented top-k. Each cluster processes one segment end-to-end: private
//! histograms are reduced into the leader block via DSMEM atomics, then every
//! block reads the merged histogram back through DSMEM, locally identifies the
//! k-th bucket, and refines its in-register key set across radix passes.
//!
//! Two kernels share the agent body:
//!   * Host: no `__cluster_dims__`, launched via `cudaLaunchKernelExC` with the
//!     cluster width chosen at runtime (up to 16 on Hopper).
//!   * CDP: static `__cluster_dims__(max_portable_cluster_blocks, 1, 1)`,
//!     gated on `CUB_RDC_ENABLED` so CDP-disabled builds skip emitting it
//!     (same pattern as `dispatch_segmented_sort.cuh`).

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
#include <cub/device/dispatch/tuning/tuning_batched_topk_cluster.cuh>
#include <cub/util_device.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__numeric/narrow.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <nv/target>

#include <cuda_runtime.h>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk_cluster
{
// -----------------------------------------------------------------------------
// Hardware constants
// -----------------------------------------------------------------------------
// Largest cluster width guaranteed on every SM 9.0+ device.
inline constexpr int max_portable_cluster_blocks = 8;

// CUDA's hardware ceiling on cluster width (Hopper supports up to 16).
inline constexpr int max_supported_cluster_blocks = 16;

// -----------------------------------------------------------------------------
// Cluster-size selection
// -----------------------------------------------------------------------------
// Tightest upper bound carried by `SegmentSizeParameterT`. For
// `per_segment_param` this can be the loose `numeric_limits<T>::max()`.
template <typename SegmentSizeParameterT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto
runtime_max_segment_size([[maybe_unused]] const SegmentSizeParameterT& segment_sizes) noexcept
{
  if constexpr (detail::params::is_static_param_v<SegmentSizeParameterT>)
  {
    return detail::params::static_max_value_v<SegmentSizeParameterT>;
  }
  else if constexpr (detail::params::is_per_segment_param_v<SegmentSizeParameterT>)
  {
    return segment_sizes.max_value;
  }
  else
  {
    return segment_sizes.value;
  }
}

// Smallest cluster width covering `max_seg_size`, capped at
// `hw_max_cluster_blocks`. Clamps first so a loose `numeric_limits<T>::max()`
// returns the cap instead of overflowing; whether the chosen tile actually
// covers the original bound is the dispatch's job to verify.
template <typename SegmentSizeValueT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr int
select_cluster_blocks(SegmentSizeValueT max_seg_size, int items_per_block, int hw_max_cluster_blocks) noexcept
{
  using value_t = SegmentSizeValueT;

  const auto max_tile = static_cast<value_t>(hw_max_cluster_blocks * items_per_block);
  const auto bounded  = ::cuda::std::min(max_seg_size, max_tile);
  return ::cuda::ceil_div(::cuda::narrow<int>(bounded), items_per_block);
}

// -----------------------------------------------------------------------------
// Kernel entry points
// -----------------------------------------------------------------------------
// Dynamic-cluster kernel for host launches; the agent reads the active cluster
// width via cooperative groups.
template <int ThreadsPerBlock,
          int ItemsPerThread,
          int BitsPerPass,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
__launch_bounds__(ThreadsPerBlock) _CCCL_KERNEL_ATTRIBUTES void device_segmented_topk_cluster_kernel(
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k_param,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments)
{
  using agent_t = agent_batched_topk_cluster<
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

  agent_t agent(
    temp_storage, d_key_segments_it, d_key_segments_out_it, segment_sizes, k_param, select_directions, num_segments);

  agent.Process();
}

#ifdef CUB_RDC_ENABLED
// CDP-only static-cluster kernel: compile-time `__cluster_dims__` so the
// triple-chevron launch from device code needs no `cudaFuncSetAttribute`.
template <int ThreadsPerBlock,
          int ItemsPerThread,
          int BitsPerPass,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
__launch_bounds__(ThreadsPerBlock) __cluster_dims__(max_portable_cluster_blocks, 1, 1)
  _CCCL_KERNEL_ATTRIBUTES void device_segmented_topk_cluster_kernel_static(
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k_param,
    SelectDirectionParameterT select_directions,
    NumSegmentsParameterT num_segments)
{
  using agent_t = agent_batched_topk_cluster<
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

  agent_t agent(
    temp_storage, d_key_segments_it, d_key_segments_out_it, segment_sizes, k_param, select_directions, num_segments);

  agent.Process();
}
#endif // CUB_RDC_ENABLED

// -----------------------------------------------------------------------------
// NVCC kernel-emission workaround
// -----------------------------------------------------------------------------
// NVCC `-rdc=false` only emits the device side of a templated `__global__`
// when it sees a triple-chevron in the same TU; CUDA 13's
// `-static-global-template-stub=true` makes the host stub internally linked
// on top. Address-of and `cudaLaunchKernelEx` are not enough. The host path
// must use `cudaLaunchKernelExC` for the cluster attribute, so instantiating
// `force_emit_kernel<Kernel>::emit` parses a (dead) chevron in its place.
// `Args` are deduced from the function-pointer type to avoid repeating the
// dispatch's template parameter list.
//
// See https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/.
template <auto Kernel>
struct force_emit_kernel;

template <typename... Args, void (*Kernel)(Args...)>
struct force_emit_kernel<Kernel>
{
  [[noreturn]] _CCCL_HOST static void emit(Args... args)
  {
    _CCCL_ASSERT(false, "force_emit_kernel::emit must never be called");
    // Unreachable; present only so NVCC emits `Kernel` for this TU.
    if (false)
    {
      Kernel<<<1, 1>>>(args...);
    }
    _CCCL_UNREACHABLE();
  }
};

// -----------------------------------------------------------------------------
// Dispatch
// -----------------------------------------------------------------------------
// Keys-only; every segment must fit in one cluster tile. Host picks
// `cluster_blocks` at runtime (capped by `cudaOccupancyMaxPotentialClusterSize`);
// CDP uses the static kernel at `max_portable_cluster_blocks`.

// CDP launch body, empty when CDP is disabled. Wrapped in a macro because
// `#ifdef` can't sit inside `NV_IF_TARGET`.
#ifndef CUB_RDC_ENABLED
#  define CUB_TOPK_CLUSTER_DEVICE_LAUNCH
#else // CUB_RDC_ENABLED
#  define CUB_TOPK_CLUSTER_DEVICE_LAUNCH                               \
    auto static_kernel = device_segmented_topk_cluster_kernel_static<  \
      ThreadsPerBlock,                                                 \
      ItemsPerThread,                                                  \
      BitsPerPass,                                                     \
      KeyInputItItT,                                                   \
      KeyOutputItItT,                                                  \
      SegmentSizeParameterT,                                           \
      KParameterT,                                                     \
      SelectDirectionParameterT,                                       \
      NumSegmentsParameterT>;                                          \
    if (const auto error = CubDebug(                                   \
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(       \
            static_cast<int>(grid_blocks), ThreadsPerBlock, 0, stream) \
            .doit(static_kernel,                                       \
                  d_key_segments_it,                                   \
                  d_key_segments_out_it,                               \
                  segment_sizes,                                       \
                  k_param,                                             \
                  select_directions,                                   \
                  num_segments)))                                      \
    {                                                                  \
      return error;                                                    \
    }
#endif // CUB_RDC_ENABLED

template <typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename TotalNumItemsGuaranteeT,
          typename PolicySelector = policy_selector>
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
  cudaStream_t stream                             = nullptr,
  [[maybe_unused]] PolicySelector policy_selector = {})
{
  // Clusters are SM 9.0+ only and the tuning is currently identical across
  // CCs, so pin the selector query to the minimum supported CC.
  constexpr cluster_topk_policy policy = PolicySelector{}(::cuda::compute_capability{9, 0});
  constexpr int ThreadsPerBlock        = policy.threads_per_block;
  constexpr int ItemsPerThread         = policy.items_per_thread;
  constexpr int BitsPerPass            = policy.bits_per_pass;

  constexpr int items_per_block  = ThreadsPerBlock * ItemsPerThread;
  constexpr int max_cluster_tile = max_supported_cluster_blocks * items_per_block;

  // Only static-size shapes can be rejected at compile time; runtime shapes
  // are re-checked below against the chosen tile.
  static_assert(!detail::params::is_static_param_v<SegmentSizeParameterT>
                  || detail::params::static_max_value_v<SegmentSizeParameterT> <= max_cluster_tile,
                "Static segment size exceeds max_supported_cluster_blocks * ThreadsPerBlock * ItemsPerThread.");

  const auto max_seg_size = runtime_max_segment_size(segment_sizes);

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

  using num_segments_val_t = typename NumSegmentsParameterT::value_type;
  const auto num_seg_val   = num_segments.get_param(num_segments_val_t{0});
  if (num_seg_val == 0)
  {
    return cudaSuccess;
  }

  // A zero bound would drive `clusterDim.x = 0`, which the runtime rejects.
  if (max_seg_size == 0)
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

  constexpr auto dynamic_kernel = &device_segmented_topk_cluster_kernel<
    ThreadsPerBlock,
    ItemsPerThread,
    BitsPerPass,
    KeyInputItItT,
    KeyOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT>;

  // Force NVCC to emit the device side of `dynamic_kernel` for this TU; see
  // `force_emit_kernel` above.
  [[maybe_unused]] constexpr auto force_emit = &force_emit_kernel<dynamic_kernel>::emit;

  NV_IF_TARGET(
    NV_IS_HOST,
    ({
      // Opt in to non-portable cluster widths (>8 on Hopper).
      if (const auto error = CubDebug(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(dynamic_kernel), cudaFuncAttributeNonPortableClusterSizeAllowed, 1)))
      {
        return error;
      }

      // Reused across the probe and the launch; `clusterDim.x` is a placeholder
      // until after `cudaOccupancyMaxPotentialClusterSize` (which ignores it).
      ::cudaLaunchAttribute cluster_attr{};
      cluster_attr.id               = ::cudaLaunchAttributeClusterDimension;
      cluster_attr.val.clusterDim.x = 1;
      cluster_attr.val.clusterDim.y = 1;
      cluster_attr.val.clusterDim.z = 1;

      ::cudaLaunchConfig_t cfg{};
      cfg.gridDim          = dim3(1, 1, 1);
      cfg.blockDim         = dim3(static_cast<unsigned int>(ThreadsPerBlock), 1, 1);
      cfg.dynamicSmemBytes = 0;
      cfg.stream           = stream;
      cfg.attrs            = &cluster_attr;
      cfg.numAttrs         = 1;

      int hw_max_cluster_blocks = 0;
      if (const auto error = CubDebug(cudaOccupancyMaxPotentialClusterSize(
            &hw_max_cluster_blocks, reinterpret_cast<const void*>(dynamic_kernel), &cfg)))
      {
        return error;
      }
      hw_max_cluster_blocks = ::cuda::std::min(hw_max_cluster_blocks, max_supported_cluster_blocks);

      const int cluster_blocks = select_cluster_blocks(max_seg_size, items_per_block, hw_max_cluster_blocks);

      // Exact bounds must fit the chosen tile; `per_segment_param` may carry
      // a loose upper bound and is left to the agent's contract.
      if constexpr (!detail::params::is_per_segment_param_v<SegmentSizeParameterT>)
      {
        if (max_seg_size > static_cast<decltype(max_seg_size)>(cluster_blocks) * items_per_block)
        {
          return cudaErrorInvalidValue;
        }
      }

      const auto grid_blocks =
        static_cast<::cuda::std::uint64_t>(num_seg_val) * static_cast<::cuda::std::uint64_t>(cluster_blocks);
      if (grid_blocks > static_cast<::cuda::std::uint64_t>(::cuda::std::numeric_limits<int>::max()))
      {
        return cudaErrorInvalidValue;
      }

      cfg.gridDim                   = dim3(static_cast<unsigned int>(grid_blocks), 1, 1);
      cluster_attr.val.clusterDim.x = static_cast<unsigned int>(cluster_blocks);

      if (const auto error = CubDebug(::cudaLaunchKernelEx(
            &cfg,
            dynamic_kernel,
            d_key_segments_it,
            d_key_segments_out_it,
            segment_sizes,
            k_param,
            select_directions,
            num_segments)))
      {
        return error;
      }
    }),
    ({
      // CDP path: same exact-vs-loose rule against the fixed static tile.
      constexpr int static_cluster_tile = max_portable_cluster_blocks * items_per_block;
      if constexpr (!detail::params::is_per_segment_param_v<SegmentSizeParameterT>)
      {
        if (max_seg_size > static_cast<decltype(max_seg_size)>(static_cluster_tile))
        {
          return cudaErrorInvalidValue;
        }
      }

      const auto grid_blocks = static_cast<::cuda::std::uint64_t>(num_seg_val)
                             * static_cast<::cuda::std::uint64_t>(max_portable_cluster_blocks);
      if (grid_blocks > static_cast<::cuda::std::uint64_t>(::cuda::std::numeric_limits<int>::max()))
      {
        return cudaErrorInvalidValue;
      }

      CUB_TOPK_CLUSTER_DEVICE_LAUNCH
    }));

  // Cluster launches can fail on the device while reporting success; sync.
  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  return CubDebug(detail::DebugSyncStream(stream));
}

#undef CUB_TOPK_CLUSTER_DEVICE_LAUNCH
} // namespace detail::batched_topk_cluster

CUB_NAMESPACE_END
