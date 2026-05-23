// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_batched_topk_atomic.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/util_device.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda_runtime.h>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk_atomic
{
inline constexpr int default_threads_per_block = 256;

template <int ThreadsPerBlock,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
__launch_bounds__(ThreadsPerBlock) _CCCL_KERNEL_ATTRIBUTES void device_segmented_topk_atomic_kernel(
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k_param,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments)
{
  using agent_t = agent_batched_topk_atomic<
    ThreadsPerBlock,
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

template <int ThreadsPerBlock = default_threads_per_block,
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

  using num_segments_val_t    = typename NumSegmentsParameterT::value_type;
  const auto num_seg_val      = num_segments.get_param(num_segments_val_t{0});
  const auto num_seg_unsigned = static_cast<unsigned long long>(num_seg_val);
  if (num_seg_unsigned == 0)
  {
    return cudaSuccess;
  }

  int sm_version = 0;
  if (const auto error = CubDebug(SmVersionUncached(sm_version)))
  {
    return error;
  }
  if (sm_version < 900)
  {
    return cudaErrorNotSupported;
  }

  const auto grid_blocks = static_cast<unsigned long long>(num_seg_unsigned);

  auto kernel = device_segmented_topk_atomic_kernel<
    ThreadsPerBlock,
    KeyInputItItT,
    KeyOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT>;

  if (const auto error = CubDebug(
        THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
          static_cast<unsigned int>(grid_blocks), static_cast<unsigned int>(ThreadsPerBlock), 0, stream)
          .doit(
            kernel, d_key_segments_it, d_key_segments_out_it, segment_sizes, k_param, select_directions, num_segments)))
  {
    return error;
  }

  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  return CubDebug(detail::DebugSyncStream(stream));
}
} // namespace detail::batched_topk_atomic

CUB_NAMESPACE_END
