// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Kernel entry point for device-wide batched top-k.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_batched_topk.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh>
#include <cub/util_arch.cuh>

#include <cuda/__device/arch_id.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// Finds the smallest policy whose tile size still covers the given max segment size. Returns -1 if none covers.
[[nodiscard]] _CCCL_API constexpr int
find_smallest_covering_policy_index(const batched_topk_policy& p, ::cuda::std::int64_t max_segment_size)
{
  int result = -1;
  for (int i = 0; i < static_cast<int>(p.worker_per_segment_policies.size()); ++i)
  {
    const auto& wp                       = p.worker_per_segment_policies[i];
    const ::cuda::std::int64_t tile_size = ::cuda::std::int64_t{wp.block_threads} * wp.items_per_thread;
    if (tile_size >= max_segment_size)
    {
      result = i;
    }
  }
  return result;
}

// Given a policy_selector and a segment-size parameter, resolves the agent type to be instantiated by the kernel.
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
struct find_smallest_covering_policy
{
private:
  static constexpr ::cuda::std::int64_t max_segment_size = params::static_max_value_v<SegmentSizeParameterT>;
  static constexpr batched_topk_policy active_policy     = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10});
  static constexpr int selected_index = find_smallest_covering_policy_index(active_policy, max_segment_size);

public:
  static constexpr bool supports_one_worker_per_segment = selected_index >= 0;
  // Use a safe index to avoid out-of-bounds constexpr evaluation when no policy covers. The kernel body's static_assert
  // rejects this case.
  static constexpr agent_batched_topk_policy policy =
    active_policy.worker_per_segment_policies[selected_index < 0 ? 0 : selected_index];

  using agent_t =
    agent_batched_topk_worker_per_segment<agent_batched_topk_worker_per_segment_policy<policy.block_threads,
                                                                                       policy.items_per_thread,
                                                                                       policy.load_algorithm,
                                                                                       policy.store_algorithm>,
                                          AgentParamsT...>;
};

// -----------------------------------------------------------------------------
// Global Kernel Entry Point
// -----------------------------------------------------------------------------
template <typename PolicySelector,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
#if _CCCL_HAS_CONCEPTS()
  requires batched_topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(
  find_smallest_covering_policy<
    PolicySelector,
    SegmentSizeParameterT,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT>::policy.block_threads)) __global__
  void device_segmented_topk_kernel(
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    ValueInputItItT d_value_segments_it,
    ValueOutputItItT d_value_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k,
    SelectDirectionParameterT select_directions,
    NumSegmentsParameterT num_segments)
{
  using find_smallest_covering_policy_t = find_smallest_covering_policy<
    PolicySelector,
    SegmentSizeParameterT,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT>;
  static_assert(find_smallest_covering_policy_t::supports_one_worker_per_segment,
                "No valid policy found for one-worker-per-segment approach");

  using agent_t = typename find_smallest_covering_policy_t::agent_t;

  // Static Assertions (Constraints)
  static_assert(agent_t::tile_size >= params::static_max_value_v<SegmentSizeParameterT>,
                "Block size exceeds maximum segment size supported by SegmentSizeParameterT");
  static_assert(sizeof(typename agent_t::TempStorage) <= max_smem_per_block,
                "Static shared memory per block must not exceed 48KB limit.");

  // Temporary storage allocation
  __shared__ typename agent_t::TempStorage temp_storage;

  // Instantiate agent
  agent_t agent(
    temp_storage,
    d_key_segments_it,
    d_key_segments_out_it,
    d_value_segments_it,
    d_value_segments_out_it,
    segment_sizes,
    k,
    select_directions,
    num_segments);

  // Process segments
  agent.Process();
}
} // namespace detail::batched_topk

CUB_NAMESPACE_END
