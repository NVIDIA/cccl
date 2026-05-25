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

#include <cuda/__device/compute_capability.h>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// Given a policy_selector and a segment-size parameter, resolves the agent type to be instantiated by the kernel.
// Selects the smallest policy whose tile size still covers the upper bound on segment size AND whose instantiated
// agent's shared memory usage fits within the static shared memory limit (max_smem_per_block).
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
struct find_smallest_covering_policy
{
private:
  static constexpr ::cuda::std::int64_t max_segment_size = params::static_max_value_v<SegmentSizeParameterT>;
  static constexpr batched_topk_policy active_policy     = current_policy<PolicySelector>();

  template <int Index>
  [[nodiscard]] static constexpr int find_index()
  {
    if constexpr (Index >= active_policy.worker_per_segment_policies.size())
    {
      return -1;
    }
    else
    {
      constexpr worker_policy wp = active_policy.worker_per_segment_policies[Index];
      constexpr auto tile_size   = ::cuda::std::int64_t{wp.block_threads} * wp.items_per_thread;

      struct policy_getter_17 // TODO(bgruber): drop this in C++17 and pass wp directly
      {
        _CCCL_API constexpr auto operator()() const
        {
          return active_policy.worker_per_segment_policies[Index];
        }
      };
      using candidate_agent_t  = agent_batched_topk_worker_per_segment<policy_getter_17, AgentParamsT...>;
      constexpr bool covers    = tile_size >= max_segment_size;
      constexpr bool fits_smem = sizeof(typename candidate_agent_t::TempStorage) <= max_smem_per_block;
      constexpr int next       = find_index<Index + 1>();
      if constexpr (covers && fits_smem)
      {
        return next >= 0 ? next : Index;
      }
      else
      {
        return next;
      }
    }
  }

  static constexpr int selected_index = find_index<0>();

public:
  // TODO (elstehle): extend support for variable-size segments
  static_assert(selected_index >= 0, "No valid policy found for one-worker-per-segment approach");
  static constexpr worker_policy policy = active_policy.worker_per_segment_policies[selected_index];

  struct policy_getter_17 // TODO(bgruber): drop this in C++17 and pass policy directly
  {
    _CCCL_API constexpr auto operator()() const
    {
      return policy;
    }
  };
  using agent_t = agent_batched_topk_worker_per_segment<policy_getter_17, AgentParamsT...>;
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
  using agent_t = typename find_smallest_covering_policy<
    PolicySelector,
    SegmentSizeParameterT,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT>::agent_t;

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
