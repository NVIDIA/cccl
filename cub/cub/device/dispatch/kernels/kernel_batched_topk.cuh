// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved. SPDX-License-Identifier:
// Apache-2.0 WITH LLVM-exception
//!
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
#include <cub/util_arch.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// -----------------------------------------------------------------------------
// One-worker-per-segment policy selection
// -----------------------------------------------------------------------------
template <typename PoliciesT,
          ::cuda::std::int64_t Index,
          ::cuda::std::int64_t Count,
          template <typename...> class WorkerPerSegmentAgentT,
          typename... AgentParamsT>
struct find_valid_policy_impl;

// Base case: End of policy chain reached: If we reach Index == Count, it means we checked all with no match
template <typename PoliciesT,
          ::cuda::std::int64_t Count,
          template <typename...> class WorkerPerSegmentAgentT,
          typename... AgentParamsT>
struct find_valid_policy_impl<PoliciesT, Count, Count, WorkerPerSegmentAgentT, AgentParamsT...>
{
  using policy_t                         = void;
  static constexpr bool has_valid_policy = false;
};

template <typename PoliciesT,
          ::cuda::std::int64_t Index,
          ::cuda::std::int64_t Count,
          template <typename...> class WorkerPerSegmentAgentT,
          typename... AgentParamsT>
struct find_valid_policy_impl
{
  // Inspect the current policy
  using current_policy_t = ::cuda::std::tuple_element_t<Index, PoliciesT>;

  // Instantiate agent to check temporary storage size
  using current_agent_t      = WorkerPerSegmentAgentT<current_policy_t, AgentParamsT...>;
  static constexpr bool fits = (sizeof(typename current_agent_t::TempStorage) <= max_smem_per_block);

  // The 'next' policy in the chain
  using next_step = find_valid_policy_impl<PoliciesT, Index + 1, Count, WorkerPerSegmentAgentT, AgentParamsT...>;

  // Select result:
  // If 'fits' is true, we stop here.
  // If 'fits' is false, we take the result from 'next_step'.
  using policy_t = ::cuda::std::conditional_t<fits, current_policy_t, typename next_step::policy_t>;

  // Whether there's a valid policy that we can instantiate the agent with such that the agent's shared memory doesn't
  // exceed the static shared memory limimt
  static constexpr bool has_valid_policy = fits ? true : next_step::has_valid_policy;
};

template <typename SegmentedTopKPolicy, template <typename...> class WorkerPerSegmentAgentT, typename... AgentParamsT>
struct find_valid_policy
{
  // The list of policies for the one-worker-per-segment approach
  using worker_per_segment_policies = typename SegmentedTopKPolicy::worker_per_segment_policies;

  // Helper to find a valid policy that we can successfully instantiate the agent with
  using find_valid_policy_impl_t =
    find_valid_policy_impl<worker_per_segment_policies,
                           0,
                           ::cuda::std::tuple_size<worker_per_segment_policies>::value,
                           WorkerPerSegmentAgentT,
                           AgentParamsT...>;

  // Whether there's a valid policy for one-worker-per-segment approach
  static constexpr bool supports_one_worker_per_segment = find_valid_policy_impl_t::has_valid_policy;

  // Policy selected for one-worker-per-segment approach, if there is a valid policy
  using worker_per_segment_policy_t = typename find_valid_policy_impl_t::policy_t;

  // Agent for the one-worker-per-segment approach, if there is a valid policy
  using worker_per_segment_agent_t =
    ::cuda::std::conditional_t<supports_one_worker_per_segment,
                               WorkerPerSegmentAgentT<worker_per_segment_policy_t, AgentParamsT...>,
                               void>;
};

// -----------------------------------------------------------------------------
// Global Kernel Entry Point
// -----------------------------------------------------------------------------
template <typename ChainedPolicyT,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
__launch_bounds__(int(
  find_valid_policy<typename ChainedPolicyT::ActivePolicy,
                    agent_batched_topk_worker_per_segment,
                    KeyInputItItT,
                    KeyOutputItItT,
                    ValueInputItItT,
                    ValueOutputItItT,
                    SegmentSizeParameterT,
                    KParameterT,
                    SelectDirectionParameterT,
                    NumSegmentsParameterT>::worker_per_segment_policy_t::block_threads)) __global__
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
  using active_policy_t = typename ChainedPolicyT::ActivePolicy;

  using find_valid_policy_t = find_valid_policy<
    active_policy_t,
    agent_batched_topk_worker_per_segment,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT>;

  using agent_t = typename find_valid_policy_t::worker_per_segment_agent_t;
  static_assert(!::cuda::std::is_same_v<agent_t, void>, "No valid policy found for one-worker-per-segment approach");

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
