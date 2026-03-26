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

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>

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
struct find_first_smem_fitting_policy_impl;

// Base case: End of policy chain reached: If we reach Index == Count, it means we checked all with no match
template <typename PoliciesT,
          ::cuda::std::int64_t Count,
          template <typename...> class WorkerPerSegmentAgentT,
          typename... AgentParamsT>
struct find_first_smem_fitting_policy_impl<PoliciesT, Count, Count, WorkerPerSegmentAgentT, AgentParamsT...>
{
  using policy_t                         = void;
  static constexpr bool has_valid_policy = false;
};

// Policies are ordered by decreasing tile size. This finds the first (i.e., largest) policy whose agent
// TempStorage fits within the static shared memory limit (max_smem_per_block, typically 48KB).
// This is useful to figure out which segments (given a runtime segment size) can still be addressed with a
// one-worker-per-segment approach.
template <typename PoliciesT,
          ::cuda::std::int64_t Index,
          ::cuda::std::int64_t Count,
          template <typename...> class WorkerPerSegmentAgentT,
          typename... AgentParamsT>
struct find_first_smem_fitting_policy_impl
{
  // Inspect the current policy
  using current_policy_t = ::cuda::std::tuple_element_t<Index, PoliciesT>;

  // Instantiate agent to check temporary storage size
  using current_agent_t      = WorkerPerSegmentAgentT<current_policy_t, AgentParamsT...>;
  static constexpr bool fits = (sizeof(typename current_agent_t::TempStorage) <= max_smem_per_block);

  // The 'next' policy in the chain
  using next_step =
    find_first_smem_fitting_policy_impl<PoliciesT, Index + 1, Count, WorkerPerSegmentAgentT, AgentParamsT...>;

  // Select result:
  // If 'fits' is true, we stop here.
  // If 'fits' is false, we take the result from 'next_step'.
  using policy_t = ::cuda::std::conditional_t<fits, current_policy_t, typename next_step::policy_t>;

  // Whether there's a valid policy that we can instantiate the agent with such that the agent's shared memory doesn't
  // exceed the static shared memory limimt
  static constexpr bool has_valid_policy = fits ? true : next_step::has_valid_policy;
};

// Policies are ordered by decreasing tile size. This finds the last (i.e., smallest) policy whose tile size
// is still large enough to cover the user-provided upper bound on segment size (tile_size >= MaxSegmentSize).
template <typename PoliciesT, ::cuda::std::int64_t Index, ::cuda::std::int64_t Count, int MaxSegmentSize>
struct find_smallest_covering_policy_impl
{
  using current_policy_t = ::cuda::std::tuple_element_t<Index, PoliciesT>;

  // Calculate the capacity of the current policy
  static constexpr int tile_size = current_policy_t::block_threads * current_policy_t::items_per_thread;

  // Does this policy still cover our segment?
  static constexpr bool covers = (tile_size >= MaxSegmentSize);

  // Peek at the next policy
  using next_step = find_smallest_covering_policy_impl<PoliciesT, Index + 1, Count, MaxSegmentSize>;

  // Selection Logic:
  // If the current policy covers the segment, we check if the next one also still does.
  // We want the last one that returns true for 'covers'.
  using policy_t =
    ::cuda::std::conditional_t<next_step::has_valid_policy,
                               typename next_step::policy_t, // The next one is even tighter and still works
                               ::cuda::std::conditional_t<covers, current_policy_t, void> // This is the tightest valid
                                                                                          // one
                               >;

  static constexpr bool has_valid_policy = covers;
};

// Base case: end of tuple
template <typename PoliciesT, ::cuda::std::int64_t Count, int MaxSegmentSize>
struct find_smallest_covering_policy_impl<PoliciesT, Count, Count, MaxSegmentSize>
{
  using policy_t                         = void;
  static constexpr bool has_valid_policy = false;
};

template <typename SegmentedTopKPolicy,
          typename SegmentSizeParameterT,
          template <typename...> class WorkerPerSegmentAgentT,
          typename... AgentParamsT>
struct find_smallest_covering_policy
{
  using worker_per_segment_policies     = typename SegmentedTopKPolicy::worker_per_segment_policies;
  static constexpr int max_segment_size = params::static_max_value_v<SegmentSizeParameterT>;

  // Finds the smallest policy whose tile size still covers the upper bound of the segment size.
  // Since the policy list is ordered by decreasing tile size, this is the last policy where
  // tile_size >= max_segment_size. A smaller tile means less shared memory, so a covering policy
  // found here is guaranteed to also fit within the static shared memory limit.
  using segment_optimized_impl =
    find_smallest_covering_policy_impl<worker_per_segment_policies,
                                       0,
                                       ::cuda::std::tuple_size<worker_per_segment_policies>::value,
                                       max_segment_size>;
  static constexpr bool supports_one_worker_per_segment = segment_optimized_impl::has_valid_policy;

  using worker_per_segment_policy_t =
    ::cuda::std::conditional_t<supports_one_worker_per_segment, typename segment_optimized_impl::policy_t, void>;

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
  find_smallest_covering_policy<
    typename ChainedPolicyT::ActivePolicy,
    SegmentSizeParameterT,
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

  using find_smallest_covering_policy_t = find_smallest_covering_policy<
    active_policy_t,
    SegmentSizeParameterT,
    agent_batched_topk_worker_per_segment,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT>;

  using agent_t = typename find_smallest_covering_policy_t::worker_per_segment_agent_t;
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
