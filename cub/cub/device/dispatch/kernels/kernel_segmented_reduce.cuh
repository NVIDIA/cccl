// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_reduce.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce.cuh> // finalize_and_store_aggregate
#include <cub/device/dispatch/tuning/tuning_segmented_reduce.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>

#include <cuda/__device/arch_id.h>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_reduce
{
/// Normalize input iterator to segment offset
template <typename T, typename OffsetT, typename IteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void NormalizeReductionOutput(T& /*val*/, OffsetT /*base_offset*/, IteratorT /*itr*/)
{}

/// Normalize input iterator to segment offset (specialized for arg-index)
template <typename KeyValuePairT, typename OffsetT, typename WrappedIteratorT, typename OutputValueT>
_CCCL_DEVICE _CCCL_FORCEINLINE void NormalizeReductionOutput(
  KeyValuePairT& val, OffsetT base_offset, ArgIndexInputIterator<WrappedIteratorT, OffsetT, OutputValueT> /*itr*/)
{
  val.key -= base_offset;
}

/**
 * Segmented reduction (one block per segment)
 * @tparam PolicySelector
 *   Policy selector
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
 *
 * @tparam BeginOffsetIteratorT
 *   Random-access input iterator type for reading segment beginning offsets
 *   @iterator
 *
 * @tparam EndOffsetIteratorT
 *   Random-access input iterator type for reading segment ending offsets
 *   @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `T operator()(const T &a, const U &b)`
 *
 * @tparam InitT
 *   Initial value type
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out
 *   Pointer to the output aggregate
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of
 *   length `num_segments`, such that `d_begin_offsets[i]` is the first element
 *   of the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length
 *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of
 *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
 *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is
 *   considered empty.
 *
 * @param[in] num_segments
 *   The number of segments on which the reduction is performed
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 *
 * @param[in] init
 *   The initial value of the reduction
 *
 * @param[in] num_segments
 *   The number of segments on which the reduction is performed
 *
 * @param[in] max_segment_size
 *   Maximum segment size guarantee
 */
template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_reduce_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(int(PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).segmented_reduce.block_threads)) //
  void DeviceSegmentedReduceKernel(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    int num_segments,
    ReductionOpT reduction_op,
    InitT init,
    size_t max_segment_size)
{
  static constexpr segmented_reduce_policy full_policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10});

  // Large segment agent (one block per segment)
  static constexpr reduce::agent_reduce_policy large_pol = full_policy.segmented_reduce;
  using large_agent_policy_t =
    AgentReducePolicy<large_pol.block_threads,
                      large_pol.items_per_thread,
                      AccumT,
                      large_pol.vector_load_length,
                      large_pol.block_algorithm,
                      large_pol.load_modifier,
                      NoScaling<large_pol.block_threads, large_pol.items_per_thread, AccumT>>;
  using AgentReduceT = reduce::AgentReduce<large_agent_policy_t, InputIteratorT, OffsetT, ReductionOpT, AccumT>;

  // Medium segment agent (one warp per segment)
  static constexpr agent_warp_reduce_policy med_pol = full_policy.medium_reduce;
  using medium_agent_policy_t =
    AgentWarpReducePolicy<med_pol.block_threads,
                          med_pol.warp_threads,
                          med_pol.items_per_thread,
                          AccumT,
                          med_pol.vector_load_length,
                          med_pol.load_modifier>;
  using AgentMediumReduceT =
    reduce::AgentWarpReduce<medium_agent_policy_t, InputIteratorT, OffsetT, ReductionOpT, AccumT>;

  // Small segment agent (one thread per segment)
  static constexpr agent_warp_reduce_policy small_pol = full_policy.small_reduce;
  using small_agent_policy_t =
    AgentWarpReducePolicy<small_pol.block_threads,
                          small_pol.warp_threads,
                          small_pol.items_per_thread,
                          AccumT,
                          small_pol.vector_load_length,
                          small_pol.load_modifier>;
  using AgentSmallReduceT =
    reduce::AgentWarpReduce<small_agent_policy_t, InputIteratorT, OffsetT, ReductionOpT, AccumT>;

  constexpr int small_items_per_tile  = small_pol.items_per_tile();
  constexpr int medium_items_per_tile = med_pol.items_per_tile();

  constexpr int segments_per_small_block  = small_pol.segments_per_block();
  constexpr int small_threads_per_warp    = small_pol.warp_threads;
  constexpr int segments_per_medium_block = med_pol.segments_per_block();
  constexpr int medium_threads_per_warp   = med_pol.warp_threads;

  // Shared memory storage
  __shared__ union
  {
    typename AgentReduceT::TempStorage large_storage;
    typename AgentMediumReduceT::TempStorage medium_storage[segments_per_medium_block];
    typename AgentSmallReduceT::TempStorage small_storage[segments_per_small_block];
  } temp_storage;

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  auto small_medium_seg_reduction =
    [&](auto agent_tag, auto& storage, auto threads_per_warp_tag, auto segments_per_block_tag) {
      using AgentWarpReduceT           = typename decltype(agent_tag)::type;
      constexpr int threads_per_warp   = decltype(threads_per_warp_tag)::value;
      constexpr int segments_per_block = decltype(segments_per_block_tag)::value;
      const int sid_within_block       = tid / threads_per_warp;
      const int lane_id                = tid % threads_per_warp;
      const int global_segment_id      = bid * segments_per_block + sid_within_block;

      if (global_segment_id < num_segments)
      {
        const auto segment_begin = static_cast<OffsetT>(d_begin_offsets[global_segment_id]);
        const auto segment_end   = static_cast<OffsetT>(d_end_offsets[global_segment_id]);

        if (segment_begin == segment_end)
        {
          if (lane_id == 0)
          {
            *(d_out + global_segment_id) = reduce::unwrap_empty_problem_init(init);
          }
          return;
        }

        AccumT warp_aggregate =
          AgentWarpReduceT(storage[sid_within_block], d_in, reduction_op).ConsumeRange(segment_begin, segment_end);

        NormalizeReductionOutput(warp_aggregate, segment_begin, d_in);

        if (lane_id == 0)
        {
          reduce::finalize_and_store_aggregate(d_out + global_segment_id, reduction_op, init, warp_aggregate);
        }
      }
    };

  if (max_segment_size != 0 && max_segment_size <= static_cast<size_t>(small_items_per_tile))
  {
    small_medium_seg_reduction(
      ::cuda::std::type_identity<AgentSmallReduceT>{},
      temp_storage.small_storage,
      ::cuda::std::integral_constant<int, small_threads_per_warp>{},
      ::cuda::std::integral_constant<int, segments_per_small_block>{});
  }
  else if (max_segment_size != 0 && max_segment_size <= static_cast<size_t>(medium_items_per_tile))
  {
    small_medium_seg_reduction(
      ::cuda::std::type_identity<AgentMediumReduceT>{},
      temp_storage.medium_storage,
      ::cuda::std::integral_constant<int, medium_threads_per_warp>{},
      ::cuda::std::integral_constant<int, segments_per_medium_block>{});
  }
  else
  {
    OffsetT segment_begin = d_begin_offsets[bid];
    OffsetT segment_end   = d_end_offsets[bid];

    if (segment_begin == segment_end)
    {
      if (tid == 0)
      {
        *(d_out + bid) = reduce::unwrap_empty_problem_init(init);
      }
      return;
    }

    AccumT block_aggregate =
      AgentReduceT(temp_storage.large_storage, d_in, reduction_op).ConsumeRange(segment_begin, segment_end);

    NormalizeReductionOutput(block_aggregate, segment_begin, d_in);

    if (tid == 0)
    {
      reduce::finalize_and_store_aggregate(d_out + bid, reduction_op, init, block_aggregate);
    }
  }
}

/**
 * Fixed Segment Size Segmented reduction
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `T operator()(const T &a, const U &b)`
 *
 * @tparam InitT
 *   Initial value type
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out
 *   Pointer to the output aggregate
 *
 * @param[in] segment_size
 *   The fixed segment size of each the segments
 *
 * @param[in] num_segments
 *   The number of segments on which the reduction is performed
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 *
 * @param[in] init
 *   The initial value of the reduction
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT>
CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ReducePolicy::BLOCK_THREADS)) void DeviceFixedSizeSegmentedReduceKernel(
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT segment_size,
  int num_segments,
  ReductionOpT reduction_op,
  InitT init)
{
  using ActivePolicyT = typename ChainedPolicyT::ActivePolicy;

  // Thread block type for reducing input tiles
  using AgentReduceT =
    reduce::AgentReduce<typename ActivePolicyT::ReducePolicy, InputIteratorT, int, ReductionOpT, AccumT>;

  using AgentMediumReduceT =
    reduce::AgentWarpReduce<typename ActivePolicyT::MediumReducePolicy, InputIteratorT, int, ReductionOpT, AccumT>;

  using AgentSmallReduceT =
    reduce::AgentWarpReduce<typename ActivePolicyT::SmallReducePolicy, InputIteratorT, int, ReductionOpT, AccumT>;

  constexpr auto segments_per_medium_block = ActivePolicyT::MediumReducePolicy::SEGMENTS_PER_BLOCK;
  constexpr auto medium_threads_per_warp   = ActivePolicyT::MediumReducePolicy::WARP_THREADS;
  constexpr auto medium_items_per_tile     = ActivePolicyT::MediumReducePolicy::ITEMS_PER_TILE;

  constexpr auto segments_per_small_block = ActivePolicyT::SmallReducePolicy::SEGMENTS_PER_BLOCK;
  constexpr auto small_threads_per_warp   = ActivePolicyT::SmallReducePolicy::WARP_THREADS;
  constexpr auto small_items_per_tile     = ActivePolicyT::SmallReducePolicy::ITEMS_PER_TILE;

  // Shared memory storage
  __shared__ union
  {
    typename AgentReduceT::TempStorage large_storage;
    typename AgentMediumReduceT::TempStorage medium_storage[segments_per_medium_block];
    typename AgentSmallReduceT::TempStorage small_storage[segments_per_small_block];
  } temp_storage;

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  if (segment_size <= small_items_per_tile)
  {
    const int sid_within_block  = tid / small_threads_per_warp;
    const int lane_id           = tid % small_threads_per_warp;
    const int global_segment_id = bid * segments_per_small_block + sid_within_block;

    const auto segment_begin = static_cast<::cuda::std::int64_t>(global_segment_id) * segment_size;

    if (global_segment_id < num_segments)
    {
      // If empty segment, write out the initial value
      if (segment_size == 0)
      {
        if (lane_id == 0)
        {
          *(d_out + global_segment_id) = detail::reduce::unwrap_empty_problem_init(init);
        }
        return;
      }
      // Consume input tiles
      AccumT warp_aggregate =
        AgentSmallReduceT(temp_storage.small_storage[sid_within_block], d_in + segment_begin, reduction_op)
          .ConsumeRange({}, static_cast<int>(segment_size));

      if (lane_id == 0)
      {
        reduce::finalize_and_store_aggregate(d_out + global_segment_id, reduction_op, init, warp_aggregate);
      }
    }
  }
  else if (segment_size <= medium_items_per_tile)
  {
    const int sid_within_block  = tid / medium_threads_per_warp;
    const int lane_id           = tid % medium_threads_per_warp;
    const int global_segment_id = bid * segments_per_medium_block + sid_within_block;

    const auto segment_begin = static_cast<::cuda::std::int64_t>(global_segment_id) * segment_size;

    if (global_segment_id < num_segments)
    {
      // Consume input tiles
      AccumT warp_aggregate =
        AgentMediumReduceT(temp_storage.medium_storage[sid_within_block], d_in + segment_begin, reduction_op)
          .ConsumeRange({}, static_cast<int>(segment_size));

      if (lane_id == 0)
      {
        reduce::finalize_and_store_aggregate(d_out + global_segment_id, reduction_op, init, warp_aggregate);
      }
    }
  }
  else
  {
    const auto segment_begin = static_cast<::cuda::std::int64_t>(bid) * segment_size;

    // Consume input tiles
    AccumT block_aggregate = AgentReduceT(temp_storage.large_storage, d_in + segment_begin, reduction_op)
                               .ConsumeRange({}, static_cast<int>(segment_size));

    if (tid == 0)
    {
      reduce::finalize_and_store_aggregate(d_out + bid, reduction_op, init, block_aggregate);
    }
  }
}
} // namespace detail::segmented_reduce

CUB_NAMESPACE_END
