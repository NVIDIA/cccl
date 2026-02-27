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
    ReductionOpT reduction_op,
    InitT init)
{
  static constexpr reduce::agent_reduce_policy policy =
    PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).segmented_reduce;
  // TODO(bgruber): pass policy directly as template argument to AgentReduce in C++20
  using agent_policy_t =
    AgentReducePolicy<policy.block_threads,
                      policy.items_per_thread,
                      AccumT,
                      policy.vector_load_length,
                      policy.block_algorithm,
                      policy.load_modifier,
                      NoScaling<policy.block_threads, policy.items_per_thread, AccumT>>;

  // Thread block type for reducing input tiles
  using AgentReduceT = reduce::AgentReduce<agent_policy_t, InputIteratorT, OffsetT, ReductionOpT, AccumT>;

  // Shared memory storage
  __shared__ typename AgentReduceT::TempStorage temp_storage;

  OffsetT segment_begin = d_begin_offsets[blockIdx.x];
  OffsetT segment_end   = d_end_offsets[blockIdx.x];

  // Check if empty problem
  if (segment_begin == segment_end)
  {
    if (threadIdx.x == 0)
    {
      *(d_out + blockIdx.x) = init;
    }
    return;
  }

  // Consume input tiles
  AccumT block_aggregate = AgentReduceT(temp_storage, d_in, reduction_op).ConsumeRange(segment_begin, segment_end);

  // Normalize as needed
  NormalizeReductionOutput(block_aggregate, segment_begin, d_in);

  if (threadIdx.x == 0)
  {
    reduce::finalize_and_store_aggregate(d_out + blockIdx.x, reduction_op, init, block_aggregate);
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
 *
 * @param[out] d_partial_out
 *  Pointer to store partial aggregates in two-phase reduction
 *
 * @param[in] full_chunk_size
 *   The full chunk size processed by each block in two-phase reduction
 *
 * @param[in] blocks_per_segment
 *   The number of blocks to be used for reducing each segment in two-phase reduction
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
  InitT init,
  AccumT* d_partial_out,
  int full_chunk_size,
  int blocks_per_segment)
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
    if (d_partial_out != nullptr) // two-phase reduction with partial aggregates
    {
      const auto chunk_id             = bid % blocks_per_segment;
      const bool is_last_chunk        = chunk_id == (blocks_per_segment - 1);
      const bool has_incomplete_chunk = (segment_size % full_chunk_size != 0);

      // If the last chunk is incomplete, only process the valid portion of the segment
      const auto chunk_size =
        (has_incomplete_chunk && is_last_chunk) ? (segment_size % full_chunk_size) : full_chunk_size;

      const auto segment_id    = bid / blocks_per_segment;
      const auto segment_begin = static_cast<::cuda::std::int64_t>(segment_id) * segment_size;

      const auto chunk_offset = chunk_id * full_chunk_size;
      const auto chunk_begin  = segment_begin + chunk_offset;

      AccumT block_aggregate =
        AgentReduceT(temp_storage.large_storage, d_in + chunk_begin, reduction_op).ConsumeRange({}, chunk_size);
      if (tid == 0)
      {
        *(d_partial_out + bid) = block_aggregate;
      }
    }
    else // single-phase reduction with direct write-out of final aggregate
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
}
} // namespace detail::segmented_reduce

CUB_NAMESPACE_END
