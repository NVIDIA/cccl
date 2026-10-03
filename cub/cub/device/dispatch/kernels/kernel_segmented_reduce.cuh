// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/util_arch.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/__functional/operator_properties.h>
#include <cuda/__type_traits/vector_type.h>
#include <cuda/__utility/in_range.h>
#include <cuda/atomic>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cccl/cuda_capabilities.h>
#include <cuda/std/__iterator/iterator_traits.h>

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
 * @tparam InitValueT
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
          typename InitValueT,
          typename AccumT>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_reduce_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(current_policy<PolicySelector>().large_reduce.threads_per_block) //
  void DeviceSegmentedReduceKernel(
    const InputIteratorT d_in,
    const OutputIteratorT d_out,
    const BeginOffsetIteratorT d_begin_offsets,
    const EndOffsetIteratorT d_end_offsets,
    const int num_segments,
    ReductionOpT reduction_op,
    const InitValueT init,
    const size_t max_segment_size)
{
  static constexpr SegmentedReducePolicy full_policy = current_policy<PolicySelector>();

  // Large segment agent (one block per segment)
  static constexpr ReducePassPolicy large_pol = full_policy.large_reduce;

  // TODO(bgruber): pass policy directly as template argument to AgentReduce in C++20
  using large_agent_policy_t =
    agent_reduce_policy<0,
                        0,
                        void,
                        large_pol.vec_size,
                        large_pol.reduce_algorithm,
                        large_pol.load_modifier,
                        NoScaling<large_pol.threads_per_block, large_pol.items_per_thread>>;
  using AgentReduceT = reduce::AgentReduce<large_agent_policy_t, InputIteratorT, OffsetT, ReductionOpT, AccumT>;

  // Medium segment agent (one warp per segment)
  static constexpr SegmentedReduceWarpReducePolicy med_pol = full_policy.medium_reduce;
  using medium_agent_policy_t =
    agent_warp_reduce_policy<med_pol.threads_per_block,
                             med_pol.threads_per_warp,
                             med_pol.items_per_thread,
                             void,
                             med_pol.vec_size,
                             med_pol.load_modifier>;
  using AgentMediumReduceT =
    reduce::AgentWarpReduce<medium_agent_policy_t, InputIteratorT, OffsetT, ReductionOpT, AccumT>;

  // Small segment agent (one thread per segment)
  static constexpr SegmentedReduceWarpReducePolicy small_pol = full_policy.small_reduce;
  using small_agent_policy_t =
    agent_warp_reduce_policy<small_pol.threads_per_block,
                             small_pol.threads_per_warp,
                             small_pol.items_per_thread,
                             void,
                             small_pol.vec_size,
                             small_pol.load_modifier>;
  using AgentSmallReduceT =
    reduce::AgentWarpReduce<small_agent_policy_t, InputIteratorT, OffsetT, ReductionOpT, AccumT>;

  constexpr int small_items_per_tile  = small_pol.items_per_tile();
  constexpr int medium_items_per_tile = med_pol.items_per_tile();

  constexpr int segments_per_small_block  = small_pol.segments_per_block();
  constexpr int small_threads_per_warp    = small_pol.threads_per_warp;
  constexpr int segments_per_medium_block = med_pol.segments_per_block();
  constexpr int medium_threads_per_warp   = med_pol.threads_per_warp;

  // Shared memory storage
  __shared__ union
  {
    typename AgentReduceT::TempStorage large_storage;
    typename AgentMediumReduceT::TempStorage medium_storage[segments_per_medium_block];
    typename AgentSmallReduceT::TempStorage small_storage[segments_per_small_block];
  } temp_storage;

  const int bid = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);

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
            reduce::handle_empty_problem(d_out + global_segment_id, init);
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

  if (::cuda::in_range(max_segment_size, static_cast<size_t>(1), static_cast<size_t>(small_items_per_tile)))
  {
    small_medium_seg_reduction(
      ::cuda::std::type_identity<AgentSmallReduceT>{},
      temp_storage.small_storage,
      ::cuda::std::integral_constant<int, small_threads_per_warp>{},
      ::cuda::std::integral_constant<int, segments_per_small_block>{});
  }
  else if (::cuda::in_range(max_segment_size, static_cast<size_t>(1), static_cast<size_t>(medium_items_per_tile)))
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
        reduce::handle_empty_problem(d_out + bid, init);
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
 * @tparam PolicySelector
 *   Policy selector
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
 * @tparam InitValueT
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
template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitValueT,
          typename AccumT>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_reduce_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
_CCCL_KERNEL_ATTRIBUTES
__launch_bounds__(current_policy<PolicySelector>().large_reduce.threads_per_block) void DeviceFixedSizeSegmentedReduceKernel(
  const InputIteratorT d_in,
  const OutputIteratorT d_out,
  const OffsetT segment_size,
  const int num_segments,
  ReductionOpT reduction_op,
  const InitValueT init,
  AccumT* const d_partial_out,
  const int full_chunk_size,
  const int blocks_per_segment)
{
  static constexpr SegmentedReducePolicy full_policy = current_policy<PolicySelector>();

  // Large segment agent (one block per segment)
  static constexpr ReducePassPolicy large_pol = full_policy.large_reduce;

  // TODO(bgruber): pass policy directly as template argument to AgentReduce in C++20
  using large_agent_policy_t =
    agent_reduce_policy<0,
                        0,
                        void,
                        large_pol.vec_size,
                        large_pol.reduce_algorithm,
                        large_pol.load_modifier,
                        NoScaling<large_pol.threads_per_block, large_pol.items_per_thread>>;
  using AgentReduceT = reduce::AgentReduce<large_agent_policy_t, InputIteratorT, int, ReductionOpT, AccumT>;

  // Medium segment agent (one warp per segment)
  static constexpr SegmentedReduceWarpReducePolicy med_pol = full_policy.medium_reduce;
  using medium_agent_policy_t =
    agent_warp_reduce_policy<med_pol.threads_per_block,
                             med_pol.threads_per_warp,
                             med_pol.items_per_thread,
                             void,
                             med_pol.vec_size,
                             med_pol.load_modifier>;
  using AgentMediumReduceT = reduce::AgentWarpReduce<medium_agent_policy_t, InputIteratorT, int, ReductionOpT, AccumT>;

  // Small segment agent (one thread per segment)
  static constexpr SegmentedReduceWarpReducePolicy small_pol = full_policy.small_reduce;
  using small_agent_policy_t =
    agent_warp_reduce_policy<small_pol.threads_per_block,
                             small_pol.threads_per_warp,
                             small_pol.items_per_thread,
                             void,
                             small_pol.vec_size,
                             small_pol.load_modifier>;
  using AgentSmallReduceT = reduce::AgentWarpReduce<small_agent_policy_t, InputIteratorT, int, ReductionOpT, AccumT>;

  constexpr int small_items_per_tile  = small_pol.items_per_tile();
  constexpr int medium_items_per_tile = med_pol.items_per_tile();

  constexpr int segments_per_small_block  = small_pol.segments_per_block();
  constexpr int small_threads_per_warp    = small_pol.threads_per_warp;
  constexpr int segments_per_medium_block = med_pol.segments_per_block();
  constexpr int medium_threads_per_warp   = med_pol.threads_per_warp;

  // Shared memory storage
  __shared__ union
  {
    typename AgentReduceT::TempStorage large_storage;
    typename AgentMediumReduceT::TempStorage medium_storage[segments_per_medium_block];
    typename AgentSmallReduceT::TempStorage small_storage[segments_per_small_block];
  } temp_storage;

  const int bid = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);

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
          detail::reduce::handle_empty_problem(d_out + global_segment_id, init);
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

struct adaptive_segmented_reduce_policy
{
  int threads_per_block;
  int items_per_thread; // vector loads issued per lane per loop iteration
  int small_segment_size; // <= this many elements: summed serially by one thread
  int medium_segment_size; // >= this many elements: drained cooperatively by the warp
};

struct adaptive_segmented_reduce_default_policy_selector
{
  _CCCL_HOST_DEVICE_API constexpr adaptive_segmented_reduce_policy operator()(::cuda::compute_capability) const
  {
    return {256, 4, 4, 112};
  }
};

/// Seeds d_out[0..n) with val. Used as the PDL primary before the adaptive reduce.
/// Issues ItemsPerThread × 16-byte vector stores to maximise bytes in flight.
template <typename T, int ItemsPerThread>
_CCCL_KERNEL_ATTRIBUTES void AdaptiveSegmentedReduceFillKernel(T* d_out, int n, T val)
{
  constexpr int V = 16 / sizeof(T);
  using VecT      = typename ::cuda::vector_type<T, V>::type;

  const int tid    = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  const int stride = static_cast<int>(gridDim.x) * blockDim.x;

  // Prologue: scalar writes until d_out is 16B-aligned.
  const int peel = static_cast<int>((-reinterpret_cast<uintptr_t>(d_out) / sizeof(T)) & (V - 1));
  if (tid < ::cuda::std::min(peel, n))
  {
    d_out[tid] = val;
  }

  // Aligned vector body: ItemsPerThread 16-byte stores per thread.
  T* const d_aligned  = d_out + peel;
  const int n_aligned = n - peel;
  const int n_vec     = n_aligned / V;

  VecT vec_val;
#pragma unroll
  for (int v = 0; v < V; v++)
  {
    reinterpret_cast<T*>(&vec_val)[v] = val;
  }

#pragma unroll
  for (int u = 0; u < ItemsPerThread; u++)
  {
    const int i = tid + u * stride;
    if (i < n_vec)
    {
      reinterpret_cast<VecT*>(d_aligned)[i] = vec_val;
    }
  }

  // Epilogue: at most V-1 elements.
  // Assigned to the thread that performed the last vector write.
  if (tid == n_vec % stride)
  {
#pragma unroll
    for (int k = 0; k < V - 1; k++)
    {
      if (n_vec * V + k < n_aligned)
      {
        d_aligned[n_vec * V + k] = val;
      }
    }
  }
}

/// Find the segment owning the specified offset using a binary search over the segments.
template <class EndOffsetIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE int FindSegment(const int num_segments, EndOffsetIteratorT end_offsets, const int offset)
{
  int lo = 0, hi = num_segments;
  while (lo < hi)
  {
    const int mid = lo + (hi - lo) / 2;
    if ((int) end_offsets[mid] <= offset)
    {
      lo = mid + 1;
    }
    else
    {
      hi = mid;
    }
  }
  return lo;
}

/**
 * Segmented reduction (element partitioned, adaptive parallelism per segment).
 *
 * Each warp owns a contiguous chunk of the element index range and processes
 * its segments sequentially. Small segments are assigned to one thread and
 * large segments are handled cooperatively by the entire warp. Segments that
 * cross warps (or blocks) combine partial results via atomic operations.
 *
 * `d_end_offsets` is required to be monotonically increasing, since a binary
 * search is used to determine the starting element for each warp. This is
 * always true when `d_end_offsets == d_begin_offsets + 1`.
 *
 * @tparam PolicySelector
 *   Policy selector
 *
 * @tparam InputIterator
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIterator
 *   Output iterator type for recording the reduced aggregates @iterator
 *
 * @tparam BeginOffsetIteratorT
 *   Random-access input iterator type for reading segment beginning offsets @iterator
 *
 * @tparam EndOffsetIteratorT
 *   Random-access input iterator type for reading segment ending offsets @iterator
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member `T operator()(const T &a, const T &b)`
 *
 * @tparam InitValueT
 *   Initial value type
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out
 *   Pointer to the output aggregates (one per segment), pre-seeded with `init`
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of length `num_segments`
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length `num_segments`;
 *   must be monotonically increasing
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
template <class PolicySelector,
          class InputIterator,
          class OutputIterator,
          class BeginOffsetIteratorT,
          class EndOffsetIteratorT,
          class ReductionOpT,
          class InitValueT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(int(current_policy<PolicySelector>().threads_per_block),
                                          2048 / int(current_policy<PolicySelector>().threads_per_block)) //
  void DeviceAdaptiveSegmentedReduceKernel(
    const InputIterator d_in,
    const OutputIterator d_out,
    const BeginOffsetIteratorT d_begin_offsets,
    const EndOffsetIteratorT d_end_offsets,
    const int num_segments,
    ReductionOpT reduction_op,
    const InitValueT init)
{
  constexpr auto policy = current_policy<PolicySelector>();

  using T          = typename ::cuda::std::iterator_traits<InputIterator>::value_type;
  constexpr int V  = 16 / sizeof(T);
  using vecT       = typename ::cuda::vector_type<T, V>::type;
  const T identity = ::cuda::identity_element<ReductionOpT, T>();

  const int lane       = threadIdx.x & 31;
  const int warp_id    = (blockIdx.x * policy.threads_per_block + threadIdx.x) / 32;
  const int num_warps  = gridDim.x * (policy.threads_per_block / 32);
  const int local_warp = threadIdx.x >> 5;

  using WarpReduceT = cub::WarpReduce<T>;
  __shared__ typename WarpReduceT::TempStorage warp_temp[policy.threads_per_block / 32];

  // Assign one contiguous chunk of the input to each warp.
  const int num_items      = (int) d_end_offsets[num_segments - 1];
  const int items_per_warp = (num_items + num_warps - 1) / num_warps;
  const int warp_begin     = warp_id * items_per_warp;
  if (warp_begin >= num_items)
  {
    return;
  }
  const int warp_end = ::cuda::std::min(warp_begin + items_per_warp, num_items);

  // Search for the first and last segment assigned to this warp.
  // Lanes 0 and 1 run these two searches concurrently.
  int seg = 0;
  if (lane < 2)
  {
    seg = FindSegment(num_segments, d_end_offsets, (lane == 0) ? warp_begin : (warp_end - 1));
  }
  const int first_seg = __shfl_sync(0xffffffffu, seg, 0);
  const int last_seg  = __shfl_sync(0xffffffffu, seg, 1);

  // Iterate over all segments assigned to this warp in chunks of 32.
  // TODO: Use FindSegment to jump past large numbers of empty segments.
  for (int seg_base = first_seg; seg_base <= last_seg; seg_base += 32)
  {
    // Each thread takes ownership of one segment in the current chunk.
    const int segment = seg_base + lane;

    int seg_begin, seg_end;
    if (segment < num_segments)
    {
      seg_begin = (int) d_begin_offsets[segment];
      seg_end   = (int) d_end_offsets[segment];
    }
    else
    {
      seg_begin = 0;
      seg_end   = 0;
    }

    const int owned_begin = ::cuda::std::max(seg_begin, warp_begin);
    const int owned_end   = ::cuda::std::min(seg_end, warp_end);
    const int owned_len   = ::cuda::std::max(0, owned_end - owned_begin);

    // Initialize the accumulator for this thread.
    T acc = identity;

    // One thread fully processes each small or medium segment.
    if (owned_len > 0 && owned_len < policy.medium_segment_size)
    {
      // Prologue: peel up to (V - 1) elements to reach 16B alignment.
      const int prologue = ::cuda::std::min((-owned_begin) & (V - 1), owned_len);
#pragma unroll
      for (int k = 0; k < V - 1; k++)
      {
        if (k < prologue)
        {
          acc = reduction_op(acc, d_in[owned_begin + k]);
        }
      }
      const int body_begin = owned_begin + prologue;

// Process small segments completely, using a fully unrolled loop.
// Also peels iterations off the beginning of medium segments.
#pragma unroll
      for (int u = 0; u < policy.small_segment_size / V; u++)
      {
        const int j = body_begin + V * u;
        if (j + V <= owned_end)
        {
          const vecT chunk = *reinterpret_cast<const vecT*>(&d_in[j]);
          acc              = reduction_op(acc, cub::ThreadReduce(reinterpret_cast<const T(&)[V]>(chunk), reduction_op));
        }
      }

      // Process the remainder of a medium segment using a run-time-bounded loop.
      if (owned_len > policy.small_segment_size)
      {
        for (int j = body_begin + policy.small_segment_size; j + V <= owned_end; j += V * policy.items_per_thread)
        {
#pragma unroll
          for (int u = 0; u < policy.items_per_thread; u++)
          {
            const int jj = j + V * u;
            if (jj + V <= owned_end)
            {
              const vecT chunk = *reinterpret_cast<const vecT*>(&d_in[jj]);
              acc = reduction_op(acc, cub::ThreadReduce(reinterpret_cast<const T(&)[V]>(chunk), reduction_op));
            }
          }
        }
      }

      // Epilogue: the < V tail.
      const int tail_begin = body_begin + ((owned_end - body_begin) & ~(V - 1));
#pragma unroll
      for (int k = 0; k < V - 1; k++)
      {
        if (tail_begin + k < owned_end)
        {
          acc = reduction_op(acc, d_in[tail_begin + k]);
        }
      }
    }

    // Iterate over any large segments, with the whole warp cooperating on each segment.
    unsigned long_mask = __ballot_sync(0xffffffffu, owned_len >= policy.medium_segment_size);
    while (long_mask)
    {
      // Broadcast this segment's start and end to all threads.
      const int owner = __ffs(long_mask) - 1;
      long_mask &= long_mask - 1;
      const int coop_begin = __shfl_sync(0xffffffffu, owned_begin, owner);
      const int coop_end   = __shfl_sync(0xffffffffu, owned_end, owner);
      T lane_partial       = identity;

      // Peel for alignment, now cooperatively.
      const int prologue = ::cuda::std::min((-coop_begin) & (V - 1), coop_end - coop_begin);
      if (lane < prologue)
      {
        lane_partial = reduction_op(lane_partial, d_in[coop_begin + lane]);
      }

      // Use aligned vector loads until the end of the segment.
      const int body_begin = coop_begin + prologue;
      for (int j = body_begin + V * lane; j + V <= coop_end; j += V * 32 * policy.items_per_thread)
      {
#pragma unroll
        for (int u = 0; u < policy.items_per_thread; u++)
        {
          const int jj = j + V * 32 * u;
          if (jj + V <= coop_end)
          {
            const vecT chunk = cub::ThreadLoad<cub::LOAD_LDG>(reinterpret_cast<const vecT*>(&d_in[jj]));
            lane_partial =
              reduction_op(lane_partial, cub::ThreadReduce(reinterpret_cast<const T(&)[V]>(chunk), reduction_op));
          }
        }
      }

      // Handle the remainder, now cooperatively.
      const int tail_begin = body_begin + ((coop_end - body_begin) & ~(V - 1));
      if (lane < coop_end - tail_begin)
      {
        lane_partial = reduction_op(lane_partial, d_in[tail_begin + lane]);
      }

      // Reduce across the warp; owner keeps the result.
      const T total_lane0 = WarpReduceT(warp_temp[local_warp]).Reduce(lane_partial, reduction_op);
      const T coop_total  = __shfl_sync(0xffffffffu, total_lane0, 0);
      if (lane == owner)
      {
        acc = reduction_op(acc, coop_total);
      }
    }

    // Wait for the previous kernel to finish writing identity values.
    // This is only required once, but safe to call redundantly.
    _CCCL_PDL_GRID_DEPENDENCY_SYNC();

    // Skip invalid segments (i.e., from uneven division).
    if (segment < num_segments && owned_begin < owned_end)
    {
      // If multiple warps worked on this segment, combine atomically.
      if (owned_begin > seg_begin || owned_end < seg_end)
      {
        ::cuda::atomic_ref<T, ::cuda::thread_scope_device> ref(d_out[segment]);
        if constexpr (::cuda::__is_cuda_std_plus_v<ReductionOpT>)
        {
          ref.fetch_add(acc, ::cuda::memory_order_relaxed);
        }
        else if constexpr (::cuda::__is_cuda_minimum_v<ReductionOpT>)
        {
          ref.fetch_min(acc, ::cuda::memory_order_relaxed);
        }
        else if constexpr (::cuda::__is_cuda_maximum_v<ReductionOpT>)
        {
          ref.fetch_max(acc, ::cuda::memory_order_relaxed);
        }
      }
      else
      {
        // Otherwise, this thread can write directly to memory.
        d_out[segment] = reduction_op(init, acc);
      }
    }
  }
}
} // namespace detail::segmented_reduce

CUB_NAMESPACE_END
