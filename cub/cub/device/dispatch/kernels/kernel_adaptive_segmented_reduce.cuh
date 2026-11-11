// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/device/dispatch/tuning/tuning_segmented_reduce.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/util_arch.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/__functional/operator_properties.h>
#include <cuda/__type_traits/vector_type.h>
#include <cuda/atomic>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cccl/cuda_capabilities.h>
#include <cuda/std/__iterator/iterator_traits.h>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_reduce
{
/// Seeds d_out[0..n) with val. Used as the PDL primary before the adaptive reduce.
/// Issues ItemsPerThread × 16-byte vector stores to maximise bytes in flight.
template <typename T, int ItemsPerThread>
_CCCL_KERNEL_ATTRIBUTES void AdaptiveSegmentedReduceInitKernel(T* d_out, int n, T val)
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

  // Elements from p up to the next 16-byte boundary. Uses only the low address bits, which is exact because
  // 16 divides 2^32 and T is naturally aligned.
  auto elements_to_16B_boundary = [](const T* p) {
    const auto elem_index = static_cast<unsigned>(reinterpret_cast<uintptr_t>(p) / sizeof(T));
    return static_cast<int>((0u - elem_index) & (V - 1));
  };

  const int lane       = threadIdx.x & 31;
  const int warp_id    = (blockIdx.x * policy.threads_per_block + threadIdx.x) / 32;
  const int num_warps  = gridDim.x * (policy.threads_per_block / 32);
  const int local_warp = threadIdx.x >> 5;

  using WarpReduceT = cub::WarpReduce<T>;
  __shared__ typename WarpReduceT::TempStorage warp_temp[policy.threads_per_block / 32];

  // Assign one contiguous chunk of the input to each warp.
  // The products and sums below can exceed INT_MAX when num_items is close to it, so use 64-bit.
  const int num_items           = (int) d_end_offsets[num_segments - 1];
  const int items_per_warp      = num_items / num_warps + (num_items % num_warps != 0);
  const long long warp_begin_ll = static_cast<long long>(warp_id) * items_per_warp;
  if (warp_begin_ll >= num_items)
  {
    return;
  }
  const int warp_begin = static_cast<int>(warp_begin_ll);
  const int warp_end   = static_cast<int>(::cuda::std::min<long long>(warp_begin_ll + items_per_warp, num_items));

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
      // Prologue: peel up to (V - 1) elements until d_in + element is 16B-aligned.
      const int prologue = ::cuda::std::min(elements_to_16B_boundary(d_in + owned_begin), owned_len);
#pragma unroll
      for (int k = 0; k < V - 1; k++)
      {
        if (k < prologue)
        {
          acc = reduction_op(acc, d_in[owned_begin + k]);
        }
      }
      // Unsigned indices: j + V stays below 2^32 even when owned_end is close to INT_MAX, so there is no overflow.
      const unsigned body_begin = owned_begin + prologue;
      const unsigned body_end   = owned_end;
      const unsigned body_len   = body_end - body_begin;

// Process small segments completely, using a fully unrolled loop.
// Also peels iterations off the beginning of medium segments.
#pragma unroll
      for (int u = 0; u < policy.small_segment_size / V; u++)
      {
        const unsigned j = body_begin + V * u;
        if (j + V <= body_end)
        {
          const vecT chunk = *reinterpret_cast<const vecT*>(&d_in[j]);
          acc              = reduction_op(acc, cub::ThreadReduce(reinterpret_cast<const T(&)[V]>(chunk), reduction_op));
        }
      }

      // Process the remainder of a medium segment using a run-time-bounded loop.
      if (owned_len > policy.small_segment_size)
      {
        for (unsigned j = body_begin + policy.small_segment_size; j + V <= body_end; j += V * policy.items_per_thread)
        {
#pragma unroll
          for (int u = 0; u < policy.items_per_thread; u++)
          {
            const unsigned jj = j + V * u;
            if (jj + V <= body_end)
            {
              const vecT chunk = *reinterpret_cast<const vecT*>(&d_in[jj]);
              acc = reduction_op(acc, cub::ThreadReduce(reinterpret_cast<const T(&)[V]>(chunk), reduction_op));
            }
          }
        }
      }

      // Epilogue: the < V tail.
      const int tail_len        = static_cast<int>(body_len & (V - 1));
      const unsigned tail_begin = body_end - tail_len;
#pragma unroll
      for (int k = 0; k < V - 1; k++)
      {
        if (k < tail_len)
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
      const int prologue = ::cuda::std::min(elements_to_16B_boundary(d_in + coop_begin), coop_end - coop_begin);
      if (lane < prologue)
      {
        lane_partial = reduction_op(lane_partial, d_in[coop_begin + lane]);
      }

      // Use aligned vector loads until the end of the segment.
      // `rem` counts the elements left from this lane's position to coop_end, which cannot overflow.
      const int body_begin = coop_begin + prologue;
      for (int rem = coop_end - body_begin - V * lane; rem >= V; rem -= V * 32 * policy.items_per_thread)
      {
#pragma unroll
        for (int u = 0; u < policy.items_per_thread; u++)
        {
          const int rem_u = rem - V * 32 * u;
          if (rem_u >= V)
          {
            const vecT chunk = cub::ThreadLoad<cub::LOAD_LDG>(reinterpret_cast<const vecT*>(&d_in[coop_end - rem_u]));
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
