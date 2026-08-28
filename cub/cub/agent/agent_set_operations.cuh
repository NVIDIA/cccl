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

#include <cub/agent/single_pass_scan_operators.cuh> // ScanTileState, TilePrefixCallbackOp
#include <cub/block/block_merge_sort.cuh> // cub::MergePath
#include <cub/block/block_scan.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_type.cuh>

#include <cuda/__memory/uninitialized_array.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstddef>

CUB_NAMESPACE_BEGIN
namespace detail::set_ops
{
// One step of a (biased) binary search. Advances @p begin past @p key (UpperBound) or up to the first element not
// ordered before @p key (lower bound). @p shift controls how the midpoint is biased towards @p begin: shift==1 yields
// an unbiased binary search, larger shifts probe closer to @p begin (useful when the searched-for run is expected to be
// short).
template <bool UpperBound, typename IntT, typename Offset, typename It, typename T, typename CompareOp>
_CCCL_DEVICE _CCCL_FORCEINLINE void
binary_search_iteration(It data, Offset& begin, Offset& end, T key, int shift, CompareOp compare_op)
{
  const IntT scale     = (IntT{1} << shift) - 1;
  const Offset mid     = (begin + scale * end) >> shift;
  const T key2         = data[mid];
  const bool pred      = UpperBound ? !compare_op(key, key2) : compare_op(key2, key);
  (pred ? begin : end) = pred ? mid + 1 : mid;
}

// Unbiased binary search returning the number of elements in [0, count) ordered before @p key (lower bound) or not
// after @p key (upper bound).
template <bool UpperBound, typename Offset, typename T, typename It, typename CompareOp>
_CCCL_DEVICE _CCCL_FORCEINLINE Offset binary_search(It data, Offset count, T key, CompareOp compare_op)
{
  Offset begin = 0;
  Offset end   = count;
  while (begin < end)
  {
    binary_search_iteration<UpperBound, int>(data, begin, end, key, 1, compare_op);
  }
  return begin;
}

// Binary search that first probes close to @p begin for up to @p levels iterations before falling back to an unbiased
// search. This accelerates the common case where the searched run starts near the front of the range.
template <bool UpperBound, typename IntT, typename Offset, typename T, typename It, typename CompareOp>
_CCCL_DEVICE _CCCL_FORCEINLINE Offset
biased_binary_search(It data, Offset count, T key, IntT levels, CompareOp compare_op)
{
  Offset begin = 0;
  Offset end   = count;

  if (levels >= 4 && begin < end)
  {
    binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 9, compare_op);
  }
  if (levels >= 3 && begin < end)
  {
    binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 7, compare_op);
  }
  if (levels >= 2 && begin < end)
  {
    binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 5, compare_op);
  }
  if (levels >= 1 && begin < end)
  {
    binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 4, compare_op);
  }

  while (begin < end)
  {
    binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 1, compare_op);
  }
  return begin;
}

//! Duplicate-aware variant of the merge path. In addition to intersecting the diagonal @p diag with the merge path
//! (like @ref cub::MergePath), it evenly distributes runs of equal keys between the two input sequences so that set
//! operations observe consistent multiplicities. Returns the pair (index into @p keys1, index into @p keys2); the
//! second component may be incremented by one (the "star") to break ties on the boundary of an equal-key run.
template <typename It1, typename It2, typename Offset, typename IntT, typename CompareOp>
_CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::pair<Offset, Offset>
balanced_path(It1 keys1, It2 keys2, Offset num_keys1, Offset num_keys2, Offset diag, IntT levels, CompareOp compare_op)
{
  using key_t = it_value_t<It1>;

  // cub::MergePath computes the lower-bound merge path (it advances keys1 whenever !compare(keys2, keys1)).
  Offset index1 = cub::MergePath(keys1, keys2, num_keys1, num_keys2, diag, compare_op);
  Offset index2 = diag - index1;

  bool star = false;
  if (index2 < num_keys2)
  {
    const key_t x = keys2[index2];

    // Search for the beginning of the duplicate run of x in both A and B.
    const Offset start1 = biased_binary_search<false>(keys1, index1, x, levels, compare_op);
    const Offset start2 = biased_binary_search<false>(keys2, index2, x, levels, compare_op);

    // The distance between x's merge path and its lower bound is its rank. We add up the A and B ranks and evenly
    // distribute them to obtain a stairstep path.
    const Offset run1      = index1 - start1;
    const Offset run2_lb   = index2 - start2;
    const Offset total_run = run1 + run2_lb;

    // Attempt to advance B and regress A.
    Offset advance2       = (::cuda::std::max) (total_run >> 1, total_run - run1);
    const Offset end2     = (::cuda::std::min) (num_keys2, start2 + advance2 + 1);
    const Offset run2     = (index2 + binary_search<true>(keys2 + index2, end2 - index2, x, compare_op)) - start2;
    advance2              = (::cuda::std::min) (advance2, run2);
    const Offset advance1 = total_run - advance2;

    const bool round_up = (advance1 == advance2 + 1) && (advance2 < run2);
    if (round_up)
    {
      star = true;
    }

    index1 = start1 + advance1;
  }
  return ::cuda::std::make_pair(index1, (diag - index1) + Offset{star});
}

//---------------------------------------------------------------------
// Serial set operations
//
// Each functor consumes the two per-thread sub-ranges of a shared-memory buffer (interleaved as [keys1 | keys2]) and
// emits up to ITEMS_PER_THREAD results into @p output, recording the source shared-memory index of each result in
// @p indices (needed to gather the matching value in the by-key case). The return value is a per-item bitmask marking
// which of the ITEMS_PER_THREAD slots are live. The shared buffer is over-allocated so the trailing ++begin reads stay
// in bounds without explicit range checks.
//---------------------------------------------------------------------

//! Emit A when A and B are both in range and equal.
struct serial_set_intersection
{
  // max_input_size <= 32
  template <typename T, typename CompareOp, int ItemsPerThread>
  _CCCL_DEVICE _CCCL_FORCEINLINE int operator()(
    T* keys,
    int keys1_beg,
    int keys2_beg,
    int keys1_count,
    int keys2_count,
    T (&output)[ItemsPerThread],
    int (&indices)[ItemsPerThread],
    CompareOp compare_op) const
  {
    int active_mask = 0;

    int a_begin     = keys1_beg;
    int b_begin     = keys2_beg;
    const int a_end = keys1_beg + keys1_count;
    const int b_end = keys2_beg + keys2_count;

    T a_key = keys[a_begin];
    T b_key = keys[b_begin];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      const bool p_a = compare_op(a_key, b_key);
      const bool p_b = compare_op(b_key, a_key);

      // The outputs must come from A by definition of set intersection.
      output[i]  = a_key;
      indices[i] = a_begin;

      if ((a_begin < a_end) && (b_begin < b_end) && p_a == p_b)
      {
        active_mask |= 1 << i;
      }

      if (!p_b)
      {
        a_key = keys[++a_begin];
      }
      if (!p_a)
      {
        b_key = keys[++b_begin];
      }
    }
    return active_mask;
  }
};

//! Emit A when A < B and B when B < A.
struct serial_set_symmetric_difference
{
  // max_input_size <= 32
  template <typename T, typename CompareOp, int ItemsPerThread>
  _CCCL_DEVICE _CCCL_FORCEINLINE int operator()(
    T* keys,
    int keys1_beg,
    int keys2_beg,
    int keys1_count,
    int keys2_count,
    T (&output)[ItemsPerThread],
    int (&indices)[ItemsPerThread],
    CompareOp compare_op) const
  {
    int active_mask = 0;

    int a_begin     = keys1_beg;
    int b_begin     = keys2_beg;
    const int a_end = keys1_beg + keys1_count;
    const int b_end = keys2_beg + keys2_count;
    const int end   = a_end + b_end;

    T a_key = keys[a_begin];
    T b_key = keys[b_begin];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      bool p_b = a_begin >= a_end;
      bool p_a = !p_b && b_begin >= b_end;

      if (!p_a && !p_b)
      {
        p_a = compare_op(a_key, b_key);
        p_b = !p_a && compare_op(b_key, a_key);
      }

      output[i]  = p_a ? a_key : b_key;
      indices[i] = p_a ? a_begin : b_begin;

      if (a_begin + b_begin < end && p_a != p_b)
      {
        active_mask |= 1 << i;
      }

      if (!p_b)
      {
        a_key = keys[++a_begin];
      }
      if (!p_a)
      {
        b_key = keys[++b_begin];
      }
    }
    return active_mask;
  }
};

//! Emit A when A < B.
struct serial_set_difference
{
  // max_input_size <= 32
  template <typename T, typename CompareOp, int ItemsPerThread>
  _CCCL_DEVICE _CCCL_FORCEINLINE int operator()(
    T* keys,
    int keys1_beg,
    int keys2_beg,
    int keys1_count,
    int keys2_count,
    T (&output)[ItemsPerThread],
    int (&indices)[ItemsPerThread],
    CompareOp compare_op) const
  {
    int active_mask = 0;

    int a_begin     = keys1_beg;
    int b_begin     = keys2_beg;
    const int a_end = keys1_beg + keys1_count;
    const int b_end = keys2_beg + keys2_count;
    const int end   = a_end + b_end;

    T a_key = keys[a_begin];
    T b_key = keys[b_begin];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      bool p_b = a_begin >= a_end;
      bool p_a = !p_b && b_begin >= b_end;

      if (!p_a && !p_b)
      {
        p_a = compare_op(a_key, b_key);
        p_b = !p_a && compare_op(b_key, a_key);
      }

      // The outputs must come from A by definition of set difference.
      output[i]  = a_key;
      indices[i] = a_begin;

      if (a_begin + b_begin < end && p_a)
      {
        active_mask |= 1 << i;
      }

      if (!p_b)
      {
        a_key = keys[++a_begin];
      }
      if (!p_a)
      {
        b_key = keys[++b_begin];
      }
    }
    return active_mask;
  }
};

//! Emit A when A <= B, otherwise emit B.
struct serial_set_union
{
  // max_input_size <= 32
  template <typename T, typename CompareOp, int ItemsPerThread>
  _CCCL_DEVICE _CCCL_FORCEINLINE int operator()(
    T* keys,
    int keys1_beg,
    int keys2_beg,
    int keys1_count,
    int keys2_count,
    T (&output)[ItemsPerThread],
    int (&indices)[ItemsPerThread],
    CompareOp compare_op) const
  {
    int active_mask = 0;

    int a_begin     = keys1_beg;
    int b_begin     = keys2_beg;
    const int a_end = keys1_beg + keys1_count;
    const int b_end = keys2_beg + keys2_count;
    const int end   = a_end + b_end;

    T a_key = keys[a_begin];
    T b_key = keys[b_begin];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      bool p_b = a_begin >= a_end;
      bool p_a = !p_b && b_begin >= b_end;

      if (!p_a && !p_b)
      {
        p_a = compare_op(a_key, b_key);
        p_b = !p_a && compare_op(b_key, a_key);
      }

      // Output A in case of a tie, so check if b < a.
      output[i]  = p_b ? b_key : a_key;
      indices[i] = p_b ? b_begin : a_begin;

      if (a_begin + b_begin < end)
      {
        active_mask |= 1 << i;
      }

      if (!p_b)
      {
        a_key = keys[++a_begin];
      }
      if (!p_a)
      {
        b_key = keys[++b_begin];
      }
    }
    return active_mask;
  }
};

//! The tuning policy governing a single instantiation of @ref agent_set_op.
template <int BlockThreads,
          int ItemsPerThread,
          CacheLoadModifier LoadModifier   = LOAD_LDG,
          BlockScanAlgorithm ScanAlgorithm = BLOCK_SCAN_WARP_SCANS>
struct agent_set_op_policy
{
  static constexpr int BLOCK_THREADS                 = BlockThreads;
  static constexpr int ITEMS_PER_THREAD              = ItemsPerThread;
  static constexpr CacheLoadModifier LOAD_MODIFIER   = LoadModifier;
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM = ScanAlgorithm;
};

//! One block consumes one tile. @p partitions holds the merge-path partition boundaries (one per tile plus a trailing
//! sentinel) computed by the balanced-partition kernel; @p tile_state carries the decoupled look-back scan state used
//! to place each tile's compacted output. The total number of emitted elements is written to @p output_count by the
//! last tile.
template <typename SetOpPolicyT,
          typename KeysIt1,
          typename KeysIt2,
          typename ValuesIt1,
          typename ValuesIt2,
          typename KeysOutputIt,
          typename ValuesOutputIt,
          typename Offset,
          typename CompareOp,
          typename SetOp,
          typename NumSelectedIteratorT,
          bool HasValues>
struct agent_set_op
{
  using key_type   = it_value_t<KeysIt1>;
  using value_type = it_value_t<ValuesIt1>;

  using ScanTileStateT = ScanTileState<Offset>;

  static constexpr int BLOCK_THREADS    = SetOpPolicyT::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = SetOpPolicyT::ITEMS_PER_THREAD;
  // One item is left in reserve so the serial set operations can read one past their range without a bounds check.
  static constexpr int ITEMS_PER_TILE = BLOCK_THREADS * ITEMS_PER_THREAD - 1;

  static constexpr CacheLoadModifier LOAD_MODIFIER = SetOpPolicyT::LOAD_MODIFIER;

  using TilePrefixCallbackT = TilePrefixCallbackOp<Offset, ::cuda::std::plus<>, ScanTileStateT>;
  using BlockScanT          = BlockScan<Offset, BLOCK_THREADS, SetOpPolicyT::SCAN_ALGORITHM>;

  union TempStorage
  {
    struct ScanStorage
    {
      typename BlockScanT::TempStorage scan;
      typename TilePrefixCallbackT::TempStorage prefix;
    } scan_storage;

    struct LoadStorage
    {
      ::cuda::__uninitialized_array<int, BLOCK_THREADS> offset;
      union
      {
        // Over-allocated by BLOCK_THREADS items so serial set operations can read one past their range without range
        // checks (see ITEMS_PER_TILE).
        ::cuda::__uninitialized_array<key_type, ITEMS_PER_TILE + BLOCK_THREADS> keys_shared;
        ::cuda::__uninitialized_array<value_type, ITEMS_PER_TILE + BLOCK_THREADS> values_shared;
      };
    } load_storage;
  };

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  TempStorage& storage;
  ScanTileStateT& tile_state;
  KeysIt1 keys1_in;
  KeysIt2 keys2_in;
  ValuesIt1 values1_in;
  ValuesIt2 values2_in;
  KeysOutputIt keys_out;
  ValuesOutputIt values_out;
  CompareOp compare_op;
  SetOp set_op;
  const ::cuda::std::pair<Offset, Offset>* partitions;
  NumSelectedIteratorT output_count;
  Offset num_tiles;

  //---------------------------------------------------------------------
  // Utility functions
  //---------------------------------------------------------------------

  template <bool IsFullTile, typename T, typename It1, typename It2>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  gmem_to_reg(T (&output)[ITEMS_PER_THREAD], It1 input1, It2 input2, int count1, int count2)
  {
    if constexpr (IsFullTile)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < ITEMS_PER_THREAD - 1; ++item)
      {
        const int idx = BLOCK_THREADS * item + threadIdx.x;
        output[item]  = (idx < count1) ? static_cast<T>(input1[idx]) : static_cast<T>(input2[idx - count1]);
      }

      // The last item might be a conditional load even for full tiles.
      const int item = ITEMS_PER_THREAD - 1;
      const int idx  = BLOCK_THREADS * item + threadIdx.x;
      if (idx < count1 + count2)
      {
        output[item] = (idx < count1) ? static_cast<T>(input1[idx]) : static_cast<T>(input2[idx - count1]);
      }
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < ITEMS_PER_THREAD; ++item)
      {
        const int idx = BLOCK_THREADS * item + threadIdx.x;
        if (idx < count1 + count2)
        {
          output[item] = (idx < count1) ? static_cast<T>(input1[idx]) : static_cast<T>(input2[idx - count1]);
        }
      }
    }
  }

  template <typename T, typename It>
  _CCCL_DEVICE _CCCL_FORCEINLINE void reg_to_shared(It output, T (&input)[ITEMS_PER_THREAD])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ITEMS_PER_THREAD; ++item)
    {
      const int idx = BLOCK_THREADS * item + threadIdx.x;
      output[idx]   = input[item];
    }
  }

  template <typename OutputIt, typename T, typename SharedIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void scatter(
    OutputIt output,
    T (&input)[ITEMS_PER_THREAD],
    SharedIt shared,
    int active_mask,
    Offset thread_output_prefix,
    Offset tile_output_prefix,
    int tile_output_count)
  {
    int local_scatter_idx = static_cast<int>(thread_output_prefix - tile_output_prefix);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ITEMS_PER_THREAD; ++item)
    {
      if (active_mask & (1 << item))
      {
        shared[local_scatter_idx++] = input[item];
      }
    }
    __syncthreads();

    for (int item = static_cast<int>(threadIdx.x); item < tile_output_count; item += BLOCK_THREADS)
    {
      output[tile_output_prefix + item] = shared[item];
    }
  }

  //---------------------------------------------------------------------
  // Tile processing
  //---------------------------------------------------------------------

  template <bool IsLastTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_tile(Offset tile_idx)
  {
    const ::cuda::std::pair<Offset, Offset> partition_beg = partitions[tile_idx + 0];
    const ::cuda::std::pair<Offset, Offset> partition_end = partitions[tile_idx + 1];

    const int num_keys1 = static_cast<int>(partition_end.first - partition_beg.first);
    const int num_keys2 = static_cast<int>(partition_end.second - partition_beg.second);

    // Load both key ranges into shared memory, laid out as [keys1 | keys2].
    const auto keys1_load = detail::try_make_cache_modified_iterator<LOAD_MODIFIER>(keys1_in);
    const auto keys2_load = detail::try_make_cache_modified_iterator<LOAD_MODIFIER>(keys2_in);
    key_type keys_loc[ITEMS_PER_THREAD];
    gmem_to_reg<!IsLastTile>(
      keys_loc, keys1_load + partition_beg.first, keys2_load + partition_beg.second, num_keys1, num_keys2);
    reg_to_shared(&storage.load_storage.keys_shared[0], keys_loc);
    __syncthreads();

    const int diag_loc = (::cuda::std::min) (ITEMS_PER_THREAD * static_cast<int>(threadIdx.x), num_keys1 + num_keys2);

    const ::cuda::std::pair<int, int> partition_loc = balanced_path(
      &storage.load_storage.keys_shared[0],
      &storage.load_storage.keys_shared[num_keys1],
      num_keys1,
      num_keys2,
      diag_loc,
      4,
      compare_op);

    const int keys1_beg_loc = partition_loc.first;
    const int keys2_beg_loc = partition_loc.second;

    // Compute the difference between this thread's partition and the next thread's to obtain the per-thread counts.
    // The two 16-bit coordinates are packed into a single int and shifted one slot to the left across the block.
    const int value =
      threadIdx.x == 0 ? (num_keys1 << 16) | num_keys2 : (partition_loc.first << 16) | partition_loc.second;
    const int dst                    = threadIdx.x == 0 ? BLOCK_THREADS - 1 : static_cast<int>(threadIdx.x) - 1;
    storage.load_storage.offset[dst] = value;
    __syncthreads();

    const int keys1_end_loc = storage.load_storage.offset[threadIdx.x] >> 16;
    const int keys2_end_loc = storage.load_storage.offset[threadIdx.x] & 0xFFFF;

    const int num_keys1_loc = keys1_end_loc - keys1_beg_loc;
    const int num_keys2_loc = keys2_end_loc - keys2_beg_loc;

    // Perform the serial set operation.
    int indices[ITEMS_PER_THREAD];
    const int active_mask = set_op(
      &storage.load_storage.keys_shared[0],
      keys1_beg_loc,
      keys2_beg_loc + num_keys1,
      num_keys1_loc,
      num_keys2_loc,
      keys_loc,
      indices,
      compare_op);
    __syncthreads();

    // Look-back scan over the per-thread output counts to compute the global thread output base and the tile output
    // count.
    Offset tile_output_count         = 0;
    Offset thread_output_prefix      = 0;
    Offset tile_output_prefix        = 0;
    const Offset thread_output_count = static_cast<Offset>(::cuda::std::popcount(static_cast<unsigned>(active_mask)));

    if (tile_idx == 0)
    {
      BlockScanT(storage.scan_storage.scan).ExclusiveSum(thread_output_count, thread_output_prefix, tile_output_count);
      if (threadIdx.x == 0 && !IsLastTile)
      {
        tile_state.SetInclusive(0, tile_output_count);
      }
    }
    else
    {
      TilePrefixCallbackT prefix_cb(tile_state, storage.scan_storage.prefix, ::cuda::std::plus<>{}, tile_idx);
      BlockScanT(storage.scan_storage.scan).ExclusiveSum(thread_output_count, thread_output_prefix, prefix_cb);
      tile_output_count  = prefix_cb.GetBlockAggregate();
      tile_output_prefix = prefix_cb.GetExclusivePrefix();
    }
    __syncthreads();

    // Scatter the surviving keys.
    scatter(keys_out,
            keys_loc,
            &storage.load_storage.keys_shared[0],
            active_mask,
            thread_output_prefix,
            tile_output_prefix,
            static_cast<int>(tile_output_count));

    if constexpr (HasValues)
    {
      const auto values1_load = detail::try_make_cache_modified_iterator<LOAD_MODIFIER>(values1_in);
      const auto values2_load = detail::try_make_cache_modified_iterator<LOAD_MODIFIER>(values2_in);
      value_type values_loc[ITEMS_PER_THREAD];
      gmem_to_reg<!IsLastTile>(
        values_loc, values1_load + partition_beg.first, values2_load + partition_beg.second, num_keys1, num_keys2);
      __syncthreads();

      reg_to_shared(&storage.load_storage.values_shared[0], values_loc);
      __syncthreads();

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < ITEMS_PER_THREAD; ++item)
      {
        if (active_mask & (1 << item))
        {
          values_loc[item] = storage.load_storage.values_shared[indices[item]];
        }
      }
      __syncthreads();

      scatter(values_out,
              values_loc,
              &storage.load_storage.values_shared[0],
              active_mask,
              thread_output_prefix,
              tile_output_prefix,
              static_cast<int>(tile_output_count));
    }

    if (IsLastTile && threadIdx.x == 0)
    {
      *output_count = tile_output_prefix + tile_output_count;
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()()
  {
    const Offset tile_idx = static_cast<Offset>(blockIdx.x);
    if (tile_idx < num_tiles - 1)
    {
      consume_tile<false>(tile_idx);
    }
    else
    {
      consume_tile<true>(tile_idx);
    }
  }
};
} // namespace detail::set_ops
CUB_NAMESPACE_END
