// SPDX-FileCopyrightText: Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/thread/thread_sort.cuh>
#include <cub/util_math.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/is_same.h>

CUB_NAMESPACE_BEGIN

//! Computes the intersection of the diagonal \c diag with the merge path in the merge matrix of two input sequences.
//! This implements the DiagonalIntersection algorithm from Merge-Path. Additional details can be found in:
//! * S. Odeh, O. Green, Z. Mwassi, O. Shmueli, Y. Birk, "Merge Path - Parallel Merging Made Simple", Multithreaded
//!   Architectures and Applications (MTAAP) Workshop, IEEE 26th International Parallel & Distributed Processing
//!   Symposium (IPDPS), 2012
//! * S. Odeh, O. Green, Y. Birk, "Merge Path - A Visually Intuitive Approach to Parallel Merging", 2014, URL:
//!   https://arxiv.org/abs/1406.2628
//! \returns The number of elements merged from the first sequence at the intersection of the diagonal with the merge
//! path. The number of elements merged from the second sequence is \c diag minus the returned value.
template <typename KeyIt1, typename KeyIt2, typename OffsetT, typename BinaryPred>
_CCCL_DEVICE _CCCL_FORCEINLINE OffsetT
MergePath(KeyIt1 keys1, KeyIt2 keys2, OffsetT keys1_count, OffsetT keys2_count, OffsetT diag, BinaryPred binary_pred)
{
  OffsetT keys1_begin = diag < keys2_count ? 0 : diag - keys2_count;
  OffsetT keys1_end   = (::cuda::std::min) (diag, keys1_count);

  while (keys1_begin < keys1_end)
  {
    const OffsetT mid = cub::MidPoint<OffsetT>(keys1_begin, keys1_end);
    // pull copies of the keys before calling binary_pred so proxy references are unwrapped
    const detail::it_value_t<KeyIt1> key1 = keys1[mid];
    const detail::it_value_t<KeyIt2> key2 = keys2[diag - 1 - mid];
    if (binary_pred(key2, key1))
    {
      keys1_end = mid;
    }
    else
    {
      keys1_begin = mid + 1;
    }
  }
  return keys1_begin;
}

//! Merges elements from two sorted sequences
//! \tparam ItemsPerThread The number of elements to merge and write to \c output
//! \param keys_shared An iterator to shared memory containing from which both sequences are reachable
//! \param keys1_beg The index into \c keys_shared where the first sequence starts
//! \param keys2_beg The index into \c keys_shared where the second sequence starts
//! \param keys1_count The maximum number of keys to merge from the first sequence. One more item may be read but is not
//! used.
//! \param keys2_count The maximum number of keys to merge from the first sequence. One more item may be read but is not
//! used.
//! \param output The output array
//! \param indices The shared memory indices relative to \c keys_shared of the elements written to \c output
template <typename KeyIt, typename KeyT, typename CompareOp, int ItemsPerThread>
_CCCL_DEVICE _CCCL_FORCEINLINE void SerialMerge(
  KeyIt keys_shared,
  int keys1_beg,
  int keys2_beg,
  int keys1_count,
  int keys2_count,
  KeyT (&output)[ItemsPerThread],
  int (&indices)[ItemsPerThread],
  CompareOp compare_op,
  KeyT oob_default)
{
  const int keys1_end = keys1_beg + keys1_count;
  const int keys2_end = keys2_beg + keys2_count;

  KeyT key1 = keys1_count != 0 ? keys_shared[keys1_beg] : oob_default;
  KeyT key2 = keys2_count != 0 ? keys_shared[keys2_beg] : oob_default;

  _CCCL_SORT_MAYBE_UNROLL()
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    const bool p  = (keys2_beg < keys2_end) && ((keys1_beg >= keys1_end) || compare_op(key2, key1));
    output[item]  = p ? key2 : key1;
    indices[item] = p ? keys2_beg++ : keys1_beg++;
    if (p)
    {
      key2 = keys_shared[keys2_beg];
    }
    else
    {
      key1 = keys_shared[keys1_beg];
    }
  }
}

template <typename KeyIt, typename KeyT, typename CompareOp, int ItemsPerThread>
_CCCL_DEVICE _CCCL_FORCEINLINE void SerialMerge(
  KeyIt keys_shared,
  int keys1_beg,
  int keys2_beg,
  int keys1_count,
  int keys2_count,
  KeyT (&output)[ItemsPerThread],
  int (&indices)[ItemsPerThread],
  CompareOp compare_op)
{
  SerialMerge(keys_shared, keys1_beg, keys2_beg, keys1_count, keys2_count, output, indices, compare_op, output[0]);
}

/**
 * @brief Generalized merge sort algorithm
 *
 * This class is used to reduce code duplication. Warp and Block merge sort
 * differ only in how they compute thread index and how they synchronize
 * threads. Since synchronization might require access to custom data
 * (like member mask), CRTP is used.
 *
 * @par
 * The code snippet below illustrates the way this class can be used.
 * @par
 * @code
 * #include <cub/cub.cuh> // or equivalently <cub/block/block_merge_sort.cuh>
 *
 * constexpr int BLOCK_THREADS = 256;
 * constexpr int ItemsPerThread = 9;
 *
 * class BlockMergeSort : public BlockMergeSortStrategy<int,
 *                                                      cub::NullType,
 *                                                      BLOCK_THREADS,
 *                                                      ItemsPerThread,
 *                                                      BlockMergeSort>
 * {
 *   using BlockMergeSortStrategyT =
 *     BlockMergeSortStrategy<int,
 *                            cub::NullType,
 *                            BLOCK_THREADS,
 *                            ItemsPerThread,
 *                            BlockMergeSort>;
 * public:
 *   __device__ __forceinline__ explicit BlockMergeSort(
 *     typename BlockMergeSortStrategyT::TempStorage &temp_storage)
 *       : BlockMergeSortStrategyT(temp_storage, threadIdx.x)
 *   {}
 *
 *   __device__ __forceinline__ void SyncImplementation() const
 *   {
 *     __syncthreads();
 *   }
 * };
 * @endcode
 *
 * @tparam KeyT
 *   KeyT type
 *
 * @tparam ValueT
 *   ValueT type. cub::NullType indicates a keys-only sort
 *
 * @tparam SynchronizationPolicy
 *   Provides a way of synchronizing threads. Should be derived from
 *   `BlockMergeSortStrategy`.
 */
template <typename KeyT, typename ValueT, int NumThreads, int ItemsPerThread, typename SynchronizationPolicy>
class BlockMergeSortStrategy
{
  static_assert(::cuda::is_power_of_two(NumThreads), "NumThreads must be a power of two");

private:
  static constexpr int ITEMS_PER_TILE = ItemsPerThread * NumThreads;

  // Whether or not there are values to be trucked along with keys
  static constexpr bool KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  /// Shared memory type required by this thread block
  union _TempStorage
  {
    KeyT keys_shared[ITEMS_PER_TILE + 1];
    ValueT items_shared[ITEMS_PER_TILE + 1];
  }; // union TempStorage
#endif // _CCCL_DOXYGEN_INVOKED

  /// Shared storage reference
  _TempStorage& temp_storage;

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  const unsigned int linear_tid;

public:
  /// \smemstorage{BlockMergeSort}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  BlockMergeSortStrategy() = delete;
  explicit _CCCL_DEVICE _CCCL_FORCEINLINE BlockMergeSortStrategy(unsigned int linear_tid)
      : temp_storage(PrivateStorage())
      , linear_tid(linear_tid)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE BlockMergeSortStrategy(TempStorage& temp_storage, unsigned int linear_tid)
      : temp_storage(temp_storage.Alias())
      , linear_tid(linear_tid)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int get_linear_tid() const
  {
    return linear_tid;
  }

  /**
   * @brief Sorts items partitioned across a CUDA thread block using
   *        a merge sorting method.
   *
   * @par
   * Sort is not guaranteed to be stable. That is, suppose that i and j are
   * equivalent: neither one is less than the other. It is not guaranteed
   * that the relative order of these two elements will be preserved by sort.
   *
   * @tparam CompareOp
   *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`.
   *   `CompareOp` is a model of [Strict Weak Ordering].
   *
   * @param[in,out] keys
   *   Keys to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   */
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Sort(KeyT (&keys)[ItemsPerThread], CompareOp compare_op)
  {
    ValueT items[ItemsPerThread];
    Sort<CompareOp, false>(keys, items, compare_op, ITEMS_PER_TILE, keys[0]);
  }

  /**
   * @brief Sorts items partitioned across a CUDA thread block using
   *        a merge sorting method.
   *
   * @par
   * - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
   *   are equivalent: neither one is less than the other. It is not guaranteed
   *   that the relative order of these two elements will be preserved by sort.
   * - The value of `oob_default` is assigned to all elements that are out of
   *   `valid_items` boundaries. It's expected that `oob_default` is ordered
   *   after any value in the `valid_items` boundaries. The algorithm always
   *   sorts a fixed amount of elements, which is equal to
   *   `ItemsPerThread * BLOCK_THREADS`. If there is a value that is ordered
   *   after `oob_default`, it won't be placed within `valid_items` boundaries.
   *
   * @tparam CompareOp
   *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`.
   *   `CompareOp` is a model of [Strict Weak Ordering].
   *
   * @param[in,out] keys
   *   Keys to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * @param[in] valid_items
   *   Number of valid items to sort
   *
   * @param[in] oob_default
   *   Default value to assign out-of-bound items
   *
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   */
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int valid_items, KeyT oob_default)
  {
    ValueT items[ItemsPerThread];
    Sort<CompareOp, true>(keys, items, compare_op, valid_items, oob_default);
  }

  /**
   * @brief Sorts items partitioned across a CUDA thread block using a merge sorting method.
   *
   * @par
   * Sort is not guaranteed to be stable. That is, suppose that `i` and `j` are
   * equivalent: neither one is less than the other. It is not guaranteed
   * that the relative order of these two elements will be preserved by sort.
   *
   * @tparam CompareOp
   *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`.
   *   `CompareOp` is a model of [Strict Weak Ordering].
   *
   * @param[in,out] keys
   *   Keys to sort
   *
   * @param[in,out] items
   *   Values to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   */
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[ItemsPerThread], ValueT (&items)[ItemsPerThread], CompareOp compare_op)
  {
    Sort<CompareOp, false>(keys, items, compare_op, ITEMS_PER_TILE, keys[0]);
  }

  /**
   * @brief Sorts items partitioned across a CUDA thread block using
   *        a merge sorting method.
   *
   * @par
   * - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
   *   are equivalent: neither one is less than the other. It is not guaranteed
   *   that the relative order of these two elements will be preserved by sort.
   * - The value of `oob_default` is assigned to all elements that are out of
   *   `valid_items` boundaries. It's expected that `oob_default` is ordered
   *   after any value in the `valid_items` boundaries. The algorithm always
   *   sorts a fixed amount of elements, which is equal to
   *   `ItemsPerThread * BLOCK_THREADS`. If there is a value that is ordered
   *   after `oob_default`, it won't be placed within `valid_items` boundaries.
   *
   * @tparam CompareOp
   *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`
   *   `CompareOp` is a model of [Strict Weak Ordering].
   *
   * @tparam IS_LAST_TILE
   *   True if `valid_items` isn't equal to the `ITEMS_PER_TILE`
   *
   * @param[in,out] keys
   *   Keys to sort
   *
   * @param[in,out] items
   *   Values to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * @param[in] valid_items
   *   Number of valid items to sort
   *
   * @param[in] oob_default
   *   Default value to assign out-of-bound items
   *
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   */
  template <typename CompareOp, bool IS_LAST_TILE = true>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[ItemsPerThread],
       ValueT (&items)[ItemsPerThread],
       CompareOp compare_op,
       int valid_items,
       KeyT oob_default)
  {
    if constexpr (IS_LAST_TILE)
    {
      // if last tile, find valid max_key
      // and fill the remaining keys with it
      //
      KeyT max_key = oob_default;

      _CCCL_SORT_MAYBE_UNROLL()
      for (int item = 1; item < ItemsPerThread; ++item)
      {
        if (ItemsPerThread * linear_tid + item < valid_items)
        {
          max_key = compare_op(max_key, keys[item]) ? keys[item] : max_key;
        }
        else
        {
          keys[item] = max_key;
        }
      }
    }

    // if first element of thread is in input range, stable sort items
    //
    if (!IS_LAST_TILE || ItemsPerThread * linear_tid < valid_items)
    {
      StableOddEvenSort(keys, items, compare_op);
    }

    // each thread has sorted keys
    // merge sort keys in shared memory
    //
    for (int target_merged_threads_number = 2; target_merged_threads_number <= NumThreads;
         target_merged_threads_number *= 2)
    {
      const int merged_threads_number = target_merged_threads_number / 2;
      const int mask                  = target_merged_threads_number - 1;

      Sync();

      // store keys in shmem
      //
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        int idx                       = ItemsPerThread * linear_tid + item;
        temp_storage.keys_shared[idx] = keys[item];
      }

      Sync();

      int indices[ItemsPerThread];

      const int first_thread_idx_in_thread_group_being_merged = ~mask & linear_tid;
      const int start = ItemsPerThread * first_thread_idx_in_thread_group_being_merged;
      const int size  = ItemsPerThread * merged_threads_number;

      const int thread_idx_in_thread_group_being_merged = mask & linear_tid;

      const int diag = (::cuda::std::min) (valid_items, ItemsPerThread * thread_idx_in_thread_group_being_merged);

      const int keys1_beg = (::cuda::std::min) (valid_items, start);
      const int keys1_end = (::cuda::std::min) (valid_items, keys1_beg + size);
      const int keys2_beg = keys1_end;
      const int keys2_end = (::cuda::std::min) (valid_items, keys2_beg + size);

      const int keys1_count = keys1_end - keys1_beg;
      const int keys2_count = keys2_end - keys2_beg;

      const int partition_diag = MergePath(
        &temp_storage.keys_shared[keys1_beg],
        &temp_storage.keys_shared[keys2_beg],
        keys1_count,
        keys2_count,
        diag,
        compare_op);

      const int keys1_beg_loc   = keys1_beg + partition_diag;
      const int keys1_end_loc   = keys1_end;
      const int keys2_beg_loc   = keys2_beg + diag - partition_diag;
      const int keys2_end_loc   = keys2_end;
      const int keys1_count_loc = keys1_end_loc - keys1_beg_loc;
      const int keys2_count_loc = keys2_end_loc - keys2_beg_loc;
      SerialMerge(
        &temp_storage.keys_shared[0],
        keys1_beg_loc,
        keys2_beg_loc,
        keys1_count_loc,
        keys2_count_loc,
        keys,
        indices,
        compare_op,
        oob_default);

      if constexpr (!KEYS_ONLY)
      {
        Sync();

        // store keys in shmem
        //
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int item = 0; item < ItemsPerThread; ++item)
        {
          int idx                        = ItemsPerThread * linear_tid + item;
          temp_storage.items_shared[idx] = items[item];
        }

        Sync();

        // gather items from shmem
        //
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int item = 0; item < ItemsPerThread; ++item)
        {
          items[item] = temp_storage.items_shared[indices[item]];
        }
      }
    }
  } // func block_merge_sort

  /**
   * @brief Sorts items partitioned across a CUDA thread block using
   *        a merge sorting method.
   *
   * @par
   * StableSort is stable: it preserves the relative ordering of equivalent
   * elements. That is, if `x` and `y` are elements such that `x` precedes `y`,
   * and if the two elements are equivalent (neither `x < y` nor `y < x`) then
   * a postcondition of StableSort is that `x` still precedes `y`.
   *
   * @tparam CompareOp
   *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`.
   *   `CompareOp` is a model of [Strict Weak Ordering].
   *
   * @param[in,out] keys
   *   Keys to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   */
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void StableSort(KeyT (&keys)[ItemsPerThread], CompareOp compare_op)
  {
    Sort(keys, compare_op);
  }

  /**
   * @brief Sorts items partitioned across a CUDA thread block using
   *        a merge sorting method.
   *
   * @par
   * StableSort is stable: it preserves the relative ordering of equivalent
   * elements. That is, if `x` and `y` are elements such that `x` precedes `y`,
   * and if the two elements are equivalent (neither `x < y` nor `y < x`) then
   * a postcondition of StableSort is that `x` still precedes `y`.
   *
   * @tparam CompareOp
   *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`.
   *   `CompareOp` is a model of [Strict Weak Ordering].
   *
   * @param[in,out] keys
   *   Keys to sort
   *
   * @param[in,out] items
   *   Values to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   */
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  StableSort(KeyT (&keys)[ItemsPerThread], ValueT (&items)[ItemsPerThread], CompareOp compare_op)
  {
    Sort(keys, items, compare_op);
  }

  /**
   * @brief Sorts items partitioned across a CUDA thread block using
   *        a merge sorting method.
   *
   * @par
   * - StableSort is stable: it preserves the relative ordering of equivalent
   *   elements. That is, if `x` and `y` are elements such that `x` precedes
   *   `y`, and if the two elements are equivalent (neither `x < y` nor `y < x`)
   *   then a postcondition of StableSort is that `x` still precedes `y`.
   * - The value of `oob_default` is assigned to all elements that are out of
   *   `valid_items` boundaries. It's expected that `oob_default` is ordered
   *   after any value in the `valid_items` boundaries. The algorithm always
   *   sorts a fixed amount of elements, which is equal to
   *   `ItemsPerThread * BLOCK_THREADS`.
   *   If there is a value that is ordered after `oob_default`, it won't be
   *   placed within `valid_items` boundaries.
   *
   * @tparam CompareOp
   *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`.
   *   `CompareOp` is a model of [Strict Weak Ordering].
   *
   * @param[in,out] keys
   *   Keys to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * @param[in] valid_items
   *   Number of valid items to sort
   *
   * @param[in] oob_default
   *   Default value to assign out-of-bound items
   *
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   */
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  StableSort(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int valid_items, KeyT oob_default)
  {
    Sort(keys, compare_op, valid_items, oob_default);
  }

  /**
   * @brief Sorts items partitioned across a CUDA thread block using
   *        a merge sorting method.
   *
   * @par
   * - StableSort is stable: it preserves the relative ordering of equivalent
   *   elements. That is, if `x` and `y` are elements such that `x` precedes
   *   `y`, and if the two elements are equivalent (neither `x < y` nor `y < x`)
   *   then a postcondition of StableSort is that `x` still precedes `y`.
   * - The value of `oob_default` is assigned to all elements that are out of
   *   `valid_items` boundaries. It's expected that `oob_default` is ordered
   *   after any value in the `valid_items` boundaries. The algorithm always
   *   sorts a fixed amount of elements, which is equal to
   *   `ItemsPerThread * BLOCK_THREADS`. If there is a value that is ordered
   *   after `oob_default`, it won't be placed within `valid_items` boundaries.
   *
   * @tparam CompareOp
   *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`.
   *   `CompareOp` is a model of [Strict Weak Ordering].
   *
   * @tparam IS_LAST_TILE
   *   True if `valid_items` isn't equal to the `ITEMS_PER_TILE`
   *
   * @param[in,out] keys
   *   Keys to sort
   *
   * @param[in,out] items
   *   Values to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * @param[in] valid_items
   *   Number of valid items to sort
   *
   * @param[in] oob_default
   *   Default value to assign out-of-bound items
   *
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   */
  template <typename CompareOp, bool IS_LAST_TILE = true>
  _CCCL_DEVICE _CCCL_FORCEINLINE void StableSort(
    KeyT (&keys)[ItemsPerThread],
    ValueT (&items)[ItemsPerThread],
    CompareOp compare_op,
    int valid_items,
    KeyT oob_default)
  {
    Sort<CompareOp, IS_LAST_TILE>(keys, items, compare_op, valid_items, oob_default);
  }

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE void Sync() const
  {
    static_cast<const SynchronizationPolicy*>(this)->SyncImplementation();
  }
};

/**
 * @brief The BlockMergeSort class provides methods for sorting items
 *        partitioned across a CUDA thread block using a merge sorting method.
 *
 * @tparam KeyT
 *   KeyT type
 *
 * @tparam BLOCK_DIM_X
 *   The thread block length in threads along the X dimension
 *
 * @tparam ItemsPerThread
 *   The number of items per thread
 *
 * @tparam ValueT
 *   **[optional]** ValueT type (default: `cub::NullType`, which indicates
 *   a keys-only sort)
 *
 * @tparam BLOCK_DIM_Y
 *   **[optional]** The thread block length in threads along the Y dimension
 *   (default: 1)
 *
 * @tparam BLOCK_DIM_Z
 *   **[optional]** The thread block length in threads along the Z dimension
 *   (default: 1)
 *
 * @par Overview
 *   BlockMergeSort arranges items into ascending order using a comparison
 *   functor with less-than semantics. Merge sort can handle arbitrary types
 *   and comparison functors, but is slower than BlockRadixSort when sorting
 *   arithmetic types into ascending/descending order.
 *
 * @par A Simple Example
 * @blockcollective{BlockMergeSort}
 * @par
 * The code snippet below illustrates a sort of 512 integer keys that are
 * partitioned across 128 threads * where each thread owns 4 consecutive items.
 * @par
 * @code
 * #include <cub/cub.cuh>  // or equivalently <cub/block/block_merge_sort.cuh>
 *
 * struct CustomLess
 * {
 *   template <typename DataType>
 *   __device__ bool operator()(const DataType &lhs, const DataType &rhs)
 *   {
 *     return lhs < rhs;
 *   }
 * };
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Specialize BlockMergeSort for a 1D block of 128 threads owning 4 integer items each
 *     using BlockMergeSort = cub::BlockMergeSort<int, 128, 4>;
 *
 *     // Allocate shared memory for BlockMergeSort
 *     __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;
 *
 *     // Obtain a segment of consecutive items that are blocked across threads
 *     int thread_keys[4];
 *     ...
 *
 *     BlockMergeSort(temp_storage_shuffle).Sort(thread_keys, CustomLess());
 *     ...
 * }
 * @endcode
 * @par
 * Suppose the set of input `thread_keys` across the block of threads is
 * `{ [0,511,1,510], [2,509,3,508], [4,507,5,506], ..., [254,257,255,256] }`.
 * The corresponding output `thread_keys` in those threads will be
 * `{ [0,1,2,3], [4,5,6,7], [8,9,10,11], ..., [508,509,510,511] }`.
 *
 * @par Re-using dynamically allocating shared memory
 * The ``block/example_block_reduce_dyn_smem.cu`` example illustrates usage of
 * dynamically shared memory with BlockReduce and how to re-purpose
 * the same memory region.
 *
 * This example can be easily adapted to the storage required by BlockMergeSort.
 */
template <typename KeyT, int BlockDimX, int ItemsPerThread, typename ValueT = NullType, int BlockDimY = 1, int BlockDimZ = 1>
class BlockMergeSort
    : public BlockMergeSortStrategy<KeyT,
                                    ValueT,
                                    BlockDimX * BlockDimY * BlockDimZ,
                                    ItemsPerThread,
                                    BlockMergeSort<KeyT, BlockDimX, ItemsPerThread, ValueT, BlockDimY, BlockDimZ>>
{
private:
  // The thread block size in threads
  static constexpr int BLOCK_THREADS  = BlockDimX * BlockDimY * BlockDimZ;
  static constexpr int ITEMS_PER_TILE = ItemsPerThread * BLOCK_THREADS;

  using BlockMergeSortStrategyT = BlockMergeSortStrategy<KeyT, ValueT, BLOCK_THREADS, ItemsPerThread, BlockMergeSort>;

public:
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockMergeSort()
      : BlockMergeSortStrategyT(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE explicit BlockMergeSort(typename BlockMergeSortStrategyT::TempStorage& temp_storage)
      : BlockMergeSortStrategyT(temp_storage, RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE void SyncImplementation() const
  {
    __syncthreads();
  }

  friend BlockMergeSortStrategyT;
};

CUB_NAMESPACE_END
