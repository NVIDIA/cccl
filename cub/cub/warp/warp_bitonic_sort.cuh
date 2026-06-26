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

#include <cub/util_arch.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/void_t.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <class T, typename = void>
inline constexpr bool has_native_shfl_v = false;

template <class T>
inline constexpr bool has_native_shfl_v<T, ::cuda::std::void_t<decltype(__shfl_sync(0u, T{}, 0))>> = true;

//! @rst
//! The WarpBitonicSort class provides methods for sorting items partitioned across a CUDA warp
//! using a bitonic sorting network.
//!
//! Overview
//! ++++++++++++++++
//!
//!   WarpBitonicSort arranges items into ascending order using a comparison functor with less-than
//!   semantics.
//!
//! A Simple Example
//! ++++++++++++++++
//!
//! The code snippet below illustrates a sort of 64 integer keys that are partitioned across
//! 32 threads where each thread owns 2 items in a striped arrangement.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>  // or equivalently <cub/warp/warp_bitonic_sort.cuh>
//!
//!    struct CustomLess
//!    {
//!      template <typename DataType>
//!      __device__ bool operator()(const DataType &lhs, const DataType &rhs) const
//!      {
//!        return lhs < rhs;
//!      }
//!    };
//!
//!    __global__ void ExampleKernel(...)
//!    {
//!        constexpr int items_per_thread = 2;
//!
//!        using WarpBitonicSortT = cub::detail::WarpBitonicSort<items_per_thread, int>;
//!
//!        int thread_keys[items_per_thread];
//!        // ...
//!
//!        WarpBitonicSortT{}.Sort(thread_keys, CustomLess());
//!    }
//!
//! Suppose the set of input ``thread_keys`` across a warp of threads is
//! ``{ [0,63], [1,62], [2,61], ..., [31,32] }``.
//! The corresponding output ``thread_keys`` in those threads will be
//! ``{ [0,32], [1,33], [2,34], ..., [31,63] }``.
//! Note keys are in a :ref:`striped arrangement <flexible-data-arrangement>` across warp lanes.
//!
//! @endrst
//!
//! @tparam ITEMS_PER_THREAD
//!   The number of items per thread.
//!
//! @tparam KeyT
//!   Key type
//!
//! @tparam ValueT
//!   <b>[optional]</b> Value type (default: cub::NullType, which indicates a keys-only sort)
//!
template <int ITEMS_PER_THREAD, typename KeyT, typename ValueT = NullType>
class WarpBitonicSort
{
public:
  //! @brief Sorts keys across a warp of threads using bitonic sorting network.
  //!
  //! @par
  //! - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
  //!   are equivalent: neither one is less than the other. It is not guaranteed
  //!   that the relative order of these two elements will be preserved by sort.
  //!
  //! @tparam CompareOp Comparison functor type
  //!
  //! @param[in,out] keys Keys to sort, in striped arrangement
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void Sort(KeyT (&keys)[ITEMS_PER_THREAD], CompareOp compare_op)
  {
    Sort_<CompareOp, false>(keys, nullptr, compare_op);
  }

  //! @brief Sorts keys across a warp of threads using bitonic sorting network.
  //!
  //! @par
  //! - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
  //!   are equivalent: neither one is less than the other. It is not guaranteed
  //!   that the relative order of these two elements will be preserved by sort.
  //! - The value of `oob_default` is assigned to all elements that are out of
  //!   `valid_items` boundaries. It's expected that `oob_default` is ordered
  //!   after any value in the `valid_items` boundaries.
  //!
  //! @tparam CompareOp Comparison functor type
  //!
  //! @param[in,out] keys Keys to sort, in striped arrangement
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  //! @param[in] valid_items Total number of valid items across the warp
  //! @param[in] oob_default Default value for out-of-bound items
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD], CompareOp compare_op, int valid_items, KeyT oob_default)
  {
    const int lane = threadIdx.x % WARP_THREADS_;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      if (i * WARP_THREADS_ + lane >= valid_items)
      {
        keys[i] = oob_default;
      }
    }
    Sort_<CompareOp, false>(keys, nullptr, compare_op);
  }

  //! @brief Sorts keys across a warp of threads using bitonic sorting network.
  //!
  //! @par
  //! - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
  //!   are equivalent: neither one is less than the other. It is not guaranteed
  //!   that the relative order of these two elements will be preserved by sort.
  //!
  //! @tparam CompareOp Comparison functor type
  //!
  //! @param[in,out] keys Keys to sort, in striped arrangement
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  //! @param[in] valid_items Total number of valid items across the warp
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void Sort(KeyT (&keys)[ITEMS_PER_THREAD], CompareOp compare_op, int valid_items)
  {
    Sort_<CompareOp, false>(keys, nullptr, compare_op, valid_items);
  }

  //! @brief Sorts key-value pairs across a warp of threads using bitonic sorting network.
  //!
  //! @par
  //! - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
  //!   are equivalent: neither one is less than the other. It is not guaranteed
  //!   that the relative order of these two elements will be preserved by sort.
  //!
  //! @tparam CompareOp Comparison functor type
  //!
  //! @param[in,out] keys Keys to sort, in striped arrangement
  //! @param[in,out] values Values to sort (reordered to match key order)
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], CompareOp compare_op)
  {
    Sort_<CompareOp, false>(keys, values, compare_op);
  }

  //! @brief Sorts key-value pairs across a warp of threads using bitonic sorting network.
  //!
  //! @par
  //! - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
  //!   are equivalent: neither one is less than the other. It is not guaranteed
  //!   that the relative order of these two elements will be preserved by sort.
  //! - The value of `oob_default` is assigned to all elements that are out of
  //!   `valid_items` boundaries. It's expected that `oob_default` is ordered
  //!   after any value in the `valid_items` boundaries.
  //!
  //! @tparam CompareOp Comparison functor type
  //!
  //! @param[in,out] keys Keys to sort, in striped arrangement
  //! @param[in,out] values Values to sort (reordered to match key order)
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  //! @param[in] valid_items Total number of valid items across the warp
  //! @param[in] oob_default Default value for out-of-bound keys
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD],
       ValueT (&values)[ITEMS_PER_THREAD],
       CompareOp compare_op,
       int valid_items,
       KeyT oob_default)
  {
    const int lane = threadIdx.x % WARP_THREADS_;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      if (i * WARP_THREADS_ + lane >= valid_items)
      {
        keys[i] = oob_default;
      }
    }
    Sort_<CompareOp, false>(keys, values, compare_op);
  }

  //! @brief Sorts key-value pairs across a warp of threads using bitonic sorting network.
  //!
  //! @par
  //! - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
  //!   are equivalent: neither one is less than the other. It is not guaranteed
  //!   that the relative order of these two elements will be preserved by sort.
  //!
  //! @tparam CompareOp Comparison functor type
  //!
  //! @param[in,out] keys Keys to sort, in striped arrangement
  //! @param[in,out] values Values to sort (reordered to match key order)
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  //! @param[in] valid_items Total number of valid items across the warp
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], CompareOp compare_op, int valid_items)
  {
    Sort_<CompareOp, false>(keys, values, compare_op, valid_items);
  }

private:
  template <int, typename, typename>
  friend class WarpBitonicSort;

  static constexpr int WARP_THREADS_ = detail::warp_threads;
  static constexpr bool KEYS_ONLY_   = ::cuda::std::is_same_v<ValueT, NullType>;

  template <typename CompareOp, bool REVERSE>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Sort_(KeyT* _CCCL_RESTRICT keys, ValueT* _CCCL_RESTRICT values, CompareOp compare_op)
  {
    constexpr int first_half_len  = ::cuda::prev_power_of_two(ITEMS_PER_THREAD - 1);
    constexpr int second_half_len = ITEMS_PER_THREAD - first_half_len;

    WarpBitonicSort<first_half_len, KeyT, ValueT>::template Sort_<CompareOp, !REVERSE>(keys, values, compare_op);
    WarpBitonicSort<second_half_len, KeyT, ValueT>::template Sort_<CompareOp, REVERSE>(
      keys + first_half_len, (KEYS_ONLY_ ? nullptr : values + first_half_len), compare_op);
    Merge_<CompareOp, REVERSE>(keys, values, compare_op);
  }

  //! @brief Merges a bitonic sequence of key-value pairs across a warp of threads into a monotonic sequence.
  //!
  //! @tparam CompareOp Comparison functor type
  //! @tparam REVERSE If true, results are in reverse order
  //!
  //! @param[in,out] keys Keys to merge, in striped arrangement. Input keys must form a bitonic sequence meeting one of
  //! these conditions:
  //!   (1) `ITEMS_PER_THREAD` is a power of 2
  //!   (2) If `ITEMS_PER_THREAD` is not a power of 2 and `REVERSE` is false: the sequence must be reverse-sorted first,
  //!       then sorted
  //!   (3) If `ITEMS_PER_THREAD` is not a power of 2 and `REVERSE` is true: the sequence must be sorted first, then
  //!       reverse-sorted
  //! @param[in,out] values Values to merge (reordered to match key order)
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  template <typename CompareOp, bool REVERSE>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Merge_(KeyT* _CCCL_RESTRICT keys, ValueT* _CCCL_RESTRICT values, CompareOp compare_op)
  {
    constexpr int first_half_len  = ::cuda::prev_power_of_two(ITEMS_PER_THREAD - 1);
    constexpr int second_half_len = ITEMS_PER_THREAD - first_half_len;
    constexpr int stride          = first_half_len;
    static_assert(first_half_len >= second_half_len);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < second_half_len; ++i)
    {
      const int other_i = i + stride;
      KeyT& key         = keys[i];
      KeyT& other_key   = keys[other_i];
      bool should_swap;
      if constexpr (REVERSE)
      {
        should_swap = compare_op(key, other_key);
      }
      else
      {
        should_swap = compare_op(other_key, key);
      }
      if (should_swap)
      {
        KeyT tmp_k = key;
        key        = other_key;
        other_key  = tmp_k;

        if constexpr (!KEYS_ONLY_)
        {
          ValueT tmp_v    = values[i];
          values[i]       = values[other_i];
          values[other_i] = tmp_v;
        }
      }
    }

    WarpBitonicSort<first_half_len, KeyT, ValueT>::template Merge_<CompareOp, REVERSE>(keys, values, compare_op);
    WarpBitonicSort<second_half_len, KeyT, ValueT>::template Merge_<CompareOp, REVERSE>(
      keys + first_half_len, (KEYS_ONLY_ ? nullptr : values + first_half_len), compare_op);
  }

  template <typename CompareOp, bool REVERSE>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Sort_(KeyT* _CCCL_RESTRICT keys, ValueT* _CCCL_RESTRICT values, CompareOp compare_op, int valid_items)
  {
    constexpr int first_half_len  = ::cuda::prev_power_of_two(ITEMS_PER_THREAD - 1);
    constexpr int second_half_len = ITEMS_PER_THREAD - first_half_len;

    if (valid_items > first_half_len * WARP_THREADS_)
    {
      WarpBitonicSort<first_half_len, KeyT, ValueT>::template Sort_<CompareOp, !REVERSE>(keys, values, compare_op);
      WarpBitonicSort<second_half_len, KeyT, ValueT>::template Sort_<CompareOp, REVERSE>(
        keys + first_half_len,
        (KEYS_ONLY_ ? nullptr : values + first_half_len),
        compare_op,
        valid_items - first_half_len * WARP_THREADS_);
      Merge_<CompareOp, REVERSE>(keys, values, compare_op, valid_items);
    }
    else
    {
      WarpBitonicSort<first_half_len, KeyT, ValueT>::template Sort_<CompareOp, REVERSE>(
        keys, values, compare_op, valid_items);
    }
  }

  //! @brief Merges a bitonic sequence of key-value pairs across a warp of threads into a monotonic sequence.
  //!
  //! @tparam CompareOp Comparison functor type
  //! @tparam REVERSE If true, results are in reverse order
  //!
  //! @param[in,out] keys Keys to merge, in striped arrangement. Input keys must form a bitonic sequence meeting one of
  //! these conditions:
  //!   (1) If `REVERSE` is false: the sequence must be reverse-sorted first, then sorted
  //!   (2) If `REVERSE` is true: the sequence must be sorted first, then reverse-sorted
  //! @param[in,out] values Values to merge (reordered to match key order)
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  //! @param[in] valid_items Total number of valid items across the warp
  template <typename CompareOp, bool REVERSE>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Merge_(KeyT* _CCCL_RESTRICT keys, ValueT* _CCCL_RESTRICT values, CompareOp compare_op, int valid_items)
  {
    constexpr int first_half_len  = ::cuda::prev_power_of_two(ITEMS_PER_THREAD - 1);
    constexpr int second_half_len = ITEMS_PER_THREAD - first_half_len;
    constexpr int stride          = first_half_len;
    static_assert(first_half_len >= second_half_len);

    const int lane = threadIdx.x % WARP_THREADS_;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < second_half_len; ++i)
    {
      const int other_i = i + stride;
      KeyT& key         = keys[i];
      KeyT& other_key   = keys[other_i];
      bool should_swap;
      if constexpr (REVERSE)
      {
        should_swap = compare_op(key, other_key);
      }
      else
      {
        should_swap = compare_op(other_key, key);
      }
      if (should_swap && other_i * WARP_THREADS_ + lane < valid_items)
      {
        KeyT tmp_k = key;
        key        = other_key;
        other_key  = tmp_k;

        if constexpr (!KEYS_ONLY_)
        {
          ValueT tmp_v    = values[i];
          values[i]       = values[other_i];
          values[other_i] = tmp_v;
        }
      }
    }

    if (valid_items > first_half_len * WARP_THREADS_)
    {
      WarpBitonicSort<first_half_len, KeyT, ValueT>::template Merge_<CompareOp, REVERSE>(keys, values, compare_op);
      WarpBitonicSort<second_half_len, KeyT, ValueT>::template Merge_<CompareOp, REVERSE>(
        keys + first_half_len,
        (KEYS_ONLY_ ? nullptr : values + first_half_len),
        compare_op,
        valid_items - first_half_len * WARP_THREADS_);
    }
    else
    {
      WarpBitonicSort<first_half_len, KeyT, ValueT>::template Merge_<CompareOp, REVERSE>(
        keys, values, compare_op, valid_items);
    }
  }
};

// When each thread holds a single item, the bitonic sort operates entirely across warp lanes
// using shuffle instructions.
template <typename KeyT, typename ValueT>
class WarpBitonicSort<1, KeyT, ValueT>
{
public:
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void Sort(KeyT (&keys)[1], CompareOp compare_op)
  {
    Sort_<CompareOp, false>(keys, nullptr, compare_op);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Sort(KeyT (&keys)[1], CompareOp compare_op, int valid_items, KeyT oob_default)
  {
    const int lane = threadIdx.x % WARP_THREADS_;
    if (lane >= valid_items)
    {
      keys[0] = oob_default;
    }
    Sort_<CompareOp, false>(keys, nullptr, compare_op);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void Sort(KeyT (&keys)[1], CompareOp compare_op, int valid_items)
  {
    Sort_<CompareOp, false>(keys, nullptr, compare_op, valid_items);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void Sort(KeyT (&keys)[1], ValueT (&values)[1], CompareOp compare_op)
  {
    Sort_<CompareOp, false>(keys, values, compare_op);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Sort(KeyT (&keys)[1], ValueT (&values)[1], CompareOp compare_op, int valid_items, KeyT oob_default)
  {
    const int lane = threadIdx.x % WARP_THREADS_;
    if (lane >= valid_items)
    {
      keys[0] = oob_default;
    }
    Sort_<CompareOp, false>(keys, values, compare_op);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Sort(KeyT (&keys)[1], ValueT (&values)[1], CompareOp compare_op, int valid_items)
  {
    Sort_<CompareOp, false>(keys, values, compare_op, valid_items);
  }

private:
  template <int, typename, typename>
  friend class WarpBitonicSort;

  static constexpr int WARP_THREADS_            = detail::warp_threads;
  static constexpr unsigned int FULL_WARP_MASK_ = 0xFFFFFFFFu;
  static constexpr bool KEYS_ONLY_              = ::cuda::std::is_same_v<ValueT, NullType>;

  template <typename CompareOp, bool REVERSE>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Sort_(KeyT* _CCCL_RESTRICT keys, ValueT* _CCCL_RESTRICT values, CompareOp compare_op)
  {
    // Non-recursive implementation for 32 inputs consists of log2(32)=5 stages.
    MergeImpl_<CompareOp, REVERSE, true, 0>(keys, values, compare_op);
    MergeImpl_<CompareOp, REVERSE, true, 1>(keys, values, compare_op);
    MergeImpl_<CompareOp, REVERSE, true, 2>(keys, values, compare_op);
    MergeImpl_<CompareOp, REVERSE, true, 3>(keys, values, compare_op);
    MergeImpl_<CompareOp, REVERSE, true, 4>(keys, values, compare_op);
  }

  template <typename CompareOp, bool REVERSE>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Merge_(KeyT* _CCCL_RESTRICT keys, ValueT* _CCCL_RESTRICT values, CompareOp compare_op)
  {
    MergeImpl_<CompareOp, REVERSE, true, 4>(keys, values, compare_op);
  }

  template <typename CompareOp, bool REVERSE>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Sort_(KeyT* _CCCL_RESTRICT keys, ValueT* _CCCL_RESTRICT values, CompareOp compare_op, int valid_items)
  {
    MergeImpl_<CompareOp, REVERSE, false, 0>(keys, values, compare_op, valid_items);
    MergeImpl_<CompareOp, REVERSE, false, 1>(keys, values, compare_op, valid_items);
    MergeImpl_<CompareOp, REVERSE, false, 2>(keys, values, compare_op, valid_items);
    MergeImpl_<CompareOp, REVERSE, false, 3>(keys, values, compare_op, valid_items);
    MergeImpl_<CompareOp, REVERSE, false, 4>(keys, values, compare_op, valid_items);
  }

  template <typename CompareOp, bool REVERSE>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  Merge_(KeyT* _CCCL_RESTRICT keys, ValueT* _CCCL_RESTRICT values, CompareOp compare_op, int valid_items)
  {
    MergeImpl_<CompareOp, REVERSE, false, 4>(keys, values, compare_op, valid_items);
  }

  //! @brief Single stage of the bitonic sorting network operating across warp lanes via shuffles.
  //!
  //! @tparam FULL If true, all items are valid (no boundary checks: ``valid_items`` is not used)
  //! @tparam STAGE Stage index (0-4 for a 32-thread warp)
  template <typename CompareOp, bool REVERSE, bool FULL, int STAGE>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  MergeImpl_(KeyT* _CCCL_RESTRICT keys, ValueT* _CCCL_RESTRICT values, CompareOp compare_op, int valid_items = -1)
  {
    const int lane = threadIdx.x % WARP_THREADS_;

    // Each stage divides the inputs into groups and sorts within each group.
    // Sort direction of each group should be adjusted to maintain the bitonic property.
    bool group_reverse = REVERSE;
    if constexpr (STAGE == 4)
    {
      // The last stage contains only one group, and the sort direction is just REVERSE
    }
    else if constexpr (FULL)
    {
      // Group size doubles while the number of groups halves with each stage: stage 0 sorts
      // 16 groups of 2 elements, stage 1 sorts 8 groups of 4 elements, and so on.
      //
      // Group ID (starting from 0) is "lane >> (STAGE + 1)".
      // Odd groups sort in reverse order.
      // So group_reverse = (lane >> (STAGE + 1)) & 1, equal to the following:
      group_reverse ^= static_cast<bool>((lane >> STAGE) & 2);
    }
    else
    {
      // To allow out-of-bound pairs to be safely skipped:
      //
      // 1. Reverse direction on odd stages.
      //
      // 2. Reverse direction for groups whose ID has an odd number of set bits
      //    (i.e., groups 1, 2, 4, 7, etc.). Because group ID < 16 (at most 4 set
      //    bits), this means IDs with 1 or 3 set bits.
      group_reverse ^= static_cast<bool>(STAGE & 1);
      int num_set_bits = __popc((lane >> (STAGE + 1)));
      group_reverse ^= (num_set_bits == 1 || num_set_bits == 3);
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int stride = (1 << STAGE); stride > 0; stride /= 2)
    {
      // Because stride is a power of 2, it has exactly one bit (denoted as "stride bit") set. A thread
      // pair consists of two threads with lane IDs "lane" and "lane ^ stride", differing only in
      // the stride bit. The thread with the stride bit set (lane & stride != 0) has the larger
      // lane ID. From that thread's perspective, the sorting direction is reversed.
      const bool has_larger_lane_id = lane & stride;
      const bool reverse            = group_reverse ^ has_larger_lane_id;

      KeyT& key = *keys;
      KeyT other_key;
      if constexpr (has_native_shfl_v<KeyT>)
      {
        other_key = __shfl_xor_sync(FULL_WARP_MASK_, key, stride);
      }
      else
      {
        other_key = ::cuda::device::warp_shuffle_xor(key, stride);
      }

      [[maybe_unused]] ValueT other_value;
      if constexpr (!KEYS_ONLY_)
      {
        if constexpr (has_native_shfl_v<ValueT>)
        {
          other_value = __shfl_xor_sync(FULL_WARP_MASK_, *values, stride);
        }
        else
        {
          other_value = ::cuda::device::warp_shuffle_xor(*values, stride);
        }
      }

      const bool key_precede_other = compare_op(key, other_key);
      const bool other_precede_key = compare_op(other_key, key);
      bool valid;
      if constexpr (FULL)
      {
        valid = true;
      }
      else
      {
        valid = (lane | stride) < valid_items;
      }
      if (valid
          // Using a ternary operator, "reverse ? compare_op(key, other_key) :
          // compare_op(other_key, key)", degrades perf
          && (key_precede_other || other_precede_key) && reverse == key_precede_other)
      {
        key = other_key;
        if constexpr (!KEYS_ONLY_)
        {
          *values = other_value;
        }
      }
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
