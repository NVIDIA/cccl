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
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__bit/popcount.h>
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
//! @tparam ItemsPerThread
//!   The number of items per thread.
//!
//! @tparam KeyT
//!   Key type
//!
//! @tparam ValueT
//!   <b>[optional]</b> Value type (default: cub::NullType, which indicates a keys-only sort)
//!
template <int ItemsPerThread, typename KeyT, typename ValueT = NullType>
class WarpBitonicSort
{
public:
  //! @brief Sorts keys across a warp of threads using bitonic sorting network.
  //!
  //! @par
  //! - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
  //!   are equivalent: neither one is less than the other. It is not guaranteed
  //!   that the relative order of these two elements will be preserved by sort.
  //! - All threads in the calling warp must invoke this collective.
  //!
  //! @tparam CompareOp Comparison functor type
  //!
  //! @param[in,out] keys Keys to sort, in striped arrangement
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Sort(KeyT (&keys)[ItemsPerThread], CompareOp compare_op) const
  {
    sort<CompareOp, false>(keys, nullptr, compare_op);
  }

  //! @brief Sorts keys across a warp of threads using bitonic sorting network.
  //!
  //! @par
  //! - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
  //!   are equivalent: neither one is less than the other. It is not guaranteed
  //!   that the relative order of these two elements will be preserved by sort.
  //! - All threads in the calling warp must invoke this collective.
  //! - All threads in the calling warp must agree on the same value for `valid_items`.
  //! - The value of `oob_default` is assigned to all keys that are out of
  //!   `valid_items` boundaries. It's expected that `oob_default` is ordered
  //!   after any key in the `valid_items` boundaries.
  //!
  //! @tparam CompareOp Comparison functor type
  //!
  //! @param[in,out] keys Keys to sort, in striped arrangement
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  //! @param[in] valid_items Total number of valid items across the warp
  //! @param[in] oob_default Default value for out-of-bound items
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int valid_items, KeyT oob_default) const
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      if (i * warp_threads + lane >= valid_items)
      {
        keys[i] = oob_default;
      }
    }
    sort<CompareOp, false>(keys, nullptr, compare_op);
  }

  //! @brief Sorts keys across a warp of threads using bitonic sorting network.
  //!
  //! @par
  //! - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
  //!   are equivalent: neither one is less than the other. It is not guaranteed
  //!   that the relative order of these two elements will be preserved by sort.
  //! - All threads in the calling warp must invoke this collective.
  //! - All threads in the calling warp must agree on the same value for `valid_items`.
  //! - `KeyT` should be default constructible. Keys out of `valid_items` boundary may get overwritten.
  //!
  //! @tparam CompareOp Comparison functor type
  //!
  //! @param[in,out] keys Keys to sort, in striped arrangement
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  //! @param[in] valid_items Total number of valid items across the warp
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Sort(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int valid_items) const
  {
    // Padding keys beyond valid_items ensures no uninitialized data is read.
    // Faster than guarding reads with valid_items check.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      if (i * warp_threads + lane >= valid_items)
      {
        keys[i] = KeyT{};
      }
    }
    sort<CompareOp, false>(keys, nullptr, compare_op, valid_items);
  }

  //! @brief Sorts key-value pairs across a warp of threads using bitonic sorting network.
  //!
  //! @par
  //! - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
  //!   are equivalent: neither one is less than the other. It is not guaranteed
  //!   that the relative order of these two elements will be preserved by sort.
  //! - All threads in the calling warp must invoke this collective.
  //!
  //! @tparam CompareOp Comparison functor type
  //!
  //! @param[in,out] keys Keys to sort, in striped arrangement
  //! @param[in,out] values Values to sort (reordered to match key order)
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], CompareOp compare_op) const
  {
    sort<CompareOp, false>(keys, values, compare_op);
  }

  //! @brief Sorts key-value pairs across a warp of threads using bitonic sorting network.
  //!
  //! @par
  //! - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
  //!   are equivalent: neither one is less than the other. It is not guaranteed
  //!   that the relative order of these two elements will be preserved by sort.
  //! - All threads in the calling warp must invoke this collective.
  //! - All threads in the calling warp must agree on the same value for `valid_items`.
  //! - The value of `oob_default` is assigned to all keys that are out of
  //!   `valid_items` boundaries. It's expected that `oob_default` is ordered
  //!   after any key in the `valid_items` boundaries.
  //! - `ValueT` should be default constructible. Values out of `valid_items` boundary may get overwritten.
  //!
  //! @tparam CompareOp Comparison functor type
  //!
  //! @param[in,out] keys Keys to sort, in striped arrangement
  //! @param[in,out] values Values to sort (reordered to match key order)
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  //! @param[in] valid_items Total number of valid items across the warp
  //! @param[in] oob_default Default value for out-of-bound keys
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[ItemsPerThread],
       ValueT (&values)[ItemsPerThread],
       CompareOp compare_op,
       int valid_items,
       KeyT oob_default) const
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      if (i * warp_threads + lane >= valid_items)
      {
        keys[i]   = oob_default;
        values[i] = ValueT{};
      }
    }
    sort<CompareOp, false>(keys, values, compare_op);
  }

  //! @brief Sorts key-value pairs across a warp of threads using bitonic sorting network.
  //!
  //! @par
  //! - Sort is not guaranteed to be stable. That is, suppose that `i` and `j`
  //!   are equivalent: neither one is less than the other. It is not guaranteed
  //!   that the relative order of these two elements will be preserved by sort.
  //! - All threads in the calling warp must invoke this collective.
  //! - All threads in the calling warp must agree on the same value for `valid_items`.
  //! - `KeyT` and `ValueT` should be default constructible. Keys and values out of `valid_items` boundary may get
  //! overwritten.
  //!
  //! @tparam CompareOp Comparison functor type
  //!
  //! @param[in,out] keys Keys to sort, in striped arrangement
  //! @param[in,out] values Values to sort (reordered to match key order)
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  //! @param[in] valid_items Total number of valid items across the warp
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], CompareOp compare_op, int valid_items) const
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      if (i * warp_threads + lane >= valid_items)
      {
        keys[i]   = KeyT{};
        values[i] = ValueT{};
      }
    }
    sort<CompareOp, false>(keys, values, compare_op, valid_items);
  }

private:
  template <int, typename, typename>
  friend class WarpBitonicSort;

  static constexpr int warp_threads = detail::warp_threads;
  static constexpr bool keys_only   = ::cuda::std::is_same_v<ValueT, NullType>;

  int lane = static_cast<int>(::cuda::ptx::get_sreg_laneid());

  template <typename CompareOp, bool Reverse>
  _CCCL_DEVICE _CCCL_FORCEINLINE void sort(KeyT* keys, ValueT* values, CompareOp compare_op) const
  {
    constexpr int first_half_len  = ::cuda::prev_power_of_two(ItemsPerThread - 1);
    constexpr int second_half_len = ItemsPerThread - first_half_len;

    WarpBitonicSort<first_half_len, KeyT, ValueT>{}.template sort<CompareOp, !Reverse>(keys, values, compare_op);
    WarpBitonicSort<second_half_len, KeyT, ValueT>{}.template sort<CompareOp, Reverse>(
      keys + first_half_len, (keys_only ? nullptr : values + first_half_len), compare_op);
    merge<CompareOp, Reverse>(keys, values, compare_op);
  }

  //! @brief Merges a bitonic sequence of key-value pairs across a warp of threads into a monotonic sequence.
  //!
  //! @tparam CompareOp Comparison functor type
  //! @tparam Reverse If true, results are in reverse order
  //!
  //! @param[in,out] keys Keys to merge, in striped arrangement. Input keys must form a bitonic sequence meeting one of
  //! these conditions:
  //!   (1) `ItemsPerThread` is a power of 2
  //!   (2) If `ItemsPerThread` is not a power of 2 and `Reverse` is false: the sequence must be reverse-sorted first,
  //!       then sorted
  //!   (3) If `ItemsPerThread` is not a power of 2 and `Reverse` is true: the sequence must be sorted first, then
  //!       reverse-sorted
  //! @param[in,out] values Values to merge (reordered to match key order)
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  template <typename CompareOp, bool Reverse>
  _CCCL_DEVICE _CCCL_FORCEINLINE void merge(KeyT* keys, ValueT* values, CompareOp compare_op) const
  {
    constexpr int first_half_len  = ::cuda::prev_power_of_two(ItemsPerThread - 1);
    constexpr int second_half_len = ItemsPerThread - first_half_len;
    constexpr int stride          = first_half_len;
    static_assert(first_half_len >= second_half_len);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < second_half_len; ++i)
    {
      const int other_i = i + stride;
      KeyT& key         = keys[i];
      KeyT& other_key   = keys[other_i];
      bool should_swap;
      if constexpr (Reverse)
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

        if constexpr (!keys_only)
        {
          ValueT tmp_v    = values[i];
          values[i]       = values[other_i];
          values[other_i] = tmp_v;
        }
      }
    }

    WarpBitonicSort<first_half_len, KeyT, ValueT>{}.template merge<CompareOp, Reverse>(keys, values, compare_op);
    WarpBitonicSort<second_half_len, KeyT, ValueT>{}.template merge<CompareOp, Reverse>(
      keys + first_half_len, (keys_only ? nullptr : values + first_half_len), compare_op);
  }

  template <typename CompareOp, bool Reverse>
  _CCCL_DEVICE _CCCL_FORCEINLINE void sort(KeyT* keys, ValueT* values, CompareOp compare_op, int valid_items) const
  {
    constexpr int first_half_len  = ::cuda::prev_power_of_two(ItemsPerThread - 1);
    constexpr int second_half_len = ItemsPerThread - first_half_len;

    if (valid_items > first_half_len * warp_threads)
    {
      WarpBitonicSort<first_half_len, KeyT, ValueT>{}.template sort<CompareOp, !Reverse>(keys, values, compare_op);
      WarpBitonicSort<second_half_len, KeyT, ValueT>{}.template sort<CompareOp, Reverse>(
        keys + first_half_len,
        (keys_only ? nullptr : values + first_half_len),
        compare_op,
        valid_items - first_half_len * warp_threads);
      merge<CompareOp, Reverse>(keys, values, compare_op, valid_items);
    }
    else
    {
      WarpBitonicSort<first_half_len, KeyT, ValueT>{}.template sort<CompareOp, Reverse>(
        keys, values, compare_op, valid_items);
    }
  }

  //! @brief Merges a bitonic sequence of key-value pairs across a warp of threads into a monotonic sequence.
  //!
  //! @tparam CompareOp Comparison functor type
  //! @tparam Reverse If true, results are in reverse order
  //!
  //! @param[in,out] keys Keys to merge, in striped arrangement. Input keys must form a bitonic sequence meeting one of
  //! these conditions:
  //!   (1) If `Reverse` is false: the sequence must be reverse-sorted first, then sorted
  //!   (2) If `Reverse` is true: the sequence must be sorted first, then reverse-sorted
  //! @param[in,out] values Values to merge (reordered to match key order)
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second
  //! @param[in] valid_items Total number of valid items across the warp
  template <typename CompareOp, bool Reverse>
  _CCCL_DEVICE _CCCL_FORCEINLINE void merge(KeyT* keys, ValueT* values, CompareOp compare_op, int valid_items) const
  {
    constexpr int first_half_len  = ::cuda::prev_power_of_two(ItemsPerThread - 1);
    constexpr int second_half_len = ItemsPerThread - first_half_len;
    constexpr int stride          = first_half_len;
    static_assert(first_half_len >= second_half_len);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < second_half_len; ++i)
    {
      const int other_i = i + stride;
      KeyT& key         = keys[i];
      KeyT& other_key   = keys[other_i];
      bool should_swap;
      if constexpr (Reverse)
      {
        should_swap = compare_op(key, other_key);
      }
      else
      {
        should_swap = compare_op(other_key, key);
      }
      if (should_swap && other_i * warp_threads + lane < valid_items)
      {
        KeyT tmp_k = key;
        key        = other_key;
        other_key  = tmp_k;

        if constexpr (!keys_only)
        {
          ValueT tmp_v    = values[i];
          values[i]       = values[other_i];
          values[other_i] = tmp_v;
        }
      }
    }

    if (valid_items > first_half_len * warp_threads)
    {
      WarpBitonicSort<first_half_len, KeyT, ValueT>{}.template merge<CompareOp, Reverse>(keys, values, compare_op);
      WarpBitonicSort<second_half_len, KeyT, ValueT>{}.template merge<CompareOp, Reverse>(
        keys + first_half_len,
        (keys_only ? nullptr : values + first_half_len),
        compare_op,
        valid_items - first_half_len * warp_threads);
    }
    else
    {
      WarpBitonicSort<first_half_len, KeyT, ValueT>{}.template merge<CompareOp, Reverse>(
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
  _CCCL_DEVICE _CCCL_FORCEINLINE void Sort(KeyT (&keys)[1], CompareOp compare_op) const
  {
    sort<CompareOp, false>(keys, nullptr, compare_op);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[1], CompareOp compare_op, int valid_items, KeyT oob_default) const
  {
    if (lane >= valid_items)
    {
      keys[0] = oob_default;
    }
    sort<CompareOp, false>(keys, nullptr, compare_op);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Sort(KeyT (&keys)[1], CompareOp compare_op, int valid_items) const
  {
    if (lane >= valid_items)
    {
      keys[0] = KeyT{};
    }
    sort<CompareOp, false>(keys, nullptr, compare_op, valid_items);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Sort(KeyT (&keys)[1], ValueT (&values)[1], CompareOp compare_op) const
  {
    sort<CompareOp, false>(keys, values, compare_op);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[1], ValueT (&values)[1], CompareOp compare_op, int valid_items, KeyT oob_default) const
  {
    if (lane >= valid_items)
    {
      keys[0]   = oob_default;
      values[0] = ValueT{};
    }
    sort<CompareOp, false>(keys, values, compare_op);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[1], ValueT (&values)[1], CompareOp compare_op, int valid_items) const
  {
    if (lane >= valid_items)
    {
      keys[0]   = KeyT{};
      values[0] = ValueT{};
    }
    sort<CompareOp, false>(keys, values, compare_op, valid_items);
  }

private:
  template <int, typename, typename>
  friend class WarpBitonicSort;

  static constexpr int warp_threads            = detail::warp_threads;
  static constexpr unsigned int full_warp_mask = 0xFFFFFFFFu;
  static constexpr bool keys_only              = ::cuda::std::is_same_v<ValueT, NullType>;

  int lane = static_cast<int>(::cuda::ptx::get_sreg_laneid());

  template <typename CompareOp, bool Reverse>
  _CCCL_DEVICE _CCCL_FORCEINLINE void sort(KeyT* keys, ValueT* values, CompareOp compare_op) const
  {
    // Non-recursive implementation for 32 inputs consists of log2(32)=5 stages.
    merge_stage<CompareOp, Reverse, true, 0>(keys, values, compare_op);
    merge_stage<CompareOp, Reverse, true, 1>(keys, values, compare_op);
    merge_stage<CompareOp, Reverse, true, 2>(keys, values, compare_op);
    merge_stage<CompareOp, Reverse, true, 3>(keys, values, compare_op);
    merge_stage<CompareOp, Reverse, true, 4>(keys, values, compare_op);
  }

  template <typename CompareOp, bool Reverse>
  _CCCL_DEVICE _CCCL_FORCEINLINE void merge(KeyT* keys, ValueT* values, CompareOp compare_op) const
  {
    merge_stage<CompareOp, Reverse, true, 4>(keys, values, compare_op);
  }

  template <typename CompareOp, bool Reverse>
  _CCCL_DEVICE _CCCL_FORCEINLINE void sort(KeyT* keys, ValueT* values, CompareOp compare_op, int valid_items) const
  {
    merge_stage<CompareOp, Reverse, false, 0>(keys, values, compare_op, valid_items);
    merge_stage<CompareOp, Reverse, false, 1>(keys, values, compare_op, valid_items);
    merge_stage<CompareOp, Reverse, false, 2>(keys, values, compare_op, valid_items);
    merge_stage<CompareOp, Reverse, false, 3>(keys, values, compare_op, valid_items);
    merge_stage<CompareOp, Reverse, false, 4>(keys, values, compare_op, valid_items);
  }

  template <typename CompareOp, bool Reverse>
  _CCCL_DEVICE _CCCL_FORCEINLINE void merge(KeyT* keys, ValueT* values, CompareOp compare_op, int valid_items) const
  {
    merge_stage<CompareOp, Reverse, false, 4>(keys, values, compare_op, valid_items);
  }

  //! @brief Single stage of the bitonic sorting network operating across warp lanes via shuffles.
  //!
  //! @tparam Full If true, all items are valid (no boundary checks: ``valid_items`` is not used)
  //! @tparam Stage Stage index (0-4 for a 32-thread warp)
  template <typename CompareOp, bool Reverse, bool Full, int Stage>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  merge_stage(KeyT* keys, ValueT* values, CompareOp compare_op, int valid_items = -1) const
  {
    // Each stage divides the inputs into groups and sorts within each group.
    // Sort direction of each group should be adjusted to maintain the bitonic property.
    bool group_reverse = Reverse;
    if constexpr (Stage == 4)
    {
      // The last stage contains only one group, and the sort direction is just Reverse
    }
    else if constexpr (Full)
    {
      // Group size doubles while the number of groups halves with each stage: stage 0 sorts
      // 16 groups of 2 elements, stage 1 sorts 8 groups of 4 elements, and so on.
      //
      // Group ID (starting from 0) is "lane >> (Stage + 1)".
      // Odd groups sort in reverse order.
      // So group_reverse = (lane >> (Stage + 1)) & 1, equal to the following:
      group_reverse ^= static_cast<bool>((lane >> Stage) & 2);
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
      group_reverse ^= static_cast<bool>(Stage & 1);
      const int num_set_bits = ::cuda::std::popcount((static_cast<unsigned>(lane) >> (Stage + 1)));
      group_reverse ^= (num_set_bits == 1 || num_set_bits == 3);
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int stride = (1 << Stage); stride > 0; stride /= 2)
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
        other_key = __shfl_xor_sync(full_warp_mask, key, stride);
      }
      else
      {
        other_key = ::cuda::device::warp_shuffle_xor(key, stride);
      }

      [[maybe_unused]] ValueT other_value;
      if constexpr (!keys_only)
      {
        if constexpr (has_native_shfl_v<ValueT>)
        {
          other_value = __shfl_xor_sync(full_warp_mask, *values, stride);
        }
        else
        {
          other_value = ::cuda::device::warp_shuffle_xor(*values, stride);
        }
      }

      const bool key_precede_other = compare_op(key, other_key);
      const bool other_precede_key = compare_op(other_key, key);
      bool valid;
      if constexpr (Full)
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
        if constexpr (!keys_only)
        {
          *values = other_value;
        }
      }
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
