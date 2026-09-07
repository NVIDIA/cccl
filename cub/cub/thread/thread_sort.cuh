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

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/swap.h>

// TODO: simplify these #includes
#include <cuda_runtime.h>
#include <stdio.h>
#include <limits>
#include <cassert>
#include <type_traits>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <bool Unroll = true, typename KeyT, typename ValueT, typename CompareOp, int ITEMS_PER_THREAD>
_CCCL_DEVICE _CCCL_FORCEINLINE void
stable_odd_even_sort(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&items)[ITEMS_PER_THREAD], CompareOp compare_op)
{
  constexpr bool KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  _CCCL_PRAGMA_UNROLL(Unroll ? ITEMS_PER_THREAD : 1)
  for (int i = 0; i < ITEMS_PER_THREAD; ++i)
  {
    _CCCL_PRAGMA_UNROLL(Unroll ? ITEMS_PER_THREAD : 1) // unroll count is higher than loop count, but that's fine
    for (int j = 1 & i; j < ITEMS_PER_THREAD - 1; j += 2)
    {
      if (compare_op(keys[j + 1], keys[j]))
      {
        using ::cuda::std::swap;
        swap(keys[j], keys[j + 1]);
        if constexpr (!KEYS_ONLY)
        {
          swap(items[j], items[j + 1]);
        }
      }
    } // inner loop
  } // outer loop
}


// TODO: make unrolling optional (there is an iterative way of generating this sorting network, just use that)
/** Unstable odd-even sorting network */
template <int MaxCapacity, typename KeyT, typename ValueT = cub::NullType>
class BatcherOddEvenMergesortNetwork {
  static_assert((MaxCapacity > 0) && ((MaxCapacity & (MaxCapacity - 1)) == 0),
                "sorting network requires a positive power-of-two max capacity");

  private:

  constexpr static bool HasValues =
    !(std::is_same_v<ValueT, cub::NullType> || std::is_void_v<ValueT>);

  template <typename CompareOp>
  __device__ __forceinline__ static void CE(KeyT& left, KeyT& right, ValueT& left_value, ValueT& right_value, CompareOp compare_op) {
    static_assert(HasValues);
    if (compare_op(right, left)) {
      using ::cuda::std::swap;
      swap(left, right);
      swap(left_value, right_value);
    }
  }

  template <typename CompareOp>
  __device__ __forceinline__ static void CE(KeyT& left, KeyT& right, CompareOp compare_op) {
    static_assert(!HasValues);
    const KeyT left_tmp = left;
    const KeyT right_tmp = right;
    bool swap = compare_op(right_tmp, left_tmp);
    left = swap ? right_tmp : left_tmp;
    right = swap ? left_tmp : right_tmp;
  }

  template <int IndexA, int IndexB, typename CompareOp>
  __device__ __forceinline__ static void CompareExchange(KeyT* keys, ValueT* values, CompareOp compare_op) {
    if constexpr (HasValues) {
      CE(keys[IndexA], keys[IndexB], values[IndexA], values[IndexB], compare_op);
    } else {
      CE(keys[IndexA], keys[IndexB], compare_op);
    }
  }

  template <int Begin, int End, int Stride, typename CompareOp>
  __device__ __forceinline__ static void MergeCompareExchanges(KeyT* keys, ValueT* values, CompareOp compare_op) {
    if constexpr (Begin < End) {
      CompareExchange<Begin, Begin+Stride>(keys, values, compare_op);
      MergeCompareExchanges<Begin + 2*Stride, End, Stride>(keys, values, compare_op);
    }
  }
  
  template <int Begin, int End, int Stride, typename CompareOp>
  __device__ __forceinline__ static void Merge(KeyT* keys, ValueT* values, CompareOp compare_op) {
    if constexpr ((Stride << 1) >= (End - Begin)) {
      CompareExchange<Begin, Begin + Stride>(keys, values, compare_op);
    } else {
      Merge<Begin, End, (Stride << 1)>(keys, values, compare_op);
      Merge<Begin + Stride, End, (Stride << 1)>(keys, values, compare_op);
      MergeCompareExchanges<Begin + Stride, End - Stride, Stride>(keys, values, compare_op);
    }
  }
  
  template <int Begin, int End, typename CompareOp>
  __device__ __forceinline__ static void Sort(KeyT* keys, ValueT* values, CompareOp compare_op) {
    if constexpr (End - Begin >= 2) {
      constexpr int Mid = Begin + ((End - Begin) >> 1);
      Sort<Begin, Mid>(keys, values, compare_op);
      Sort<Mid, End>(keys, values, compare_op);
      Merge<Begin, End, 1>(keys, values, compare_op);
    }
  }

  // TODO: consider using a different data loading strategy?
  template <int Capacity>
  __device__ __forceinline__ static void Load(const KeyT* keys, const ValueT* values,
                                              KeyT* temp_keys, ValueT* temp_values,
                                              int items, KeyT max_key) {
    #pragma unroll
    for (int item = 0; item < Capacity; ++item) {
      if (item < items) {
        temp_keys[item] = keys[item];
        if constexpr (HasValues) {
          temp_values[item] = values[item];
        }
      } else {
        temp_keys[item] = max_key;
      }
    }
  }

  // TODO: consider using a different data storing strategy?
  template <int Capacity>
  __device__ __forceinline__ static void Store(KeyT* keys, ValueT* values,
                                               const KeyT* temp_keys, const ValueT* temp_values, int items) {
    #pragma unroll
    for (int item = 0; item < Capacity; ++item) {
      if (item < items) {
        keys[item] = temp_keys[item];
        if constexpr (HasValues) {
          values[item] = temp_values[item];
        }
      }
    }
  }

  template <int Capacity, typename CompareOp>
  __device__ __forceinline__ static void sort_dynamic_warp_capacity(KeyT* keys, ValueT* values, int items, int warp_max_items, CompareOp compare_op, KeyT max_key) {
    if (warp_max_items <= Capacity) {
      KeyT temp_keys[Capacity];
      ValueT temp_values[Capacity];
      Load<Capacity>(keys, values, temp_keys, temp_values, items, max_key);
      Sort<0, Capacity>(temp_keys, temp_values, compare_op);
      Store<Capacity>(keys, values, temp_keys, temp_values, items);
    } else if constexpr (Capacity < MaxCapacity) {
      sort_dynamic_warp_capacity<(Capacity << 1)>(keys, values, items, warp_max_items, compare_op, max_key);
    } else {
      printf("BatcherOddEvenMergesortNetwork::sort number of items exceeds MaxCapacity");
      assert(false);
    }
  }

  public:
  // Use this method if keys and values have at least MaxCapacity many elements
  template <typename CompareOp>
  __device__ __forceinline__ static void sort(KeyT* keys, ValueT* values, CompareOp compare_op) {
    Sort<0, MaxCapacity>(keys, values, compare_op);
  }

  // Use this method if keys has at least MaxCapacity many elements
  template <typename CompareOp>
  __device__ __forceinline__ static void sort(KeyT* keys, CompareOp compare_op) {
    static_assert(!HasValues, "must call `sort` with keys and values");
    Sort<0, MaxCapacity>(keys, nullptr, compare_op);
  }

  // Creates temporary key and value buffers; if items < MaxCapacity,
  // pads key buffer with max_key and leaves parts of value buffer uninitialized
  template <typename CompareOp>
  __device__ __forceinline__ static void sort(KeyT* keys, ValueT* values, int items, CompareOp compare_op, KeyT max_key = std::numeric_limits<KeyT>::max()) {
    static_assert(!HasValues || std::is_trivially_default_constructible_v<ValueT>,
                  "padded values must not require construction");
    const int warp_max_items = __reduce_max_sync(__activemask(), items);
    sort_dynamic_warp_capacity<1>(keys, values, items, warp_max_items, compare_op, max_key);
  }

  // Creates temporary key buffer; if items < Capacity, pads with max_key
  template <typename CompareOp>
  __device__ __forceinline__ static void sort(KeyT* keys, int items, CompareOp compare_op, KeyT max_key = std::numeric_limits<KeyT>::max()) {
    static_assert(!HasValues, "must call `sort` with keys and values");
    sort(keys, nullptr, items, compare_op, max_key);
  }
};

template <typename KeyT, typename ValueT, typename CompareOp, int ITEMS_PER_THREAD>
_CCCL_DEVICE _CCCL_FORCEINLINE void
unstable_odd_even_sort(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], CompareOp compare_op) {
  return detail::BatcherOddEvenMergesortNetwork<ITEMS_PER_THREAD, KeyT, ValueT>::sort(&keys, &values, compare_cop);
}
} // namespace detail

/**
 * @brief Sorts data using odd-even sort method
 *
 * The sorting method is stable. Further details can be found in:
 * A. Nico Habermann. Parallel neighbor sort (or the glory of the induction
 * principle). Technical Report AD-759 248, Carnegie Mellon University, 1972.
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type. If `cub::NullType` is used as `ValueT`, only keys are sorted.
 *
 * @tparam CompareOp
 *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`
 *
 * @tparam ITEMS_PER_THREAD
 *   The number of items per thread
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
 */
template <typename KeyT, typename ValueT, typename CompareOp, int ITEMS_PER_THREAD>
_CCCL_DEVICE _CCCL_FORCEINLINE void
StableOddEvenSort(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&items)[ITEMS_PER_THREAD], CompareOp compare_op)
{
  return detail::stable_odd_even_sort(keys, items, compare_op);
}

CUB_NAMESPACE_END
