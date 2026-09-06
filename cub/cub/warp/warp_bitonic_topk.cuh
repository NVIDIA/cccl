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
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_bitonic_sort.cuh>
#include <cub/warp/warp_utils.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__type_traits/is_same.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace warp_bitonic_topk
{
template <int Len, typename KeyT, typename ValueT, typename CompareOp>
_CCCL_DEVICE _CCCL_FORCEINLINE void
compare_and_replace(KeyT* keys1, ValueT* values1, const KeyT* keys2, const ValueT* values2, CompareOp compare_op)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < Len; ++i)
  {
    if (compare_op(keys2[i], keys1[i]))
    {
      keys1[i] = keys2[i];
      if constexpr (!::cuda::std::is_same_v<ValueT, NullType>)
      {
        values1[i] = values2[i];
      }
    }
  }
}

template <int Len, int LogicalWarpThreads, typename KeyT, typename ValueT, typename CompareOp>
_CCCL_DEVICE _CCCL_FORCEINLINE void compare_and_replace(
  KeyT* keys1, ValueT* values1, const KeyT* keys2, const ValueT* values2, CompareOp compare_op, int num_items2, int lane)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < Len; ++i)
  {
    if (i * LogicalWarpThreads + lane < num_items2 && compare_op(keys2[i], keys1[i]))
    {
      keys1[i] = keys2[i];
      if constexpr (!::cuda::std::is_same_v<ValueT, NullType>)
      {
        values1[i] = values2[i];
      }
    }
  }
}

template <int Len, int LogicalWarpThreads, typename KeyT, typename ValueT>
_CCCL_DEVICE _CCCL_FORCEINLINE void reverse_items(KeyT* keys, ValueT* values, int lane, unsigned int member_mask)
{
  const int src_lane = LogicalWarpThreads - lane - 1;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < Len / 2; ++i)
  {
    const int other_i = Len - i - 1;

    const KeyT key = keys[i];
    keys[i]        = keys[other_i];
    keys[other_i]  = key;

    if constexpr (!::cuda::std::is_same_v<ValueT, NullType>)
    {
      const ValueT value = values[i];
      values[i]          = values[other_i];
      values[other_i]    = value;
    }
  }

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < Len; ++i)
  {
    keys[i] = ::cuda::device::warp_shuffle_idx<LogicalWarpThreads>(keys[i], src_lane, member_mask);
    if constexpr (!::cuda::std::is_same_v<ValueT, NullType>)
    {
      values[i] = ::cuda::device::warp_shuffle_idx<LogicalWarpThreads>(values[i], src_lane, member_mask);
    }
  }
}
} // namespace warp_bitonic_topk

//! @brief Algorithms used by WarpBitonicTopK.
enum class WarpBitonicTopKAlgorithm
{
  //! Eagerly sort and merge each input tile.
  eager,
  //! Buffer candidates that pass the current key threshold before merging them.
  buffered,
};

template <int MaxK,
          typename KeyT,
          typename ValueT                    = NullType,
          WarpBitonicTopKAlgorithm Algorithm = WarpBitonicTopKAlgorithm::eager,
          int LogicalWarpThreads             = detail::warp_threads>
class WarpBitonicTopK;

//! @rst
//! The WarpBitonicTopK class provides methods for selecting top-k items from data partitioned across a logical warp.
//!
//! Overview
//! ++++++++++++++++
//!
//!   WarpBitonicTopK selects the ``k`` items ordered first by a comparison functor with less-than semantics.
//!
//!   The TopK functions operate on items already held by each lane or read arbitrary-length input through random-access
//!   iterators in array-sized tiles. Output items use a striped arrangement across logical warp lanes.
//!
//! Simple Examples
//! ++++++++++++++++
//!
//! The code snippet below illustrates the array API. The input contains 64 integer keys partitioned across
//! 32 threads, with each thread owning 2 items in a striped arrangement. The top 30 items are selected.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>  // or equivalently <cub/warp/warp_bitonic_topk.cuh>
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
//!    __global__ void ArrayExampleKernel(...)
//!    {
//!        constexpr int max_k            = 32;
//!        constexpr int items_per_thread = 2;
//!
//!        using WarpBitonicTopKT = cub::detail::WarpBitonicTopK<max_k, int>;
//!        __shared__ typename WarpBitonicTopKT::TempStorage temp_storage;
//!
//!        int thread_keys[items_per_thread];
//!        // ...
//!
//!        WarpBitonicTopKT{temp_storage}.TopK(thread_keys, CustomLess{}, 30);
//!    }
//!
//! Suppose the set of input ``thread_keys`` across a logical warp of threads is
//! ``{ [0,63], [1,62], [2,61], ..., [31,32] }``.
//! The corresponding output ``thread_keys`` in those threads will be
//! ``{ [0,?], [1,?], [2,?], ..., [29,?], [?,?], [?,?] }``.
//! Note keys are in a :ref:`striped arrangement <flexible-data-arrangement>` across logical warp lanes.
//!
//! @endrst
//!
//! @tparam MaxK
//!   The maximum number of selected items. Must be a multiple of the logical warp size.
//!
//! @tparam KeyT
//!   Key type.
//!
//! @tparam ValueT
//!   <b>[optional]</b> Value type (default: cub::NullType, which indicates keys-only top-k).
//!
//! @tparam LogicalWarpThreads
//!   <b>[optional]</b> Number of threads per logical warp. Must be a power of two no greater than the architectural
//!   warp size.
template <int MaxK, typename KeyT, typename ValueT, int LogicalWarpThreads>
class WarpBitonicTopK<MaxK, KeyT, ValueT, WarpBitonicTopKAlgorithm::eager, LogicalWarpThreads>
{
private:
  static_assert(detail::is_valid_logical_warp_size_v<LogicalWarpThreads>,
                "LogicalWarpThreads must not exceed the architectural warp size");
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");
  static_assert(MaxK % LogicalWarpThreads == 0, "MaxK must be a multiple of LogicalWarpThreads");
  static constexpr int max_k_per_thread = MaxK / LogicalWarpThreads;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<ValueT, NullType>;

  template <int ItemsPerThread>
  using WarpBitonicSortT = WarpBitonicSort<KeyT, ItemsPerThread, LogicalWarpThreads, ValueT>;

  using MaxKSortT    = WarpBitonicSortT<max_k_per_thread>;
  using _TempStorage = cub::NullType;

public:
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @brief Constructs a WarpBitonicTopK object.
  //!
  //! @param[in] temp_storage Temporary storage.
  explicit _CCCL_DEVICE_API _CCCL_FORCEINLINE WarpBitonicTopK(TempStorage&) {}

  //! @brief Selects top-k keys from per-thread arrays across a logical warp.
  //!
  //! @tparam ItemsPerThread Number of keys per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Must not exceed ``MaxK``.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void TopK(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int k) const
  {
    static_assert(keys_only);
    ValueT values[ItemsPerThread];
    TopK(keys, values, compare_op, k);
  }

  //! @brief Selects top-k keys from partially valid per-thread arrays across a logical warp. An out-of-bound key
  //! ordered after any valid key must be provided.
  //!
  //! @tparam ItemsPerThread Number of keys per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Must not exceed ``MaxK`` or ``num_items``.
  //! @param[in] num_items Total number of valid keys across the logical warp.
  //! @param[in] oob_default Default value for out-of-bound key.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int k, int num_items, KeyT oob_default) const
  {
    static_assert(keys_only);
    ValueT values[ItemsPerThread];
    TopK(keys, values, compare_op, k, num_items, oob_default);
  }

  //! @brief Selects top-k keys from partially valid per-thread arrays across a logical warp.
  //!
  //! @tparam ItemsPerThread Number of keys per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Must not exceed ``MaxK`` or ``num_items``.
  //! @param[in] num_items Total number of valid keys across the logical warp.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int k, int num_items) const
  {
    static_assert(keys_only);
    ValueT values[ItemsPerThread];
    TopK(keys, values, compare_op, k, num_items);
  }

  //! @brief Selects top-k key-value pairs from per-thread arrays across a logical warp.
  //!
  //! @tparam ItemsPerThread Number of key-value pairs per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in,out] values Values selected together with their corresponding keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Must not exceed ``MaxK``.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void TopK(
    KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], CompareOp compare_op, [[maybe_unused]] int k) const
  {
    static_assert(ItemsPerThread * LogicalWarpThreads >= MaxK);

    if constexpr (ItemsPerThread * LogicalWarpThreads == MaxK)
    {
      MaxKSortT{}.template sort<CompareOp, false>(keys, values, compare_op);
    }
    else
    {
      // Due to the requirement of WarpBitonicSort::merge, when MaxK is not a power of 2, the result must be
      // reverse-sorted while merging incoming data, then reversed again at the end.
      constexpr bool reverse = !::cuda::is_power_of_two(MaxK);
      MaxKSortT{}.template sort<CompareOp, reverse>(keys, values, compare_op);

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 1; i < ItemsPerThread / max_k_per_thread; ++i)
      {
        const int offset = max_k_per_thread * i;
        MaxKSortT{}.template sort<CompareOp, !reverse>(keys + offset, values + offset, compare_op);
        warp_bitonic_topk::compare_and_replace<max_k_per_thread>(
          keys, values, keys + offset, values + offset, compare_op);
        MaxKSortT{}.template merge<CompareOp, reverse>(keys, values, compare_op);
      }

      if constexpr (constexpr int remain = ItemsPerThread % max_k_per_thread; remain != 0)
      {
        constexpr int offset = ItemsPerThread / max_k_per_thread * max_k_per_thread;
        WarpBitonicSortT<remain>{}.template sort<CompareOp, !reverse>(keys + offset, values + offset, compare_op);
        if constexpr (reverse)
        {
          warp_bitonic_topk::compare_and_replace<remain>(keys, values, keys + offset, values + offset, compare_op);
        }
        else
        {
          warp_bitonic_topk::compare_and_replace<remain>(
            keys + max_k_per_thread - remain,
            values + max_k_per_thread - remain,
            keys + offset,
            values + offset,
            compare_op);
        }
        MaxKSortT{}.template merge<CompareOp, reverse>(keys, values, compare_op);
      }

      if constexpr (reverse)
      {
        warp_bitonic_topk::reverse_items<max_k_per_thread, LogicalWarpThreads>(keys, values, lane, member_mask);
      }
    }
  }

  //! @brief Selects top-k key-value pairs from partially valid per-thread arrays across a logical warp. An out-of-bound
  //! key ordered after any valid key must be provided.
  //!
  //! @tparam ItemsPerThread Number of key-value pairs per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in,out] values Values selected together with their corresponding keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Must not exceed ``MaxK`` or ``num_items``.
  //! @param[in] num_items Total number of valid pairs across the logical warp.
  //! @param[in] oob_default Default value for out-of-bound key.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ItemsPerThread],
       ValueT (&values)[ItemsPerThread],
       CompareOp compare_op,
       int k,
       int num_items,
       KeyT oob_default) const
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      if (i * LogicalWarpThreads + lane >= num_items)
      {
        keys[i] = oob_default;
      }
    }
    TopK(keys, values, compare_op, k);
  }

  //! @brief Selects top-k key-value pairs from partially valid arrays across a logical warp of threads.
  //!
  //! @tparam ItemsPerThread Number of key-value pairs per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in,out] values Values selected together with their corresponding keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Must not exceed ``MaxK`` or ``num_items``.
  //! @param[in] num_items Total number of valid pairs across the logical warp.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ItemsPerThread],
       ValueT (&values)[ItemsPerThread],
       CompareOp compare_op,
       [[maybe_unused]] int k,
       int num_items) const
  {
    static_assert(ItemsPerThread * LogicalWarpThreads >= MaxK);

    if (num_items < MaxK) // using "<=" is slower
    {
      MaxKSortT{}.template sort<CompareOp, false>(keys, values, compare_op, num_items);
      return;
    }

    MaxKSortT{}.template sort<CompareOp, true>(keys, values, compare_op);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = max_k_per_thread; i <= ItemsPerThread - max_k_per_thread; i += max_k_per_thread)
    {
      const int remain_items = num_items - i * LogicalWarpThreads;
      if (remain_items >= MaxK)
      {
        MaxKSortT{}.template sort<CompareOp, false>(keys + i, values + i, compare_op);
        warp_bitonic_topk::compare_and_replace<max_k_per_thread>(keys, values, keys + i, values + i, compare_op);
        MaxKSortT{}.template merge<CompareOp, true>(keys, values, compare_op);
      }
      else if (remain_items > 0)
      {
        MaxKSortT{}.template sort<CompareOp, false>(keys + i, values + i, compare_op, remain_items);
        warp_bitonic_topk::compare_and_replace<max_k_per_thread, LogicalWarpThreads>(
          keys, values, keys + i, values + i, compare_op, remain_items, lane);
        MaxKSortT{}.template merge<CompareOp, true>(keys, values, compare_op);
        warp_bitonic_topk::reverse_items<max_k_per_thread, LogicalWarpThreads>(keys, values, lane, member_mask);
        return;
      }
    }

    if constexpr (constexpr int remain = ItemsPerThread % max_k_per_thread; remain != 0)
    {
      constexpr int offset   = ItemsPerThread / max_k_per_thread * max_k_per_thread;
      const int remain_items = num_items - offset * LogicalWarpThreads;
      if (remain_items > 0)
      {
        WarpBitonicSortT<remain>{}.template sort<CompareOp, false>(
          keys + offset, values + offset, compare_op, remain_items);
        warp_bitonic_topk::compare_and_replace<remain, LogicalWarpThreads>(
          keys, values, keys + offset, values + offset, compare_op, remain_items, lane);
        MaxKSortT{}.template merge<CompareOp, true>(keys, values, compare_op);
      }
    }

    warp_bitonic_topk::reverse_items<max_k_per_thread, LogicalWarpThreads>(keys, values, lane, member_mask);
  }

  //! @brief Selects top-k key-value pairs from arbitrary-length iterator input using array tiles.
  //!
  //! @tparam KeyInputIteratorT Random-access iterator type for input keys.
  //! @tparam ValueInputIteratorT Random-access iterator type for input values.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in] keys_in Iterator pointing to the first input key of a logical warp. All lanes in a logical warp should
  //! pass the same iterator.
  //! @param[in] values_in Iterator pointing to the first input value of a logical warp. All lanes in a logical warp
  //! should pass the same iterator.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Must not exceed ``MaxK`` or ``num_items``.
  //! @param[in] num_items Number of input pairs.
  //! @param[out] keys_out Selected keys in striped arrangement.
  //! @param[out] values_out Values selected together with their corresponding keys.
  template <typename KeyInputIteratorT, typename ValueInputIteratorT, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyInputIteratorT keys_in,
       ValueInputIteratorT values_in,
       CompareOp compare_op,
       int k,
       int num_items,
       KeyT (&keys_out)[MaxK / LogicalWarpThreads],
       ValueT (&values_out)[MaxK / LogicalWarpThreads]) const
  {
    constexpr int tile_items_per_thread = 2 * max_k_per_thread;
    constexpr int tile_items            = tile_items_per_thread * LogicalWarpThreads;
    KeyT keys[tile_items_per_thread];
    ValueT values[tile_items_per_thread];

    const int first_tile_items = num_items < tile_items ? num_items : tile_items;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < tile_items_per_thread; ++i)
    {
      const int pos = i * LogicalWarpThreads + lane;
      if (pos < first_tile_items)
      {
        keys[i] = keys_in[pos];
        if constexpr (!keys_only)
        {
          values[i] = values_in[pos];
        }
      }
    }
    TopK(keys, values, compare_op, k, first_tile_items);

    for (int offset = first_tile_items; offset < num_items; offset += MaxK)
    {
      const int incoming_items = num_items - offset < MaxK ? num_items - offset : MaxK;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < max_k_per_thread; ++i)
      {
        const int pos = i * LogicalWarpThreads + lane;
        if (pos < incoming_items)
        {
          keys[max_k_per_thread + i] = keys_in[offset + pos];
          if constexpr (!keys_only)
          {
            values[max_k_per_thread + i] = values_in[offset + pos];
          }
        }
      }
      TopK(keys, values, compare_op, k, MaxK + incoming_items);
    }

    const int output_items = num_items < MaxK ? num_items : MaxK;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < max_k_per_thread; ++i)
    {
      if (i * LogicalWarpThreads + lane < output_items)
      {
        keys_out[i] = keys[i];
        if constexpr (!keys_only)
        {
          values_out[i] = values[i];
        }
      }
    }
  }

  //! @brief Selects top-k keys from arbitrary-length iterator input using array tiles.
  //!
  //! @tparam KeyInputIteratorT Random-access iterator type for input keys.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in] keys_in Iterator pointing to the first input key of a logical warp. All lanes in a logical warp should
  //! pass the same iterator.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Must not exceed ``MaxK`` or ``num_items``.
  //! @param[in] num_items Number of input keys.
  //! @param[out] keys_out Selected keys in striped arrangement.
  template <typename KeyInputIteratorT, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyInputIteratorT keys_in,
       CompareOp compare_op,
       int k,
       int num_items,
       KeyT (&keys_out)[MaxK / LogicalWarpThreads]) const
  {
    static_assert(keys_only);
    ValueT values_out[max_k_per_thread];
    TopK(keys_in, nullptr, compare_op, k, num_items, keys_out, values_out);
  }

private:
  int lane                 = detail::logical_lane_id<LogicalWarpThreads>();
  unsigned int member_mask = WarpMask<LogicalWarpThreads>(detail::logical_warp_id<LogicalWarpThreads>());
};

//! @rst
//! The buffered WarpBitonicTopK specialization selects top-k items from arrays or arbitrary-length iterator input
//! across a logical warp.
//!
//! Overview
//! ++++++++++++++++
//!
//!   The buffered WarpBitonicTopK specialization initializes a retained top-k set from per-thread arrays or
//!   random-access iterators, buffers candidates that pass the current key threshold, and merges them into the retained
//!   set. Input and output items use a striped arrangement across logical warp lanes.
//!
//! Simple Example
//! ++++++++++++++
//!
//! The code snippet below selects the top 30 keys from ``num_items`` integer keys.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>  // or equivalently <cub/warp/warp_bitonic_topk.cuh>
//!
//!    __global__ void IteratorExampleKernel(const int* keys_in, int num_items)
//!    {
//!        constexpr int max_k = 32;
//!
//!        using WarpBitonicTopKT = cub::detail::WarpBitonicTopK<
//!          max_k, int, cub::NullType, cub::detail::WarpBitonicTopKAlgorithm::buffered>;
//!        __shared__ typename WarpBitonicTopKT::TempStorage temp_storage;
//!
//!        int keys_out[max_k / 32];
//!
//!        WarpBitonicTopKT{temp_storage}.TopK(keys_in, CustomLess{}, 30, num_items, keys_out);
//!    }
//!
//! Suppose the input ``keys_in`` is [0, 1, ..., 63]. The output ``keys_out`` in a logical warp of threads will be
//! ``{ [0,?], [1,?], [2,?], ..., [29,?], [?,?], [?,?] }``.
//! Note keys are in a :ref:`striped arrangement <flexible-data-arrangement>` across logical warp lanes.
//!
//! @endrst
//!
//! @tparam MaxK
//!   The maximum number of selected items. Must be a multiple of the logical warp size.
//!
//! @tparam KeyT
//!   Key type.
//!
//! @tparam ValueT
//!   <b>[optional]</b> Value type (default: cub::NullType, which indicates keys-only top-k).
//!
//! @tparam LogicalWarpThreads
//!   <b>[optional]</b> Number of threads per logical warp. Must be a power of two no greater than the architectural
//!   warp size.
template <int MaxK, typename KeyT, typename ValueT, int LogicalWarpThreads>
class WarpBitonicTopK<MaxK, KeyT, ValueT, WarpBitonicTopKAlgorithm::buffered, LogicalWarpThreads>
{
private:
  static_assert(detail::is_valid_logical_warp_size_v<LogicalWarpThreads>,
                "LogicalWarpThreads must not exceed the architectural warp size");
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");
  static_assert(MaxK % LogicalWarpThreads == 0, "MaxK must be a multiple of LogicalWarpThreads");
  static constexpr int max_k_per_thread = MaxK / LogicalWarpThreads;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<ValueT, NullType>;

  template <int ItemsPerThread>
  using WarpBitonicSortT = WarpBitonicSort<KeyT, ItemsPerThread, LogicalWarpThreads, ValueT>;

  using MaxKSortT      = WarpBitonicSortT<max_k_per_thread>;
  using CandidateSortT = WarpBitonicSortT<1>;

  struct _TempStorage
  {
    KeyT keys[LogicalWarpThreads];
    ValueT values[LogicalWarpThreads];
  };

public:
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @brief Constructs a WarpBitonicTopK object.
  //!
  //! @param[in] temp_storage Logical-warp-private temporary storage used to buffer candidates.
  explicit _CCCL_DEVICE_API _CCCL_FORCEINLINE WarpBitonicTopK(TempStorage& temp_storage)
      : storage(&temp_storage.Alias())
  {}

  //! @brief Selects top-k keys from per-thread arrays using a candidate buffer.
  //!
  //! @tparam ItemsPerThread Number of keys per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Must not exceed ``MaxK``.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void TopK(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int k)
  {
    static_assert(keys_only);
    ValueT values[ItemsPerThread];
    TopK(keys, values, compare_op, k);
  }

  //! @brief Selects top-k keys from partially valid per-thread arrays using a candidate buffer.
  //!
  //! @tparam ItemsPerThread Number of keys per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Must not exceed ``MaxK`` or ``num_items``.
  //! @param[in] num_items Total number of valid keys across the logical warp.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void TopK(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int k, int num_items)
  {
    static_assert(keys_only);
    ValueT values[ItemsPerThread];
    TopK(keys, values, compare_op, k, num_items);
  }

  //! @brief Selects top-k key-value pairs from per-thread arrays using a candidate buffer.
  //!
  //! @tparam ItemsPerThread Number of key-value pairs per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in,out] values Values selected together with their corresponding keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Must not exceed ``MaxK``.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], CompareOp compare_op, int k)
  {
    TopK(keys, values, compare_op, k, ItemsPerThread * LogicalWarpThreads);
  }

  //! @brief Selects top-k key-value pairs from partially valid per-thread arrays using a candidate buffer.
  //!
  //! @tparam ItemsPerThread Number of key-value pairs per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in,out] values Values selected together with their corresponding keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Must not exceed ``MaxK`` or ``num_items``.
  //! @param[in] num_items Total number of valid pairs across the logical warp.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], CompareOp compare_op, int k, int num_items)
  {
    static_assert(ItemsPerThread * LogicalWarpThreads >= MaxK);

    if (num_items <= MaxK)
    {
      MaxKSortT{}.template sort<CompareOp, false>(keys, values, compare_op, num_items);
      return;
    }

    const int k_th_pos  = MaxK - k;
    const int k_th_item = k_th_pos / LogicalWarpThreads;
    const int k_th_lane = k_th_pos % LogicalWarpThreads;

    MaxKSortT{}.template sort<CompareOp, true>(keys, values, compare_op);
    KeyT k_th          = get_key_threshold(keys, k_th_item, k_th_lane);
    int num_candidates = 0;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = max_k_per_thread; i < ItemsPerThread; ++i)
    {
      const bool is_candidate = i * LogicalWarpThreads + lane < num_items;
      process_candidate(
        keys, values, keys[i], values[i], compare_op, is_candidate, k_th_item, k_th_lane, k_th, num_candidates);
    }
    flush_candidates(keys, values, compare_op, num_candidates);
    warp_bitonic_topk::reverse_items<max_k_per_thread, LogicalWarpThreads>(keys, values, lane, member_mask);
  }

  //! @brief Selects top-k key-value pairs from iterator input.
  //!
  //! @tparam KeyInputIteratorT Random-access iterator type for input keys.
  //! @tparam ValueInputIteratorT Random-access iterator type for input values.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in] keys_in Iterator pointing to the first input key of a logical warp. All lanes in a logical warp should
  //! pass the same iterator.
  //! @param[in] values_in Iterator pointing to the first input value of a logical warp. All lanes in a logical warp
  //! should pass the same iterator.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Must not exceed ``MaxK`` or ``num_items``.
  //! @param[in] num_items Number of input pairs.
  //! @param[out] keys_out Selected keys in striped arrangement.
  //! @param[out] values_out Values selected together with their corresponding keys.
  template <typename KeyInputIteratorT, typename ValueInputIteratorT, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyInputIteratorT keys_in,
       ValueInputIteratorT values_in,
       CompareOp compare_op,
       int k,
       int num_items,
       KeyT (&keys_out)[MaxK / LogicalWarpThreads],
       ValueT (&values_out)[MaxK / LogicalWarpThreads])
  {
    const int k_th_pos  = MaxK - k;
    const int k_th_item = k_th_pos / LogicalWarpThreads;
    const int k_th_lane = k_th_pos % LogicalWarpThreads;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < max_k_per_thread; ++i)
    {
      const int pos = i * LogicalWarpThreads + lane;
      if (pos < num_items)
      {
        keys_out[i] = keys_in[pos];
        if constexpr (!keys_only)
        {
          values_out[i] = values_in[pos];
        }
      }
    }

    if (num_items <= MaxK)
    {
      MaxKSortT{}.template sort<CompareOp, false>(keys_out, values_out, compare_op, num_items);
      return;
    }

    MaxKSortT{}.template sort<CompareOp, true>(keys_out, values_out, compare_op);
    KeyT k_th          = get_key_threshold(keys_out, k_th_item, k_th_lane);
    int num_candidates = 0;

    const int num_items_per_thread = (num_items + LogicalWarpThreads - 1) / LogicalWarpThreads;
    for (int i = max_k_per_thread; i < num_items_per_thread; ++i)
    {
      const int pos = i * LogicalWarpThreads + lane;
      KeyT key;
      ValueT value;
      bool is_candidate = false;
      if (pos < num_items)
      {
        key = keys_in[pos];
        if constexpr (!keys_only)
        {
          value = values_in[pos];
        }
        is_candidate = true;
      }
      process_candidate(
        keys_out, values_out, key, value, compare_op, is_candidate, k_th_item, k_th_lane, k_th, num_candidates);
    }
    flush_candidates(keys_out, values_out, compare_op, num_candidates);
    warp_bitonic_topk::reverse_items<max_k_per_thread, LogicalWarpThreads>(keys_out, values_out, lane, member_mask);
  }

  //! @brief Selects top-k keys from iterator input.
  //!
  //! @tparam KeyInputIteratorT Random-access iterator type for input keys.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in] keys_in Iterator pointing to the first input key of a logical warp. All lanes in a logical warp should
  //! pass the same iterator.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Must not exceed ``MaxK`` or ``num_items``.
  //! @param[in] num_items Number of input keys.
  //! @param[out] keys_out Selected keys in striped arrangement.
  template <typename KeyInputIteratorT, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void TopK(
    KeyInputIteratorT keys_in, CompareOp compare_op, int k, int num_items, KeyT (&keys_out)[MaxK / LogicalWarpThreads])
  {
    static_assert(keys_only);
    ValueT values_out[max_k_per_thread];
    TopK(keys_in, nullptr, compare_op, k, num_items, keys_out, values_out);
  }

private:
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_candidate(
    KeyT* keys_out,
    ValueT* values_out,
    const KeyT& key,
    const ValueT& value,
    CompareOp compare_op,
    bool is_candidate,
    int k_th_item,
    int k_th_lane,
    KeyT& k_th,
    int& num_candidates)
  {
    _TempStorage& temp_storage = *storage;
    is_candidate               = is_candidate && compare_op(key, k_th);
    unsigned int mask;
    if constexpr (LogicalWarpThreads == detail::warp_threads)
    {
      constexpr unsigned int full_warp_mask = 0xFFFFFFFFu;
      mask                                  = __ballot_sync(full_warp_mask, is_candidate);
    }
    else
    {
      mask = __ballot_sync(member_mask, is_candidate);
      mask >>= logical_warp_id * LogicalWarpThreads;
    }
    if (mask == 0)
    {
      return;
    }

    int pos = num_candidates + ::cuda::std::popcount(mask & ((0x1u << lane) - 1));
    if (is_candidate && pos < LogicalWarpThreads)
    {
      temp_storage.keys[pos] = key;
      if constexpr (!keys_only)
      {
        temp_storage.values[pos] = value;
      }
      is_candidate = false;
    }
    num_candidates += ::cuda::std::popcount(mask);
    if (num_candidates >= LogicalWarpThreads)
    {
      __syncwarp(member_mask);
      ValueT value;
      if constexpr (!keys_only)
      {
        value = temp_storage.values[lane];
      }
      merge_candidates(keys_out, values_out, temp_storage.keys[lane], value, compare_op);
      k_th = get_key_threshold(keys_out, k_th_item, k_th_lane);
      num_candidates -= LogicalWarpThreads;
    }
    if (is_candidate)
    {
      pos -= LogicalWarpThreads;
      temp_storage.keys[pos] = key;
      if constexpr (!keys_only)
      {
        temp_storage.values[pos] = value;
      }
    }
    __syncwarp(member_mask);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  flush_candidates(KeyT* keys_out, ValueT* values_out, CompareOp compare_op, int num_candidates) const
  {
    if (num_candidates)
    {
      const _TempStorage& temp_storage = *storage;
      KeyT key                         = (lane < num_candidates) ? temp_storage.keys[lane] : KeyT{};
      ValueT value{};
      if constexpr (!keys_only)
      {
        value = (lane < num_candidates) ? temp_storage.values[lane] : ValueT{};
      }
      merge_candidates(keys_out, values_out, key, value, compare_op, num_candidates);
    }
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  merge_candidates(KeyT* keys_out, ValueT* values_out, KeyT key, ValueT value, CompareOp compare_op) const
  {
    CandidateSortT{}.template sort<CompareOp, false>(&key, &value, compare_op);

    warp_bitonic_topk::compare_and_replace<1>(keys_out, values_out, &key, &value, compare_op);

    MaxKSortT{}.template merge<CompareOp, true>(keys_out, values_out, compare_op);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  merge_candidates(KeyT* keys_out, ValueT* values_out, KeyT key, ValueT value, CompareOp compare_op, int len) const
  {
    CandidateSortT{}.template sort<CompareOp, false>(&key, &value, compare_op, len);

    warp_bitonic_topk::compare_and_replace<1, LogicalWarpThreads>(
      keys_out, values_out, &key, &value, compare_op, len, lane);

    MaxKSortT{}.template merge<CompareOp, true>(keys_out, values_out, compare_op);
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE KeyT
  get_key_threshold(const KeyT* keys_out, int k_th_item, int k_th_lane) const
  {
    return ::cuda::device::warp_shuffle_idx<LogicalWarpThreads>(keys_out[k_th_item], k_th_lane, member_mask);
  }

  _TempStorage* storage;
  int logical_warp_id      = detail::logical_warp_id<LogicalWarpThreads>();
  int lane                 = detail::logical_lane_id<LogicalWarpThreads>();
  unsigned int member_mask = WarpMask<LogicalWarpThreads>(logical_warp_id);
};
} // namespace detail

CUB_NAMESPACE_END
