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

#include <cub/detail/null_type.cuh>
#include <cub/detail/uninitialized.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_ptx.cuh>
#include <cub/warp/warp_bitonic_sort.cuh>
#include <cub/warp/warp_utils.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/swap.h>

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

// reverse keys and values which are in striped arrangement
template <int Len, int LogicalWarpThreads, typename KeyT, typename ValueT>
_CCCL_DEVICE _CCCL_FORCEINLINE void reverse_items(KeyT* keys, ValueT* values, int lane, unsigned int member_mask)
{
  // first reverse each lane's array
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < Len / 2; ++i)
  {
    const int other_i = Len - i - 1;

    using ::cuda::std::swap;
    swap(keys[i], keys[other_i]);

    if constexpr (!::cuda::std::is_same_v<ValueT, NullType>)
    {
      swap(values[i], values[other_i]);
    }
  }

  // then exchange between between lanes with reversed indices
  const int src_lane = LogicalWarpThreads - lane - 1;

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

template <int ItemsPerThread, int LogicalWarpThreads, typename KeyT, typename ValueT>
_CCCL_DEVICE _CCCL_FORCEINLINE void initialize_invalid_items(KeyT* keys, ValueT* values, int valid_items, int lane)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    if (i * LogicalWarpThreads + lane >= valid_items)
    {
      keys[i] = KeyT{};
      if constexpr (!::cuda::std::is_same_v<ValueT, NullType>)
      {
        values[i] = ValueT{};
      }
    }
  }
}

// WarpBitonicSortNetwork::Sort(..., valid_items) requires all items to be initialized, including those beyond
// valid_items. This wrapper handles that initialization.
template <int ItemsPerThread, int LogicalWarpThreads, typename KeyT, typename ValueT, typename CompareOp>
_CCCL_DEVICE _CCCL_FORCEINLINE void
sort_partial(KeyT* keys, ValueT* values, CompareOp compare_op, int valid_items, int lane)
{
  initialize_invalid_items<ItemsPerThread, LogicalWarpThreads>(keys, values, valid_items, lane);
  WarpBitonicSortNetwork<KeyT, ItemsPerThread, LogicalWarpThreads, ValueT>{}.template Sort<CompareOp, false>(
    keys, values, compare_op, valid_items);
}
} // namespace warp_bitonic_topk

//! @rst
//! The WarpBitonicTopK class provides methods for selecting top-k items from data partitioned across a logical warp.
//!
//! Overview
//! ++++++++++++++++
//!
//!   WarpBitonicTopK selects the ``k`` items ordered first by a comparison functor with less-than semantics.
//!   The selected items are sorted in ascending order according to that comparison functor. The result is
//!   deterministic but not stable: the relative order of items with equivalent keys is not preserved.
//!
//!   TopK can also be performed by a "logical" warp of ``LogicalWarpThreads`` lanes, in which case each logical
//!   warp selects its own items independently of the other logical warps in the same architectural warp.
//!   All lanes of a logical warp must call the same TopK overload. The arguments ``k`` and ``valid_items``,
//!   where applicable, must be uniform across the logical warp.
//!
//!   TopK operates on per-thread arrays. Input and output items use a striped arrangement across logical warp lanes.
//!
//! Simple Examples
//! ++++++++++++++++
//!
//! The code snippet below illustrates the array overload. The input contains 64 integer keys partitioned across
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
//!    __global__ void ExampleKernel(...)
//!    {
//!        constexpr int warp_threads = 32;
//!        constexpr int items_per_thread = 2;
//!        constexpr int threads_per_block = 256;
//!        constexpr int warps_per_block = threads_per_block / warp_threads;
//!        constexpr int max_k = 32;
//!        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
//!
//!        using WarpBitonicTopKT = cub::detail::WarpBitonicTopK<int, max_k>;
//!        __shared__ typename WarpBitonicTopKT::TempStorage temp_storage[warps_per_block];
//!
//!        // Obtain items and put them in thread_keys
//!        int thread_keys[items_per_thread];
//!        // ...
//!
//!        WarpBitonicTopKT{temp_storage[warp_id]}.TopK(thread_keys, CustomLess{}, 30);
//!    }
//!
//! Suppose the set of input ``thread_keys`` across a warp of threads is
//! ``{ [63,31], [62,30], [61,29], ..., [33,1], [32,0] }``.
//! The corresponding output ``thread_keys`` in those threads will be
//! ``{ [0,?], [1,?], [2,?], ..., [29,?], [?,?], [?,?] }``.
//! (``?`` represents an undetermined value.)
//! Note keys are in a :ref:`striped arrangement <flexible-data-arrangement>` across warp lanes.
//!
//!
//! @endrst
//!
//! @tparam KeyT
//!   Key type. Should be default constructible and trivially copyable.
//!
//! @tparam MaxK
//!   The maximum number of selected items. Must be a positive multiple of the logical warp size.
//!
//! @tparam LogicalWarpThreads
//!   <b>[optional]</b> Number of threads per logical warp. Must be a power of two no greater than the architectural
//!   warp size.
//!
//! @tparam ValueT
//!   <b>[optional]</b> Value type (default: cub::NullType, which indicates keys-only top-k).
//!   Should be default constructible and trivially copyable.
template <typename KeyT, int MaxK, int LogicalWarpThreads = detail::warp_threads, typename ValueT = NullType>
class WarpBitonicTopK
{
  static_assert(detail::is_valid_logical_warp_size_v<LogicalWarpThreads>,
                "LogicalWarpThreads must not exceed the architectural warp size");
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");
  static_assert(MaxK > 0, "MaxK must be greater than 0");
  static_assert(MaxK % LogicalWarpThreads == 0, "MaxK must be a multiple of LogicalWarpThreads");
  static constexpr int max_k_per_thread = MaxK / LogicalWarpThreads;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<ValueT, NullType>;

  template <int ItemsPerThread>
  using SortNetworkT = WarpBitonicSortNetwork<KeyT, ItemsPerThread, LogicalWarpThreads, ValueT>;
  using MaxKSortT    = SortNetworkT<max_k_per_thread>;

  using _TempStorage = cub::NullType;

public:
  using TempStorage = Uninitialized<_TempStorage>;

  //! @brief Constructs a WarpBitonicTopK object.
  //!
  //! @param[in] temp_storage Temporary storage.
  explicit _CCCL_DEVICE_API _CCCL_FORCEINLINE WarpBitonicTopK(TempStorage&) {}

  //! @brief Selects top-k keys from per-thread arrays across a logical warp.
  //!
  //! @tparam ItemsPerThread Number of keys per thread. Must satisfy: ItemsPerThread * LogicalWarpThreads >= MaxK.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Valid range is [1, `MaxK`].
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void TopK(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int k) const
  {
    _CCCL_ASSERT(k >= 1 && k <= MaxK, "k must be in [1, MaxK]");
    static_assert(keys_only);
    ValueT values[ItemsPerThread];
    TopK(keys, values, compare_op, k);
  }

  //! @brief Selects top-k keys from partially valid per-thread arrays across a logical warp. An out-of-bound key
  //! ordered after any valid key must be provided.
  //!
  //! @tparam ItemsPerThread Number of keys per thread. Must satisfy: ItemsPerThread * LogicalWarpThreads >= MaxK.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Valid range is [1, min(`MaxK`, `valid_items`)].
  //! @param[in] valid_items Total number of valid keys across the logical warp.
  //! @param[in] oob_default Default value for out-of-bound key. It's assigned to all keys that are out of `valid_items`
  //! boundary. Must be ordered strictly after any key in the `valid_items` boundary according to ``compare_op``.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int k, int valid_items, KeyT oob_default) const
  {
    _CCCL_ASSERT(k >= 1 && k <= MaxK && k <= valid_items, "k must be in [1, min(MaxK, valid_items)]");
    static_assert(keys_only);
    ValueT values[ItemsPerThread];
    TopK(keys, values, compare_op, k, valid_items, oob_default);
  }

  //! @brief Selects top-k keys from partially valid per-thread arrays across a logical warp.
  //!
  //! @tparam ItemsPerThread Number of keys per thread. Must satisfy: ItemsPerThread * LogicalWarpThreads >= MaxK.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Valid range is [1, min(`MaxK`, `valid_items`)].
  //! @param[in] valid_items Total number of valid keys across the logical warp.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int k, int valid_items) const
  {
    _CCCL_ASSERT(k >= 1 && k <= MaxK && k <= valid_items, "k must be in [1, min(MaxK, valid_items)]");
    static_assert(keys_only);
    ValueT values[ItemsPerThread];
    TopK(keys, values, compare_op, k, valid_items);
  }

  //! @brief Selects top-k key-value pairs from per-thread arrays across a logical warp.
  //!
  //! @tparam ItemsPerThread Number of key-value pairs per thread. Must satisfy: ItemsPerThread * LogicalWarpThreads >=
  //! MaxK.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in,out] values Values selected together with their corresponding keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Valid range is [1, `MaxK`].
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void TopK(
    KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], CompareOp compare_op, [[maybe_unused]] int k) const
  {
    _CCCL_ASSERT(k >= 1 && k <= MaxK, "k must be in [1, MaxK]");
    constexpr int valid_items = ItemsPerThread * LogicalWarpThreads;
    static_assert(valid_items >= MaxK);

    if constexpr (valid_items == MaxK)
    {
      MaxKSortT{}.template Sort<CompareOp, false>(keys, values, compare_op);
    }
    else if constexpr (valid_items == ::cuda::next_power_of_two(static_cast<unsigned>(MaxK)))
    {
      // For non-power-of-two input size, WarpBitonicSortNetwork conceptually pads the sorting network to the next
      // power-of-two. If input size equals this rounded-up value for MaxK, sort input directly.
      SortNetworkT<ItemsPerThread>{}.template Sort<CompareOp, false>(keys, values, compare_op);
    }
    else
    {
      // For non-power-of-two MaxK, WarpBitonicSortNetwork::Merge requires the retained set to remain reverse-sorted
      // while merging incoming data.
      constexpr bool reverse = !::cuda::is_power_of_two(MaxK);
      MaxKSortT{}.template Sort<CompareOp, reverse>(keys, values, compare_op);

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 1; i < ItemsPerThread / max_k_per_thread; ++i)
      {
        const int offset = max_k_per_thread * i;
        MaxKSortT{}.template Sort<CompareOp, !reverse>(keys + offset, values + offset, compare_op);
        warp_bitonic_topk::compare_and_replace<max_k_per_thread>(
          keys, values, keys + offset, values + offset, compare_op);
        MaxKSortT{}.template Merge<CompareOp, reverse>(keys, values, compare_op);
      }

      if constexpr (constexpr int remain = ItemsPerThread % max_k_per_thread; remain != 0)
      {
        constexpr int offset = ItemsPerThread / max_k_per_thread * max_k_per_thread;
        SortNetworkT<remain>{}.template Sort<CompareOp, !reverse>(keys + offset, values + offset, compare_op);
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
        MaxKSortT{}.template Merge<CompareOp, reverse>(keys, values, compare_op);
      }

      // restore the requested order after processing all data
      if constexpr (reverse)
      {
        warp_bitonic_topk::reverse_items<max_k_per_thread, LogicalWarpThreads>(keys, values, lane, member_mask);
      }
    }
  }

  //! @brief Selects top-k key-value pairs from partially valid per-thread arrays across a logical warp. An out-of-bound
  //! key ordered after any valid key must be provided.
  //!
  //! @tparam ItemsPerThread Number of key-value pairs per thread. Must satisfy: ItemsPerThread * LogicalWarpThreads >=
  //! MaxK.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in,out] values Values selected together with their corresponding keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Valid range is [1, min(`MaxK`, `valid_items`)].
  //! @param[in] valid_items Total number of valid pairs across the logical warp.
  //! @param[in] oob_default Default value for out-of-bound key. It's assigned to all keys that are out of `valid_items`
  //! boundary. Must be ordered strictly after any key in the `valid_items` boundary according to ``compare_op``.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ItemsPerThread],
       ValueT (&values)[ItemsPerThread],
       CompareOp compare_op,
       int k,
       int valid_items,
       KeyT oob_default) const
  {
    _CCCL_ASSERT(k >= 1 && k <= MaxK && k <= valid_items, "k must be in [1, min(MaxK, valid_items)]");
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      if (i * LogicalWarpThreads + lane >= valid_items)
      {
        keys[i]   = oob_default;
        values[i] = ValueT{};
      }
    }
    TopK(keys, values, compare_op, k);
  }

  //! @brief Selects top-k key-value pairs from partially valid per-thread arrays across a logical warp of threads.
  //!
  //! @tparam ItemsPerThread Number of key-value pairs per thread. Must satisfy: ItemsPerThread * LogicalWarpThreads >=
  //! MaxK.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in,out] values Values selected together with their corresponding keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Valid range is [1, min(`MaxK`, `valid_items`)].
  //! @param[in] valid_items Total number of valid pairs across the logical warp.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ItemsPerThread],
       ValueT (&values)[ItemsPerThread],
       CompareOp compare_op,
       [[maybe_unused]] int k,
       int valid_items) const
  {
    _CCCL_ASSERT(k >= 1 && k <= MaxK && k <= valid_items, "k must be in [1, min(MaxK, valid_items)]");
    static_assert(ItemsPerThread * LogicalWarpThreads >= MaxK);

    if (valid_items <= MaxK)
    {
      warp_bitonic_topk::sort_partial<max_k_per_thread, LogicalWarpThreads>(keys, values, compare_op, valid_items, lane);
      return;
    }

    MaxKSortT{}.template Sort<CompareOp, true>(keys, values, compare_op);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = max_k_per_thread; i <= ItemsPerThread - max_k_per_thread; i += max_k_per_thread)
    {
      const int remain_items = valid_items - i * LogicalWarpThreads;
      if (remain_items >= MaxK)
      {
        MaxKSortT{}.template Sort<CompareOp, false>(keys + i, values + i, compare_op);
        warp_bitonic_topk::compare_and_replace<max_k_per_thread>(keys, values, keys + i, values + i, compare_op);
        MaxKSortT{}.template Merge<CompareOp, true>(keys, values, compare_op);
      }
      else if (remain_items > 0)
      {
        warp_bitonic_topk::sort_partial<max_k_per_thread, LogicalWarpThreads>(
          keys + i, values + i, compare_op, remain_items, lane);
        warp_bitonic_topk::compare_and_replace<max_k_per_thread, LogicalWarpThreads>(
          keys, values, keys + i, values + i, compare_op, remain_items, lane);
        MaxKSortT{}.template Merge<CompareOp, true>(keys, values, compare_op);
        // Don't reverse and then return here. A common return point here may cause the compiler to use runtime
        // indexing and put the array in local memory.
      }
    }

    if constexpr (constexpr int remain = ItemsPerThread % max_k_per_thread; remain != 0)
    {
      constexpr int offset   = ItemsPerThread / max_k_per_thread * max_k_per_thread;
      const int remain_items = valid_items - offset * LogicalWarpThreads;
      if (remain_items > 0)
      {
        warp_bitonic_topk::sort_partial<remain, LogicalWarpThreads>(
          keys + offset, values + offset, compare_op, remain_items, lane);
        warp_bitonic_topk::compare_and_replace<remain, LogicalWarpThreads>(
          keys, values, keys + offset, values + offset, compare_op, remain_items, lane);
        MaxKSortT{}.template Merge<CompareOp, true>(keys, values, compare_op);
      }
    }

    warp_bitonic_topk::reverse_items<max_k_per_thread, LogicalWarpThreads>(keys, values, lane, member_mask);
  }

private:
  int lane                 = detail::logical_lane_id<LogicalWarpThreads>();
  unsigned int member_mask = WarpMask<LogicalWarpThreads>(detail::logical_warp_id<LogicalWarpThreads>());
};
} // namespace detail

CUB_NAMESPACE_END
