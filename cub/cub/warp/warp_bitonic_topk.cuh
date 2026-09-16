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

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__bit/popcount.h>
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
} // namespace warp_bitonic_topk

enum class WarpBitonicTopKAlgorithm
{
  //! Eagerly sort and merge every MaxK items. Best suited to small arrays, such as ItemsPerThread <= 4.
  eager,
  //! Buffer candidates that pass the current key threshold before merging them.
  buffered,
};

//! @rst
//! The WarpBitonicTopK class provides methods for selecting top-k items from data partitioned across a logical warp.
//!
//! Overview
//! ++++++++++++++++
//!
//!   WarpBitonicTopK selects the ``k`` items ordered first by a comparison functor with less-than semantics.
//!
//!   Two kinds of TopK functions are provided:
//!   (1) Array overloads operate on items already held by each lane. Input and
//!       output items use a striped arrangement across logical warp lanes.
//!   (2) Iterator overloads read arbitrary-length input through random-access iterators. Output items use a striped
//!       arrangement across logical warp lanes. Supported only by the ``buffered`` algorithm.
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
//!    __global__ void ArrayExampleKernel(...)
//!    {
//!        constexpr int max_k            = 32;
//!        constexpr int items_per_thread = 2;
//!        constexpr int warp_threads = 32;
//!
//!        using WarpBitonicTopKT = cub::detail::WarpBitonicTopK<
//!          max_k, int, warp_threads, cub::NullType, cub::detail::WarpBitonicTopKAlgorithm::eager>;
//!
//!        int thread_keys[items_per_thread];
//!        // ...
//!
//!        WarpBitonicTopKT{temp_storage}.TopK(thread_keys, CustomLess{}, 30);
//!    }
//!
//! Suppose the set of input ``thread_keys`` across a warp of threads is
//! ``{ [0,63], [1,62], [2,61], ..., [31,32] }``.
//! The corresponding output ``thread_keys`` in those threads will be
//! ``{ [0,?], [1,?], [2,?], ..., [29,?], [?,?], [?,?] }``.
//! (``?`` represents an undetermined value.)
//! Note keys are in a :ref:`striped arrangement <flexible-data-arrangement>` across warp lanes.
//!
//!
//! The code snippet below illustrates the iterator overload. The input ``keys_in`` points to ``num_items`` items. The
//! top 30 keys are selected.
//!
//! .. code-block:: c++
//!
//!    __global__ void IteratorExampleKernel(const int* keys_in, int num_items)
//!    {
//!        constexpr int max_k = 32;
//!        constexpr int warp_threads = 32;
//!
//!        using WarpBitonicTopKT = cub::detail::WarpBitonicTopK<max_k, int>;
//!        __shared__ typename WarpBitonicTopKT::TempStorage temp_storage;
//!
//!        int keys_out[max_k / warp_threads];
//!
//!        WarpBitonicTopKT{temp_storage}.TopK(keys_in, CustomLess{}, 30, num_items, keys_out);
//!    }
//!
//! Suppose the input ``keys_in`` is [0, 1, ..., 63]. The output ``keys_out`` in a warp of threads will be
//! ``{ [0,?], [1,?], [2,?], ..., [29,?], [?,?], [?,?] }``.
//! (``?`` represents an undetermined value.)
//! Note keys in ``keys_out`` are in a :ref:`striped arrangement <flexible-data-arrangement>` across warp lanes.
//!
//! @endrst
//!
//! @tparam MaxK
//!   The maximum number of selected items. Must be a positive multiple of the logical warp size.
//!
//! @tparam KeyT
//!   Key type.
//!
//! @tparam LogicalWarpThreads
//!   <b>[optional]</b> Number of threads per logical warp. Must be a power of two no greater than the architectural
//!   warp size.
//!
//! @tparam ValueT
//!   <b>[optional]</b> Value type (default: cub::NullType, which indicates keys-only top-k).
//!
//! @tparam Algorithm
//!   <b>[optional]</b> WarpBitonicTopKAlgorithm to use (default: WarpBitonicTopKAlgorithm::buffered).
template <int MaxK,
          typename KeyT,
          int LogicalWarpThreads             = detail::warp_threads,
          typename ValueT                    = NullType,
          WarpBitonicTopKAlgorithm Algorithm = WarpBitonicTopKAlgorithm::buffered>
class WarpBitonicTopK;

// The eager specialization lacks an iterator overload, even though it could outperform the buffered specialization when
// `num_items` is small. This is because the array overload should be preferred in such cases.
template <int MaxK, typename KeyT, int LogicalWarpThreads, typename ValueT>
class WarpBitonicTopK<MaxK, KeyT, LogicalWarpThreads, ValueT, WarpBitonicTopKAlgorithm::eager>
{
private:
  static_assert(detail::is_valid_logical_warp_size_v<LogicalWarpThreads>,
                "LogicalWarpThreads must not exceed the architectural warp size");
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");
  static_assert(MaxK > 0, "MaxK must be greater than 0");
  static_assert(MaxK % LogicalWarpThreads == 0, "MaxK must be a multiple of LogicalWarpThreads");
  static constexpr int max_k_per_thread = MaxK / LogicalWarpThreads;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<ValueT, NullType>;

  template <int ItemsPerThread>
  using WarpBitonicSortT = WarpBitonicSort<KeyT, ItemsPerThread, LogicalWarpThreads, ValueT>;
  using MaxKSortT        = WarpBitonicSortT<max_k_per_thread>;

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
  //! @param[in] oob_default Default value for out-of-bound key.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int k, int valid_items, KeyT oob_default) const
  {
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
    constexpr int valid_items = ItemsPerThread * LogicalWarpThreads;
    static_assert(valid_items >= MaxK);

    if constexpr (valid_items == MaxK)
    {
      MaxKSortT{}.template sort<CompareOp, false>(keys, values, compare_op);
    }
    else if constexpr (valid_items == ::cuda::next_power_of_two(static_cast<unsigned>(MaxK)))
    {
      // For non-power-of-two input size, WarpBitonicSort conceptually pads the sorting network to the next
      // power-of-two. If input size equals this rounded-up value for MaxK, sort input directly.
      WarpBitonicSortT<ItemsPerThread>{}.template sort<CompareOp, false>(keys, values, compare_op);
    }
    else
    {
      // For non-power-of-two MaxK, WarpBitonicSort::merge requires the retained set to remain reverse-sorted while
      // merging incoming data.
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
  //! @param[in] oob_default Default value for out-of-bound key.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ItemsPerThread],
       ValueT (&values)[ItemsPerThread],
       CompareOp compare_op,
       int k,
       int valid_items,
       KeyT oob_default) const
  {
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
    static_assert(ItemsPerThread * LogicalWarpThreads >= MaxK);

    if (valid_items <= MaxK)
    {
      sort_partial<max_k_per_thread>(keys, values, compare_op, valid_items);
      return;
    }

    MaxKSortT{}.template sort<CompareOp, true>(keys, values, compare_op);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = max_k_per_thread; i <= ItemsPerThread - max_k_per_thread; i += max_k_per_thread)
    {
      const int remain_items = valid_items - i * LogicalWarpThreads;
      if (remain_items >= MaxK)
      {
        MaxKSortT{}.template sort<CompareOp, false>(keys + i, values + i, compare_op);
        warp_bitonic_topk::compare_and_replace<max_k_per_thread>(keys, values, keys + i, values + i, compare_op);
        MaxKSortT{}.template merge<CompareOp, true>(keys, values, compare_op);
      }
      else if (remain_items > 0)
      {
        sort_partial<max_k_per_thread>(keys + i, values + i, compare_op, remain_items);
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
      const int remain_items = valid_items - offset * LogicalWarpThreads;
      if (remain_items > 0)
      {
        sort_partial<remain>(keys + offset, values + offset, compare_op, remain_items);
        warp_bitonic_topk::compare_and_replace<remain, LogicalWarpThreads>(
          keys, values, keys + offset, values + offset, compare_op, remain_items, lane);
        MaxKSortT{}.template merge<CompareOp, true>(keys, values, compare_op);
      }
    }

    warp_bitonic_topk::reverse_items<max_k_per_thread, LogicalWarpThreads>(keys, values, lane, member_mask);
  }

private:
  // WarpBitonicSort's private sort(..., valid_items) requires all items to be initialized, including those beyond
  // valid_items. This wrapper handles that initialization.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  sort_partial(KeyT* keys, ValueT* values, CompareOp compare_op, int valid_items) const
  {
    warp_bitonic_topk::initialize_invalid_items<ItemsPerThread, LogicalWarpThreads>(keys, values, valid_items, lane);
    WarpBitonicSortT<ItemsPerThread>{}.template sort<CompareOp, false>(keys, values, compare_op, valid_items);
  }

  int lane                 = detail::logical_lane_id<LogicalWarpThreads>();
  unsigned int member_mask = WarpMask<LogicalWarpThreads>(detail::logical_warp_id<LogicalWarpThreads>());
};

// The buffered specialization initializes a retained top-k set from per-thread arrays or random-access
// iterators, buffers candidates that pass the current key threshold, and merges them into the retained set.
template <int MaxK, typename KeyT, int LogicalWarpThreads, typename ValueT>
class WarpBitonicTopK<MaxK, KeyT, LogicalWarpThreads, ValueT, WarpBitonicTopKAlgorithm::buffered>
{
private:
  static_assert(detail::is_valid_logical_warp_size_v<LogicalWarpThreads>,
                "LogicalWarpThreads must not exceed the architectural warp size");
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");
  static_assert(MaxK > 0, "MaxK must be greater than 0");
  static_assert(MaxK % LogicalWarpThreads == 0, "MaxK must be a multiple of LogicalWarpThreads");
  static constexpr int max_k_per_thread = MaxK / LogicalWarpThreads;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<ValueT, NullType>;

  template <int ItemsPerThread>
  using WarpBitonicSortT = WarpBitonicSort<KeyT, ItemsPerThread, LogicalWarpThreads, ValueT>;
  using MaxKSortT        = WarpBitonicSortT<max_k_per_thread>;
  using CandidateSortT   = WarpBitonicSortT<1>;

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
  //! @param[in] temp_storage Temporary storage.
  explicit _CCCL_DEVICE_API _CCCL_FORCEINLINE WarpBitonicTopK(TempStorage& temp_storage)
      : storage(&temp_storage.Alias())
  {}

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
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void TopK(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int k)
  {
    static_assert(keys_only);
    ValueT values[ItemsPerThread];
    TopK(keys, values, compare_op, k);
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
  TopK(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, int k, int valid_items)
  {
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
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], CompareOp compare_op, int k)
  {
    constexpr int valid_items = ItemsPerThread * LogicalWarpThreads;
    static_assert(valid_items >= MaxK);

    if constexpr (valid_items == MaxK)
    {
      MaxKSortT{}.template sort<CompareOp, false>(keys, values, compare_op);
    }
    else if constexpr (valid_items == ::cuda::next_power_of_two(static_cast<unsigned>(MaxK)))
    {
      // For non-power-of-two input size, WarpBitonicSort conceptually pads the sorting network to the next
      // power-of-two. If input size equals this rounded-up value for MaxK, sort input directly.
      WarpBitonicSortT<ItemsPerThread>{}.template sort<CompareOp, false>(keys, values, compare_op);
    }
    else
    {
      // Fall back to partial variant TopK because it already calls full variant Sort in most cases.
      // It also means adding a partial-with-oob-default variant would offer no performance benefit.
      //
      // It's possible to eliminate the final reversal when MaxK is a power-of-two as in eager specialization.
      // However, this isn't worthwhile due to overhead like extra adjustment for flushing.
      TopK(keys, values, compare_op, k, valid_items);
    }
  }

  //! @brief Selects top-k key-value pairs from partially valid per-thread arrays across a logical warp.
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
  TopK(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], CompareOp compare_op, int k, int valid_items)
  {
    static_assert(ItemsPerThread * LogicalWarpThreads >= MaxK);

    if (valid_items <= MaxK)
    {
      sort_partial<max_k_per_thread>(keys, values, compare_op, valid_items);
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
      const bool is_candidate = i * LogicalWarpThreads + lane < valid_items;
      process_candidate(
        keys, values, keys[i], values[i], compare_op, is_candidate, k_th_item, k_th_lane, k_th, num_candidates);
    }

    flush_candidates(keys, values, compare_op, num_candidates);
    warp_bitonic_topk::reverse_items<max_k_per_thread, LogicalWarpThreads>(keys, values, lane, member_mask);
  }

  //! @brief Selects top-k keys from iterator input.
  //!
  //! @tparam KeyInputIteratorT Random-access iterator type for input keys.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in] keys_in Iterator pointing to the first input key of a logical warp. All lanes in a logical warp should
  //! pass the same iterator.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Valid range is [1, min(`MaxK`, `num_items`)].
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
  //! @param[in] k Number of pairs to select. Valid range is [1, min(`MaxK`, `num_items`)].
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
      sort_partial<max_k_per_thread>(keys_out, values_out, compare_op, num_items);
      return;
    }

    // Where to find the current k-th key: k_th_lane identifies the lane that owns it, and k_th_item is the array index.
    // Note that k > 0 is required: when k=0, k_th_item becomes max_k_per_thread, causing get_key_threshold to read one
    // element past the array.
    const int k_th_pos  = MaxK - k;
    const int k_th_item = k_th_pos / LogicalWarpThreads;
    const int k_th_lane = k_th_pos % LogicalWarpThreads;

    MaxKSortT{}.template sort<CompareOp, true>(keys_out, values_out, compare_op);
    KeyT k_th          = get_key_threshold(keys_out, k_th_item, k_th_lane);
    int num_candidates = 0;

    const int num_items_per_thread = ::cuda::ceil_div(num_items, LogicalWarpThreads);
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

private:
  // WarpBitonicSort's private sort(..., valid_items) requires all items to be initialized, including those beyond
  // valid_items. This wrapper handles that initialization.
  template <int ItemsPerThread, typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  sort_partial(KeyT* keys, ValueT* values, CompareOp compare_op, int valid_items) const
  {
    warp_bitonic_topk::initialize_invalid_items<ItemsPerThread, LogicalWarpThreads>(keys, values, valid_items, lane);
    WarpBitonicSortT<ItemsPerThread>{}.template sort<CompareOp, false>(keys, values, compare_op, valid_items);
  }

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
    unsigned int mask          = __ballot_sync(member_mask, is_candidate);
    if (mask == 0)
    {
      return;
    }

    mask >>= logical_warp_id * LogicalWarpThreads;
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
      KeyT key = temp_storage.keys[lane];
      ValueT value;
      if constexpr (!keys_only)
      {
        value = temp_storage.values[lane];
      }
      merge_candidates(keys_out, values_out, key, value, compare_op);
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
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  flush_candidates(KeyT* keys_out, ValueT* values_out, CompareOp compare_op, int num_candidates) const
  {
    if (num_candidates)
    {
      __syncwarp(member_mask);
      const _TempStorage& temp_storage = *storage;
      KeyT key                         = (lane < num_candidates) ? temp_storage.keys[lane] : KeyT{};
      ValueT value;
      if constexpr (!keys_only)
      {
        value = (lane < num_candidates) ? temp_storage.values[lane] : ValueT{};
      }
      merge_candidates(keys_out, values_out, key, value, compare_op, num_candidates);
    }
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  merge_candidates(KeyT* keys_out, ValueT* values_out, KeyT& key, ValueT& value, CompareOp compare_op) const
  {
    CandidateSortT{}.template sort<CompareOp, false>(&key, &value, compare_op);

    warp_bitonic_topk::compare_and_replace<1>(keys_out, values_out, &key, &value, compare_op);

    MaxKSortT{}.template merge<CompareOp, true>(keys_out, values_out, compare_op);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void merge_candidates(
    KeyT* keys_out, ValueT* values_out, KeyT& key, ValueT& value, CompareOp compare_op, int valid_items) const
  {
    // flush_candidates initializes key and value beyond valid_items, so it's safe to use CandidateSortT instead of
    // sort_partial here
    CandidateSortT{}.template sort<CompareOp, false>(&key, &value, compare_op, valid_items);

    warp_bitonic_topk::compare_and_replace<1, LogicalWarpThreads>(
      keys_out, values_out, &key, &value, compare_op, valid_items, lane);

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
