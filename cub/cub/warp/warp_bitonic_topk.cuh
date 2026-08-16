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
#include <cub/warp/warp_bitonic_sort.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__type_traits/is_same.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! @rst
//! The WarpBitonicTopK class provides methods for selecting top-k items from data partitioned across a CUDA warp.
//!
//! Overview
//! ++++++++++++++++
//!
//!   WarpBitonicTopK selects the ``k`` items ordered first by a comparison functor with less-than semantics.
//!
//!   Two kinds of TopK functions are provided:
//!   Array overloads operate on items already held by each lane. Input and output items use a striped arrangement
//!   across warp lanes. Iterator overloads read arbitrary-length input from memory and require ``TempStorage`` to
//!   buffer candidates while merging them into the retained top-k set. Output items also use a striped arrangement
//!   across warp lanes.
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
//!
//!        using WarpBitonicTopKT = cub::detail::WarpBitonicTopK<max_k, int>;
//!
//!        int thread_keys[items_per_thread];
//!        // ...
//!
//!        WarpBitonicTopKT{}.TopK(thread_keys, CustomLess{}, 30);
//!    }
//!
//! Suppose the set of input ``thread_keys`` across a warp of threads is
//! ``{ [0,63], [1,62], [2,61], ..., [31,32] }``.
//! The corresponding output ``thread_keys`` in those threads will be
//! ``{ [0,?], [1,?], [2,?], ..., [29,?], [?,?], [?,?] }``.
//! Note keys are in a :ref:`striped arrangement <flexible-data-arrangement>` across warp lanes.
//!
//! The code snippet below illustrates the iterator overload. The input ``keys_in`` points to ``num_items`` items.
//! The top 30 items are selected.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>  // or equivalently <cub/warp/warp_bitonic_topk.cuh>
//!
//!    __global__ void IteratorExampleKernel(const int* keys_in, int num_items)
//!    {
//!        constexpr int max_k = 32;
//!
//!        using WarpBitonicTopKT = cub::detail::WarpBitonicTopK<max_k, int>;
//!        __shared__ typename WarpBitonicTopKT::TempStorage temp_storage;
//!
//!        int keys_out[max_k / 32];
//!
//!        WarpBitonicTopKT{temp_storage}.TopK(keys_in, CustomLess{}, 30, num_items, keys_out);
//!    }
//!
//! Suppose the input ``keys_in`` is [0, 1, ..., 63],
//! The output ``keys_out`` in a warp of threads will be
//! ``{ [0,?], [1,?], [2,?], ..., [29,?], [?,?], [?,?] }``.
//! Note keys are in a :ref:`striped arrangement <flexible-data-arrangement>` across warp lanes.
//!
//! @endrst
//!
//! @tparam MAX_K
//!   The maximum number of selected items. Must be a multiple of the warp size.
//!
//! @tparam KeyT
//!   Key type.
//!
//! @tparam ValueT
//!   <b>[optional]</b> Value type (default: cub::NullType, which indicates keys-only top-k).
template <int MAX_K, typename KeyT, typename ValueT = NullType>
class WarpBitonicTopK
{
private:
  static constexpr unsigned int FULL_WARP_MASK_ = 0xFFFFFFFFu;
  static constexpr int WARP_THREADS_            = detail::warp_threads;
  static_assert(MAX_K % WARP_THREADS_ == 0);
  static constexpr int max_k_per_thread_ = MAX_K / WARP_THREADS_;
  static constexpr bool KEYS_ONLY_       = ::cuda::std::is_same_v<ValueT, NullType>;

  struct TempStorage_
  {
    KeyT keys[WARP_THREADS_];
    ValueT values[WARP_THREADS_];
  };

public:
  struct TempStorage : Uninitialized<TempStorage_>
  {};

  //! @brief Constructs a WarpBitonicTopK object without temporary storage.
  //!
  //! This constructor is intended for array overloads. Iterator overloads require temporary storage and must
  //! use the constructor that accepts ``TempStorage``.
  _CCCL_DEVICE _CCCL_FORCEINLINE WarpBitonicTopK()
      : storage_(nullptr)
  {}

  //! @brief Constructs a WarpBitonicTopK object with temporary storage for iterator overloads.
  //!
  //! @param[in] temp_storage Warp-private temporary storage used to buffer candidates while processing iterator input.
  explicit _CCCL_DEVICE _CCCL_FORCEINLINE WarpBitonicTopK(TempStorage& temp_storage)
      : storage_(&temp_storage.Alias())
  {}

  //! @brief Selects top-k keys from per-thread arrays across a warp.
  //!
  //! @tparam ITEMS_PER_THREAD Number of keys per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Must not exceed ``MAX_K``.
  template <int ITEMS_PER_THREAD, typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void TopK(KeyT (&keys)[ITEMS_PER_THREAD], CompareOp compare_op, int k)
  {
    static_assert(KEYS_ONLY_);
    ValueT values[ITEMS_PER_THREAD];
    TopK(keys, values, compare_op, k);
  }

  //! @brief Selects top-k keys from partially valid per-thread arrays across a warp. An out-of-bound key ordered after
  //! any valid key must be provided.
  //!
  //! @tparam ITEMS_PER_THREAD Number of keys per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Must not exceed ``MAX_K`` or ``num_items``.
  //! @param[in] num_items Total number of valid keys across the warp.
  //! @param[in] oob_default Default value for out-of-bound key.
  template <int ITEMS_PER_THREAD, typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ITEMS_PER_THREAD], CompareOp compare_op, int k, int num_items, KeyT oob_default)
  {
    static_assert(KEYS_ONLY_);
    ValueT values[ITEMS_PER_THREAD];
    TopK(keys, values, compare_op, k, num_items, oob_default);
  }

  //! @brief Selects top-k keys from partially valid per-thread arrays across a warp.
  //!
  //! @tparam ITEMS_PER_THREAD Number of keys per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Must not exceed ``MAX_K`` or ``num_items``.
  //! @param[in] num_items Total number of valid keys across the warp.
  template <int ITEMS_PER_THREAD, typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void TopK(KeyT (&keys)[ITEMS_PER_THREAD], CompareOp compare_op, int k, int num_items)
  {
    static_assert(KEYS_ONLY_);
    ValueT values[ITEMS_PER_THREAD];
    TopK(keys, values, compare_op, k, num_items);
  }

  //! @brief Selects top-k key-value pairs from per-thread arrays across a warp.
  //!
  //! @tparam ITEMS_PER_THREAD Number of key-value pairs per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in,out] values Values selected together with their corresponding keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Must not exceed ``MAX_K``.
  template <int ITEMS_PER_THREAD, typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], CompareOp compare_op, [[maybe_unused]] int k)
  {
    static_assert(ITEMS_PER_THREAD * WARP_THREADS_ >= MAX_K);

    if constexpr (ITEMS_PER_THREAD * WARP_THREADS_ == MAX_K)
    {
      WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Sort_<CompareOp, false>(keys, values, compare_op);
    }
    else
    {
      // Due to a limitation of WarpBitonicSort::Merge_, when MAX_K is not a power of 2, the result must be
      // reverse-sorted while merging incoming data, then reversed again at the end.
      constexpr bool reverse = !::cuda::is_power_of_two(MAX_K);
      WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Sort_<CompareOp, reverse>(keys, values, compare_op);

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 1; i < ITEMS_PER_THREAD / max_k_per_thread_; ++i)
      {
        int offset = max_k_per_thread_ * i;
        WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Sort_<CompareOp, !reverse>(
          keys + offset, values + offset, compare_op);
        compare_and_replace_<max_k_per_thread_>(keys, values, keys + offset, values + offset, compare_op);
        WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Merge_<CompareOp, reverse>(keys, values, compare_op);
      }

      if constexpr (constexpr int remain = ITEMS_PER_THREAD % max_k_per_thread_; remain != 0)
      {
        constexpr int offset = ITEMS_PER_THREAD / max_k_per_thread_ * max_k_per_thread_;
        WarpBitonicSort<remain, KeyT, ValueT>::template Sort_<CompareOp, !reverse>(
          keys + offset, values + offset, compare_op);
        if constexpr (reverse)
        {
          compare_and_replace_<remain>(keys, values, keys + offset, values + offset, compare_op);
        }
        else
        {
          compare_and_replace_<remain>(
            keys + max_k_per_thread_ - remain,
            values + max_k_per_thread_ - remain,
            keys + offset,
            values + offset,
            compare_op);
        }
        WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Merge_<CompareOp, reverse>(keys, values, compare_op);
      }

      if constexpr (reverse)
      {
        reverse_<max_k_per_thread_>(keys, values);
      }
    }
  }

  //! @brief Selects top-k key-value pairs from partially valid per-thread arrays across a warp. An out-of-bound key
  //! ordered after any valid key must be provided.
  //!
  //! @tparam ITEMS_PER_THREAD Number of key-value pairs per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in,out] values Values selected together with their corresponding keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Must not exceed ``MAX_K`` or ``num_items``.
  //! @param[in] num_items Total number of valid pairs across the warp.
  //! @param[in] oob_default Default value for out-of-bound key.
  template <int ITEMS_PER_THREAD, typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ITEMS_PER_THREAD],
       ValueT (&values)[ITEMS_PER_THREAD],
       CompareOp compare_op,
       int k,
       int num_items,
       KeyT oob_default)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      if (i * WARP_THREADS_ + lane_ >= num_items)
      {
        keys[i] = oob_default;
      }
    }
    TopK(keys, values, compare_op, k);
  }

  //! @brief Selects top-k key-value pairs from partially valid arrays across a warp of threads.
  //!
  //! @tparam ITEMS_PER_THREAD Number of key-value pairs per thread.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in,out] keys Keys in striped arrangement. On return, the first ``k`` striped positions contain the selected
  //! top-k keys.
  //! @param[in,out] values Values selected together with their corresponding keys.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Must not exceed ``MAX_K`` or ``num_items``.
  //! @param[in] num_items Total number of valid pairs across the warp.
  template <int ITEMS_PER_THREAD, typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  TopK(KeyT (&keys)[ITEMS_PER_THREAD],
       ValueT (&values)[ITEMS_PER_THREAD],
       CompareOp compare_op,
       [[maybe_unused]] int k,
       int num_items)
  {
    static_assert(ITEMS_PER_THREAD * WARP_THREADS_ >= MAX_K);

    if (num_items < MAX_K) // using "<=" is slower
    {
      WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Sort_<CompareOp, false>(
        keys, values, compare_op, num_items);
      return;
    }

    WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Sort_<CompareOp, true>(keys, values, compare_op);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = max_k_per_thread_; i <= ITEMS_PER_THREAD - max_k_per_thread_; i += max_k_per_thread_)
    {
      const int remain_items = num_items - i * WARP_THREADS_;
      if (remain_items >= MAX_K)
      {
        WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Sort_<CompareOp, false>(
          keys + i, values + i, compare_op);
        compare_and_replace_<max_k_per_thread_>(keys, values, keys + i, values + i, compare_op);
        WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Merge_<CompareOp, true>(keys, values, compare_op);
      }
      else if (remain_items > 0)
      {
        WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Sort_<CompareOp, false>(
          keys + i, values + i, compare_op, remain_items);
        compare_and_replace_<max_k_per_thread_>(keys, values, keys + i, values + i, compare_op, remain_items);
        WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Merge_<CompareOp, true>(keys, values, compare_op);
        reverse_<max_k_per_thread_>(keys, values);
        return;
      }
    }

    if constexpr (constexpr int remain = ITEMS_PER_THREAD % max_k_per_thread_; remain != 0)
    {
      constexpr int offset   = ITEMS_PER_THREAD / max_k_per_thread_ * max_k_per_thread_;
      const int remain_items = num_items - offset * WARP_THREADS_;
      if (remain_items > 0)
      {
        WarpBitonicSort<remain, KeyT, ValueT>::template Sort_<CompareOp, false>(
          keys + offset, values + offset, compare_op, remain_items);
        compare_and_replace_<remain>(keys, values, keys + offset, values + offset, compare_op, remain_items);
        WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Merge_<CompareOp, true>(keys, values, compare_op);
      }
    }

    reverse_<max_k_per_thread_>(keys, values);
  }

  //! @brief Selects top-k key-value pairs from iterator input.
  //!
  //! This overload can process arbitrary-length input and requires the object to be constructed with ``TempStorage``.
  //!
  //! @tparam KeyInputIteratorT Random-access iterator type for input keys.
  //! @tparam ValueInputIteratorT Random-access iterator type for input values.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in] keys_in Iterator pointing to the first input key.
  //! @param[in] values_in Iterator pointing to the first input value.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of pairs to select. Must not exceed ``MAX_K`` or ``num_items``.
  //! @param[in] num_items Number of input pairs.
  //! @param[out] keys_out Selected keys in striped arrangement.
  //! @param[out] values_out Values selected together with their corresponding keys.
  template <typename KeyInputIteratorT, typename ValueInputIteratorT, typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  TopK(KeyInputIteratorT keys_in,
       ValueInputIteratorT values_in,
       CompareOp compare_op,
       int k,
       int num_items,
       KeyT (&keys_out)[MAX_K / detail::warp_threads],
       ValueT (&values_out)[MAX_K / detail::warp_threads])
  {
    const int k_th_pos  = MAX_K - k;
    const int k_th_item = k_th_pos / WARP_THREADS_;
    const int k_th_lane = k_th_pos % WARP_THREADS_;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < max_k_per_thread_; ++i)
    {
      int pos = i * WARP_THREADS_ + lane_;
      if (pos < num_items)
      {
        keys_out[i] = keys_in[pos];
        if constexpr (!KEYS_ONLY_)
        {
          values_out[i] = values_in[pos];
        }
      }
    }

    if (num_items <= MAX_K)
    {
      WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Sort_<CompareOp, false>(
        keys_out, values_out, compare_op, num_items);
      return;
    }

    WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Sort_<CompareOp, true>(keys_out, values_out, compare_op);
    KeyT k_th          = get_key_threshold_(keys_out, k_th_item, k_th_lane);
    int num_candidates = 0;

    const int num_items_per_thread = (num_items + WARP_THREADS_ - 1) / WARP_THREADS_;
    for (int i = max_k_per_thread_; i < num_items_per_thread; ++i)
    {
      int pos = i * WARP_THREADS_ + lane_;
      KeyT key;
      ValueT value;
      bool is_candidate = false;
      if (pos < num_items)
      {
        key = keys_in[pos];
        if constexpr (!KEYS_ONLY_)
        {
          value = values_in[pos];
        }
        is_candidate = true;
      }
      process_candidate_(
        keys_out, values_out, key, value, compare_op, is_candidate, k_th_item, k_th_lane, k_th, num_candidates);
    }
    flush_candidates_(keys_out, values_out, compare_op, num_candidates);
    reverse_<max_k_per_thread_>(keys_out, values_out);
  }

  //! @brief Selects top-k keys from iterator input.
  //!
  //! This overload can process arbitrary-length input and requires the object to be constructed with ``TempStorage``.
  //!
  //! @tparam KeyInputIteratorT Random-access iterator type for input keys.
  //! @tparam CompareOp Comparison functor type.
  //!
  //! @param[in] keys_in Iterator pointing to the first input key.
  //! @param[in] compare_op Comparison functor which returns true if the first argument is ordered before the second.
  //! @param[in] k Number of keys to select. Must not exceed ``MAX_K`` or ``num_items``.
  //! @param[in] num_items Number of input keys.
  //! @param[out] keys_out Selected keys in striped arrangement.
  template <typename KeyInputIteratorT, typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  TopK(KeyInputIteratorT keys_in,
       CompareOp compare_op,
       int k,
       int num_items,
       KeyT (&keys_out)[MAX_K / detail::warp_threads])
  {
    static_assert(KEYS_ONLY_);
    ValueT values_out[max_k_per_thread_];
    TopK(keys_in, nullptr, compare_op, k, num_items, keys_out, values_out);
  }

private:
  template <int LEN, typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void compare_and_replace_(
    KeyT* __restrict__ keys1,
    ValueT* __restrict__ values1,
    const KeyT* __restrict__ keys2,
    const ValueT* __restrict__ values2,
    CompareOp compare_op)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < LEN; ++i)
    {
      if (compare_op(keys2[i], keys1[i]))
      {
        keys1[i] = keys2[i];
        if constexpr (!KEYS_ONLY_)
        {
          values1[i] = values2[i];
        }
      }
    }
  }

  template <int LEN, typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void compare_and_replace_(
    KeyT* __restrict__ keys1,
    ValueT* __restrict__ values1,
    const KeyT* __restrict__ keys2,
    const ValueT* __restrict__ values2,
    CompareOp compare_op,
    int num_items2)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < LEN; ++i)
    {
      if (i * WARP_THREADS_ + lane_ < num_items2 && compare_op(keys2[i], keys1[i]))
      {
        keys1[i] = keys2[i];
        if constexpr (!KEYS_ONLY_)
        {
          values1[i] = values2[i];
        }
      }
    }
  }

  template <int LEN>
  _CCCL_DEVICE _CCCL_FORCEINLINE void reverse_(KeyT* __restrict__ keys, ValueT* __restrict__ values)
  {
    const int src_lane = WARP_THREADS_ - lane_ - 1;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < LEN / 2; ++i)
    {
      const int other_i = LEN - i - 1;

      KeyT key      = keys[i];
      keys[i]       = keys[other_i];
      keys[other_i] = key;

      if constexpr (!KEYS_ONLY_)
      {
        ValueT value    = values[i];
        values[i]       = values[other_i];
        values[other_i] = value;
      }
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < LEN; ++i)
    {
      keys[i] = shuffle_idx_(keys[i], src_lane);
      if constexpr (!KEYS_ONLY_)
      {
        values[i] = shuffle_idx_(values[i], src_lane);
      }
    }
  }

  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE static T shuffle_idx_(const T& value, int src_lane)
  {
    if constexpr (has_native_shfl_v<T>)
    {
      return __shfl_sync(FULL_WARP_MASK_, value, src_lane);
    }
    else
    {
      return ::cuda::device::warp_shuffle_idx(value, src_lane);
    }
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_candidate_(
    KeyT (&keys_out)[MAX_K / detail::warp_threads],
    ValueT (&values_out)[MAX_K / detail::warp_threads],
    const KeyT& key,
    const ValueT& value,
    CompareOp compare_op,
    bool is_candidate,
    int k_th_item,
    int k_th_lane,
    KeyT& k_th,
    int& num_candidates)
  {
    TempStorage_& storage = *storage_;
    is_candidate          = is_candidate && compare_op(key, k_th);
    uint32_t mask         = __ballot_sync(FULL_WARP_MASK_, is_candidate);
    if (mask == 0)
    {
      return;
    }

    int pos = num_candidates + __popc(mask & ((0x1u << lane_) - 1));
    if (is_candidate && pos < WARP_THREADS_)
    {
      storage.keys[pos] = key;
      if constexpr (!KEYS_ONLY_)
      {
        storage.values[pos] = value;
      }
      is_candidate = false;
    }
    num_candidates += __popc(mask);
    if (num_candidates >= WARP_THREADS_)
    {
      __syncwarp();
      ValueT value;
      if constexpr (!KEYS_ONLY_)
      {
        value = storage.values[lane_];
      }
      merge_candidates_(keys_out, values_out, storage.keys[lane_], value, compare_op);
      k_th = get_key_threshold_(keys_out, k_th_item, k_th_lane);
      num_candidates -= WARP_THREADS_;
    }
    if (is_candidate)
    {
      pos -= WARP_THREADS_;
      storage.keys[pos] = key;
      if constexpr (!KEYS_ONLY_)
      {
        storage.values[pos] = value;
      }
    }
    __syncwarp();
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void flush_candidates_(
    KeyT (&keys_out)[MAX_K / detail::warp_threads],
    ValueT (&values_out)[MAX_K / detail::warp_threads],
    CompareOp compare_op,
    int num_candidates)
  {
    if (num_candidates)
    {
      TempStorage_& storage = *storage_;
      KeyT key              = (lane_ < num_candidates) ? storage.keys[lane_] : KeyT{};
      ValueT value{};
      if constexpr (!KEYS_ONLY_)
      {
        value = (lane_ < num_candidates) ? storage.values[lane_] : ValueT{};
      }
      merge_candidates_(keys_out, values_out, key, value, compare_op, num_candidates);
    }
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void merge_candidates_(
    KeyT (&keys_out)[MAX_K / detail::warp_threads],
    ValueT (&values_out)[MAX_K / detail::warp_threads],
    KeyT key,
    ValueT value,
    CompareOp compare_op)
  {
    WarpBitonicSort<1, KeyT, ValueT>::template Sort_<CompareOp, false>(&key, &value, compare_op);

    compare_and_replace_<1>(keys_out, values_out, &key, &value, compare_op);

    WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Merge_<CompareOp, true>(keys_out, values_out, compare_op);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void merge_candidates_(
    KeyT (&keys_out)[MAX_K / detail::warp_threads],
    ValueT (&values_out)[MAX_K / detail::warp_threads],
    KeyT key,
    ValueT value,
    CompareOp compare_op,
    int len)
  {
    WarpBitonicSort<1, KeyT, ValueT>::template Sort_<CompareOp, false>(&key, &value, compare_op, len);

    compare_and_replace_<1>(keys_out, values_out, &key, &value, compare_op, len);

    WarpBitonicSort<max_k_per_thread_, KeyT, ValueT>::template Merge_<CompareOp, true>(keys_out, values_out, compare_op);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE KeyT
  get_key_threshold_(KeyT (&keys_out)[MAX_K / detail::warp_threads], int k_th_item, int k_th_lane)
  {
    return shuffle_idx_(keys_out[k_th_item], k_th_lane);
  }

  TempStorage_* storage_;
  int lane_ = threadIdx.x % WARP_THREADS_;
};
} // namespace detail

CUB_NAMESPACE_END
