// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! The @c cub::detail::block_topk class provides a :ref:`collective <collective-primitives>` method for selecting the
//! top-k elements from a set of items within a CUDA thread block.

#pragma once

#include <cub/config.cuh>

#include <cstdint>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_exchange.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename T, topk::select SelectDirection>
struct twiddle_keys_in_op_t
{
  using sort_key_t = typename Traits<T>::UnsignedBits;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE sort_key_t operator()(T key) const
  {
    auto sort_key = reinterpret_cast<typename Traits<T>::UnsignedBits&>(key);
    sort_key      = Traits<T>::TwiddleIn(sort_key);
    if constexpr (SelectDirection != topk::select::min)
    {
      sort_key = ~sort_key;
    }
    return sort_key;
  }
};

template <typename T, topk::select SelectDirection>
struct twiddle_keys_out_op_t
{
  using sort_key_t = typename Traits<T>::UnsignedBits;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T operator()(sort_key_t sort_key) const
  {
    if constexpr (SelectDirection != topk::select::min)
    {
      sort_key = ~sort_key;
    }
    sort_key = Traits<T>::TwiddleOut(sort_key);
    return reinterpret_cast<T&>(sort_key);
  }
};

template <typename SortKeyT>
struct compare_key_prefix_op
{
  SortKeyT prefix_mask;
  SortKeyT key_prefix;
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator()(SortKeyT sort_key) const
  {
    return (sort_key & prefix_mask) == (key_prefix);
  }
};

// TODO (elstehle): Add documentation
template <typename KeyT, int BlockThreads, int ItemsPerThread, typename ValueT = NullType, int RadixBits = 11>
class block_topk_air
{
private:
  static constexpr int block_threads    = BlockThreads;
  static constexpr int items_per_thread = ItemsPerThread;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr int num_buckets      = (1 << RadixBits);

  // Calculate number of buckets processed per thread
  static constexpr int buckets_per_thread = ::cuda::ceil_div(num_buckets, block_threads);
  static constexpr bool keys_only         = ::cuda::std::is_same_v<ValueT, NullType>;

  using histo_counter_t = ::cuda::std::uint32_t;
  using block_scan_t    = BlockScan<histo_counter_t, block_threads, BLOCK_SCAN_WARP_SCANS>;

  using traits                 = detail::radix::traits_t<KeyT>;
  using bit_ordered_type       = typename traits::bit_ordered_type;
  using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;
  using bit_ordered_inversion  = typename traits::bit_ordered_inversion_policy;

  using fundamental_digit_extractor_t = BFEDigitExtractor<KeyT>;

  struct TempStorage_
  {
    union
    {
      struct
      {
        histo_counter_t histogram[num_buckets];
        typename block_scan_t::TempStorage scan_temp_storage;
        struct
        {
          histo_counter_t k;
          histo_counter_t len;
          int bucket;
          histo_counter_t selected;
        } pass_state;
      } passes;

      struct
      {
        histo_counter_t selected_offset[block_threads];
        histo_counter_t candidate_offset[block_threads];
        KeyT keys[tile_items];
        ValueT values[tile_items];
      } exchange;
    } stage;
  };

  /// Shared storage reference
  TempStorage_& storage;

  /// Linear thread-id
  unsigned int linear_tid;

  // Initialize histogram bins to zero
  _CCCL_DEVICE _CCCL_FORCEINLINE void init_histograms()
  {
    const int base = static_cast<int>(linear_tid) * buckets_per_thread;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        storage.stage.passes.histogram[bin_idx] = 0;
      }
    }
  }

  // Compute histogram over keys. digit_extractor is a function object that returns the bin for each key.
  template <typename DigitExtractorT, typename FilterOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void compute_histograms(
    const bit_ordered_type (&unsigned_keys)[items_per_thread], DigitExtractorT digit_extractor, FilterOpT filter_op)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const bit_ordered_type key = unsigned_keys[i];
      if (filter_op(key))
      {
        const auto digit = digit_extractor.Digit(key);
        atomicAdd(&storage.stage.passes.histogram[digit], histo_counter_t{1});
      }
    }
  }

  // Compute histogram over keys. digit_extractor is a function object that returns the bin for each key.
  template <typename DigitExtractorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  compute_histograms(const bit_ordered_type (&unsigned_keys)[items_per_thread], DigitExtractorT digit_extractor)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const bit_ordered_type key = unsigned_keys[i];
      if (filter_op(key))
      {
        const auto digit = digit_extractor.Digit(key);
        atomicAdd(&storage.stage.passes.histogram[digit], histo_counter_t{1});
      }
    }
  }

  // Compute prefix sum over buckets
  _CCCL_DEVICE _CCCL_FORCEINLINE void compute_bin_offsets()
  {
    histo_counter_t thread_buckets[buckets_per_thread]{};
    const int base = static_cast<int>(linear_tid) * buckets_per_thread;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        thread_buckets[i] = storage.stage.passes.histogram[bin_idx];
      }
    }

    block_scan_t(storage.stage.passes.scan_temp_storage).InclusiveSum(thread_buckets, thread_buckets);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        storage.stage.passes.histogram[bin_idx] = thread_buckets[i];
      }
    }

    __syncthreads();
  }

  // Identify the bucket that the k-th item falls into
  _CCCL_DEVICE _CCCL_FORCEINLINE void choose_bucket(histo_counter_t k)
  {
    const int base = static_cast<int>(linear_tid) * buckets_per_thread;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        const histo_counter_t prev = (bin_idx == 0) ? 0 : storage.stage.passes.histogram[bin_idx - 1];
        const histo_counter_t cur  = storage.stage.passes.histogram[bin_idx];

        if (prev < k && cur >= k)
        {
          storage.stage.passes.pass_state.k        = k - prev;
          storage.stage.passes.pass_state.len      = cur - prev;
          storage.stage.passes.pass_state.bucket   = bin_idx;
          storage.stage.passes.pass_state.selected = prev;
        }
      }
    }
  }

  template <detail::topk::select SelectDirection, bool HasValues, typename ValuesT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void find_splitter_prefix(
    KeyT (&keys)[items_per_thread], ValuesT (&values)[items_per_thread], int k, int begin_bit, int end_bit)
  {}

  template <detail::topk::select SelectDirection, bool HasValues, typename ValuesT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  select_topk(KeyT (&keys)[items_per_thread], ValuesT (&values)[items_per_thread], int k, int begin_bit, int end_bit)
  {
    // TODO (elstehle): Short-circuit if k is constrained to be positive
    if (k <= 0)
    {
      return;
    }

    // TODO (elstehle): Short-circuit if begin_bit is constrained to be non-negative
    if (begin_bit < 0)
    {
      begin_bit = 0;
    }

    // TODO (elstehle): Short-circuit if end_bit is constrained to be less than the maximum number of bits in the key
    // type
    const int max_bit = int(sizeof(KeyT) * 8);
    if (end_bit > max_bit)
    {
      end_bit = max_bit;
    }
    const int total_bits = end_bit - begin_bit;

    // TODO (elstehle): Short-circuit if k is greater than the number of items in the tile
    if (k >= tile_items)
    {
      return;
    }

    // TODO (elstehle): Add support for custom decomposers
    identity_decomposer_t decomposer;

    // Get bit-twiddled sortkeys
    bit_ordered_type unsigned_keys[items_per_thread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      bit_ordered_type raw = reinterpret_cast<bit_ordered_type&>(keys[i]);
      bit_ordered_type val = bit_ordered_conversion::to_bit_ordered(decomposer, raw);
      val                  = BaseDigitExtractor<KeyT>::ProcessFloatMinusZero(val);
      if constexpr (SelectDirection == detail::topk::select::max)
      {
        val = bit_ordered_inversion::inverse(decomposer, val);
      }
      unsigned_keys[i] = val;
    }

    // We only consider candidates identified in the previous pass, i.e., ((sortkey & prefix_mask) == kth_prefix)
    // With each pass, we identify a wider prefix of the splitter key
    bit_ordered_type kth_key_bits = 0;
    bit_ordered_type prefix_mask  = 0;

    // K *within the candidates considered of this pass* of the current pass
    histo_counter_t current_k = static_cast<histo_counter_t>(k);
    // The number of candidates in the current pass
    histo_counter_t current_len = tile_items;
    // The total number of selected items
    histo_counter_t total_selected = 0;

    const int num_passes = (total_bits > 0) ? ::cuda::ceil_div(total_bits, RadixBits) : 0;
    for (int pass = 0; pass < num_passes; ++pass)
    {
      const int pass_end_bit = end_bit - pass * RadixBits;
      int pass_begin_bit     = pass_end_bit - RadixBits;
      if (pass_begin_bit < begin_bit)
      {
        pass_begin_bit = begin_bit;
      }
      const int pass_bits = pass_end_bit - pass_begin_bit;
      if (pass_bits <= 0)
      {
        break;
      }

      // TODO (elstehle): Find a more general way to compute the prefix mask
      const ::cuda::std::uint64_t pass_mask_wide =
        ((::cuda::std::uint64_t{1} << pass_bits) - ::cuda::std::uint64_t{1}) << pass_begin_bit;
      const bit_ordered_type pass_mask = static_cast<bit_ordered_type>(pass_mask_wide);

      // Zero-initialize histograms for the current pass
      init_histograms();
      __syncthreads();

      // Compute histogram over the current pass's bits pre-filtered for keys matching the previous pass's prefix mask
      auto filter_op = compare_key_prefix_op<bit_ordered_type>{prefix_mask, kth_key_bits};
      auto digit_extractor =
        traits::template digit_extractor<fundamental_digit_extractor_t>(pass_begin_bit, pass_bits, decomposer);
      compute_histograms(unsigned_keys, digit_extractor, filter_op);
      __syncthreads();

      // Compute prefix sum over buckets
      compute_bin_offsets();
      __syncthreads();

      // Identify the bucket that the k-th item falls into
      choose_bucket(current_k);
      __syncthreads();

      // Update the current k and length for the next pass
      current_k   = storage.stage.passes.pass_state.k;
      current_len = storage.stage.passes.pass_state.len;
      total_selected += storage.stage.passes.pass_state.selected;

      // Update the kth_key_bits and prefix_mask for the next pass
      // Basically, we will have current_len candidates with the prefix kth_key_bits
      kth_key_bits |= bit_ordered_type(storage.stage.passes.pass_state.bucket) << pass_begin_bit;
      prefix_mask |= pass_mask;

      // Short-circuit if we have identified the exact "splitter" key
      // If all candidates are amongst the remaining top-k, we can simply select all matching smaller or equal to the
      // splitter prefix
      if (current_len == current_k)
      {
        break;
      }

      __syncthreads();
    }
    // Ensure we can repurpose shared memory after the multi-pass stage
    __syncthreads();

    const bit_ordered_type kth_prefix = kth_key_bits & prefix_mask;

    // If all candidates are amongst the remaining top-k, we can simply select all matching smaller or equal to the
    // splitter prefix
    // TODO (elstehle): Make this configurable
    constexpr bool expand_k_to_include_ties = true;
    if (expand_k_to_include_ties && current_len == current_k)
    {
      if (linear_tid == 0)
      {
        storage.stage.exchange.selected_offset[0] = 0;
      }
      // Ensure atomic selection counter has been reset
      __syncthreads();

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        const bit_ordered_type key_prefix = unsigned_keys[i] & prefix_mask;
        const bool qualifies_for_top_k    = key_prefix <= kth_prefix;
        if (qualifies_for_top_k)
        {
          const histo_counter_t selected_offset        = atomicAdd(&storage.stage.exchange.selected_offset[0], 1);
          storage.stage.exchange.keys[selected_offset] = keys[i];
          if constexpr (!keys_only)
          {
            storage.stage.exchange.values[selected_offset] = values[i];
          }
        }
      }
    }
    else
    {
      if (linear_tid == 0)
      {
        storage.stage.exchange.selected_offset[0]  = 0;
        storage.stage.exchange.candidate_offset[0] = total_selected;
      }
      // Ensure atomic selection counter has been reset
      __syncthreads();

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        const bit_ordered_type key_prefix = unsigned_keys[i] & prefix_mask;
        const bool is_selected            = key_prefix < kth_prefix;
        const bool is_candidate           = key_prefix == kth_prefix;
        if (is_selected)
        {
          const histo_counter_t selected_offset        = atomicAdd(&storage.stage.exchange.selected_offset[0], 1);
          storage.stage.exchange.keys[selected_offset] = keys[i];
          if constexpr (!keys_only)
          {
            storage.stage.exchange.values[selected_offset] = values[i];
          }
        }
        if (is_candidate)
        {
          const histo_counter_t candidate_offset        = atomicAdd(&storage.stage.exchange.candidate_offset[0], 1);
          storage.stage.exchange.keys[candidate_offset] = keys[i];
          if constexpr (!keys_only)
          {
            storage.stage.exchange.values[candidate_offset] = values[i];
          }
        }
      }
    }

    // Ensure all threads have finished writing to shared memory
    __syncthreads();

    // Gather selected items into thread registers for return.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const int buffer_idx = static_cast<int>(linear_tid) * items_per_thread + i;
      if (buffer_idx < k)
      {
        keys[i] = storage.stage.exchange.keys[buffer_idx];
        if constexpr (!keys_only)
        {
          values[i] = storage.stage.exchange.values[buffer_idx];
        }
      }
    }
  }

public:
  struct TempStorage : Uninitialized<TempStorage_>
  {};

  _CCCL_DEVICE _CCCL_FORCEINLINE block_topk_air(TempStorage& storage)
      : storage(storage.Alias())
      , linear_tid(RowMajorTid(BlockThreads, 1, 1))
  {}

  template <detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  select_keys(KeyT (&keys)[items_per_thread], int k, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    NullType values[ItemsPerThread];
    select_topk<SelectDirection, false>(keys, values, k, begin_bit, end_bit);
  }

  template <detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void select_pairs(
    KeyT (&keys)[items_per_thread],
    ValueT (&values)[items_per_thread],
    int k,
    int begin_bit = 0,
    int end_bit   = sizeof(KeyT) * 8)
  {
    select_topk<SelectDirection, true>(keys, values, k, begin_bit, end_bit);
  }
};
} // namespace detail
CUB_NAMESPACE_END
