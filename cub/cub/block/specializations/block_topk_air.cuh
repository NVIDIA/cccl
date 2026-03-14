// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/block/block_scan.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename SortKeyT>
struct compare_key_prefix_op
{
  static_assert(::cuda::std::is_unsigned_v<SortKeyT>, "SortKeyT must be an unsigned type");

  SortKeyT prefix_mask;
  SortKeyT key_prefix;
  [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr bool operator()(SortKeyT sort_key) const noexcept
  {
    return (sort_key & prefix_mask) == (key_prefix);
  }
};

//! @brief Block-level top-k by radix selection.
//!
//! Selects the smallest (or largest) @p k keys from a tile of keys in registers, without
//! fully sorting. The algorithm has two stages: (1) Radix selection determines the bit-prefix
//! of the k-th key by processing bits MSB to LSB in passes of @p RadixBits. In each pass, a
//! histogram over the current digit is built over candidates only (keys matching the prefix so
//! far), then a prefix sum identifies the bucket containing the k-th item. Items in earlier
//! buckets are guaranteed top-k; items in later buckets are discarded; the chosen bucket
//! becomes the candidate set for the next pass. No data movement occurs during this stage—only
//! the histogram in shared memory is updated. (2) Partitioning scatters the top-k items (key
//! prefix <= k-th prefix) into shared memory via atomic counters, then each thread reads back
//! its portion. Supports key-only and key-value selection.
template <typename KeyT, int BlockThreads, int ItemsPerThread, typename ValueT = NullType, int RadixBits = 8>
class block_topk_air
{
private:
  // TODO (elstehle): Make this configurable
  // Whether to include all items tied with the k-th key when selecting top-k
  static constexpr bool expand_k_to_include_ties = false;

  static constexpr int block_threads    = BlockThreads;
  static constexpr int items_per_thread = ItemsPerThread;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr int num_buckets      = int{1u << RadixBits};

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
          histo_counter_t selected;
          histo_counter_t candidates;
          int bucket;
        } pass_state;
      } passes;

      struct
      {
        histo_counter_t selected_offset[2];
        union
        {
          KeyT keys[tile_items];
          ValueT values[tile_items];
        } exchange;
      } select;
    } stage;
  };

  /// Shared storage reference
  TempStorage_& storage;

  /// Linear thread index
  int linear_tid;

  // Initialize histogram bins to zero
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void init_histograms()
  {
    // Initialize histogram bin counts to zeros
    int histo_offset = 0;

    // Loop unrolling is beneficial for performance here
    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + block_threads <= num_buckets; histo_offset += block_threads)
    {
      storage.stage.passes.histogram[histo_offset + threadIdx.x] = 0;
    }
    // Finish up with guarded initialization if necessary
    if ((num_buckets % block_threads != 0) && (histo_offset + threadIdx.x < num_buckets))
    {
      storage.stage.passes.histogram[histo_offset + threadIdx.x] = 0;
    }
  }

  // Compute histogram over keys
  template <bool IsFullTile, typename DigitExtractorT, typename FilterOpT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void compute_histograms(
    const bit_ordered_type (&unsigned_keys)[items_per_thread],
    int valid_items,
    DigitExtractorT digit_extractor,
    FilterOpT filter_op)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const auto item_index      = linear_tid * items_per_thread + i;
      const bit_ordered_type key = unsigned_keys[i];
      if ((IsFullTile || item_index < valid_items) && filter_op(key))
      {
        const auto digit = digit_extractor.Digit(key);
        atomicAdd(&storage.stage.passes.histogram[digit], histo_counter_t{1});
      }
    }
  }

  // Compute prefix sum over buckets
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void compute_bin_offsets()
  {
    histo_counter_t thread_buckets[buckets_per_thread]{};
    const int base = linear_tid * buckets_per_thread;

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
  }

  // Identify the bucket that the k-th item falls into
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void choose_bucket(histo_counter_t k)
  {
    const int base = linear_tid * buckets_per_thread;

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
          storage.stage.passes.pass_state.bucket     = bin_idx;
          storage.stage.passes.pass_state.candidates = cur - prev;
          storage.stage.passes.pass_state.selected   = prev;
        }
      }
    }
  }

  template <typename detail::topk::select SelectDirection, bool IsFullTile, typename DecomposerT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void get_kth_key_prefix(
    bit_ordered_type (&unsigned_keys)[items_per_thread],
    int k,
    int valid_items,
    int begin_bit,
    int end_bit,
    int& total_selected,
    int& num_candidates,
    bit_ordered_type& kth_key_prefix,
    bit_ordered_type& prefix_mask,
    DecomposerT decomposer = DecomposerT{})
  {
    // Preconditions
    [[maybe_unused]] constexpr int max_bit = int(sizeof(KeyT) * 8);
    _CCCL_ASSERT(k > 0 && k <= tile_items, "k must be in (0, tile_items]");
    if constexpr (!IsFullTile)
    {
      _CCCL_ASSERT(valid_items > 0 && valid_items <= tile_items, "valid_items must be in [1, tile_items]");
    }
    _CCCL_ASSERT(begin_bit >= 0 && begin_bit < max_bit, "begin_bit must be in [0, max_bit)");
    _CCCL_ASSERT(end_bit > begin_bit && end_bit <= max_bit, "end_bit must be in (begin_bit, max_bit]");

    // We only consider candidates identified in the previous pass, i.e., ((sortkey & prefix_mask) == kth_prefix)
    // With each pass, we identify a wider prefix of the splitter key
    kth_key_prefix = 0;
    prefix_mask    = 0;

    // The total number of selected items
    total_selected = 0;

    const int total_bits = (::cuda::std::max) (end_bit - begin_bit, 0);
    const int num_passes = ::cuda::ceil_div(total_bits, RadixBits);
    for (int pass = 0; pass < num_passes; ++pass)
    {
      // Bit-range & mask of the current pass
      const int pass_end_bit           = end_bit - pass * RadixBits;
      const int pass_begin_bit         = (::cuda::std::max) (pass_end_bit - RadixBits, begin_bit);
      const int pass_bits              = pass_end_bit - pass_begin_bit;
      const bit_ordered_type pass_mask = ::cuda::bitmask<bit_ordered_type>(pass_begin_bit, pass_bits);

      // Zero-initialize histograms for the current pass
      init_histograms();
      __syncthreads();

      // Compute histogram over the current pass's, bits pre-filtered for keys matching the previous pass's prefix mask
      auto filter_op = compare_key_prefix_op<bit_ordered_type>{prefix_mask, kth_key_prefix};
      auto digit_extractor =
        traits::template digit_extractor<fundamental_digit_extractor_t>(pass_begin_bit, pass_bits, decomposer);
      compute_histograms<IsFullTile>(unsigned_keys, valid_items, digit_extractor, filter_op);
      __syncthreads();

      // Compute prefix sum over buckets
      compute_bin_offsets();
      __syncthreads();

      // Identify the bucket that the k-th item falls into
      choose_bucket(k);
      __syncthreads();

      // Update the current k and length for the next pass
      k -= storage.stage.passes.pass_state.selected;
      num_candidates = storage.stage.passes.pass_state.candidates;
      total_selected += storage.stage.passes.pass_state.selected;

      // Update the kth_key_prefix and prefix_mask for the next pass
      // Basically, we will have valid_items candidates with the prefix kth_key_prefix
      kth_key_prefix |= bit_ordered_type(storage.stage.passes.pass_state.bucket) << pass_begin_bit;
      prefix_mask |= pass_mask;

      // Short-circuit if all candidates are amongst the top-k
      if (num_candidates == k)
      {
        break;
      }
    }

    // Ensure we can repurpose shared memory after the multi-pass stage
    __syncthreads();
  }

  template <detail::topk::select SelectDirection, bool IsFullTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void select_topk(
    KeyT (&keys)[items_per_thread],
    ValueT (&values)[items_per_thread],
    int k,
    int valid_items,
    int begin_bit,
    int end_bit)
  {
    if constexpr (!IsFullTile)
    {
      _CCCL_ASSERT(valid_items > 0 && valid_items <= tile_items, "valid_items must be in [1, tile_items]");
    }

    // TODO (elstehle): Short-circuit if k is constrained to be positive
    if (k <= 0)
    {
      return;
    }

    // TODO (elstehle): Short-circuit if begin_bit is constrained to be non-negative
    begin_bit = (::cuda::std::max) (begin_bit, 0);

    // TODO (elstehle): Short-circuit if end_bit is constrained to be less than the maximum number of bits in the key
    // type
    const int max_bit = int(sizeof(KeyT) * 8);
    if (end_bit > max_bit)
    {
      end_bit = max_bit;
    }

    // TODO (elstehle): Short-circuit if k is greater than the number of items in the tile
    if ((!IsFullTile && k >= valid_items) || k >= tile_items)
    {
      return;
    }

    // TODO (elstehle): Add support for custom decomposers
    identity_decomposer_t decomposer;

    // Get bit-twiddled sortkeys. For float keys, track which were -0.0 (normalized to +0.0 for ranking) so we can
    // restore -0.0 in the output via a bitvector; no extra key buffer.
    bit_ordered_type(&unsigned_keys)[ItemsPerThread] = reinterpret_cast<bit_ordered_type(&)[ItemsPerThread]>(keys);
    constexpr int flip_back_num_words                = ::cuda::ceil_div(items_per_thread, 32);
    [[maybe_unused]] ::cuda::std::uint32_t flip_back_bits[flip_back_num_words] = {};
    if constexpr (::cuda::is_floating_point_v<KeyT>)
    {
      const bit_ordered_type twiddled_minus_zero =
        Traits<KeyT>::TwiddleIn(bit_ordered_type(1) << (8 * sizeof(bit_ordered_type) - 1));
      const bit_ordered_type twiddled_zero = Traits<KeyT>::TwiddleIn(0);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        unsigned_keys[i] = bit_ordered_conversion::to_bit_ordered(decomposer, unsigned_keys[i]);
        if (unsigned_keys[i] == twiddled_minus_zero)
        {
          flip_back_bits[i / 32] |= (1u << (i % 32));
          unsigned_keys[i] = twiddled_zero;
        }
        if constexpr (SelectDirection == detail::topk::select::max)
        {
          unsigned_keys[i] = bit_ordered_inversion::inverse(decomposer, unsigned_keys[i]);
        }
      }
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        unsigned_keys[i] = bit_ordered_conversion::to_bit_ordered(decomposer, unsigned_keys[i]);
        if constexpr (SelectDirection == detail::topk::select::max)
        {
          unsigned_keys[i] = bit_ordered_inversion::inverse(decomposer, unsigned_keys[i]);
        }
      }
    }

    // The prefix (i.e., the most significant bits) of the k-th key
    bit_ordered_type kth_prefix{};
    // The prefix mask (i.e., the bit mask with the most significant bits populated) of the k-th key
    bit_ordered_type prefix_mask{};
    // The total number of items that compare strictly less than the k-th key's prefix (i.e., the number of items that
    // are guaranteed to be selected)
    int total_selected{};
    // The number of candidates that compare equal to the k-th key's prefix
    auto num_candidates = IsFullTile ? tile_items : valid_items;

    // Identify the prefix of the k-th key
    get_kth_key_prefix<SelectDirection, IsFullTile>(
      unsigned_keys,
      k,
      valid_items,
      begin_bit,
      end_bit,
      total_selected,
      num_candidates,
      kth_prefix,
      prefix_mask,
      decomposer);

    // Scatter indices of selected items into shared memory (only for selecting key-value pairs, using a two-phase
    // approach to lower shared memory requirements).
    [[maybe_unused]] int scatter_indices[items_per_thread];
    if constexpr (!keys_only)
    {
      for (int i = 0; i < items_per_thread; ++i)
      {
        scatter_indices[i] = -1;
      }
    }

    // If all candidates are amongst the remaining top-k, we can simply select all items that compare less than or equal
    // to the splitter prefix. Otherwise, we have to make sure that *all* candidates that compare strictly less than the
    // splitter prefix are selected, and then select amongst candidates that compare equal to the splitter prefix to
    // fill up the remaining slots up to k.
    const bool select_all_candidates = expand_k_to_include_ties || num_candidates + total_selected == k;

    if (linear_tid == 0)
    {
      // Write offsets for selected items with key_prefix < kth_prefix
      storage.stage.select.selected_offset[0] = 0;
      // Write offsets for tied items across the k-th position, i.e., key_prefix == kth_prefix
      storage.stage.select.selected_offset[1] = total_selected;
    }
    // Ensure atomic selection counter has been reset
    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const bit_ordered_type key_prefix = unsigned_keys[i] & prefix_mask;

      const bool is_valid     = (IsFullTile || linear_tid * items_per_thread + i < valid_items);
      const bool is_selected  = key_prefix < kth_prefix;
      const bool is_candidate = key_prefix == kth_prefix;

      // We differentiate between candidates and selected only if not all candidates make it into the top-k items.
      int item_class = (!select_all_candidates) && is_candidate ? 1 : 0;

      // Untwiddle the key before storing in shared memory
      if constexpr (SelectDirection == detail::topk::select::max)
      {
        unsigned_keys[i] = bit_ordered_inversion::inverse(decomposer, unsigned_keys[i]);
      }
      unsigned_keys[i] = bit_ordered_conversion::from_bit_ordered(decomposer, unsigned_keys[i]);

      if (is_valid && (is_selected || is_candidate))
      {
        const histo_counter_t selected_offset = atomicAdd(&storage.stage.select.selected_offset[item_class], 1);
        if constexpr (::cuda::is_floating_point_v<KeyT>)
        {
          storage.stage.select.exchange.keys[selected_offset] =
            (flip_back_bits[i / 32] & (1u << (i % 32))) ? KeyT(-0.0) : ::cuda::std::bit_cast<KeyT>(unsigned_keys[i]);
        }
        else
        {
          storage.stage.select.exchange.keys[selected_offset] = ::cuda::std::bit_cast<KeyT>(unsigned_keys[i]);
        }
        if constexpr (!keys_only)
        {
          scatter_indices[i] = selected_offset;
        }
      }
    }

    // Ensure all threads have finished writing to shared memory
    __syncthreads();

    // Gather selected items into thread registers for return.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const int buffer_idx = linear_tid * items_per_thread + i;
      if (buffer_idx < k)
      {
        keys[i] = storage.stage.select.exchange.keys[buffer_idx];
      }
    }

    if constexpr (!keys_only)
    {
      // Ensure all keys have been loaded from shared memory before we repurpose the exchange buffer for values
      __syncthreads();

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        if (scatter_indices[i] >= 0)
        {
          storage.stage.select.exchange.values[scatter_indices[i]] = values[i];
        }
      }

      // Ensure all values have been written to shared memory before we read them back in
      __syncthreads();

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        const int buffer_idx = linear_tid * items_per_thread + i;
        if (buffer_idx < k)
        {
          values[i] = storage.stage.select.exchange.values[buffer_idx];
        }
      }
    }
  }

public:
  struct TempStorage : Uninitialized<TempStorage_>
  {};

  _CCCL_DEVICE_API _CCCL_FORCEINLINE block_topk_air(TempStorage& storage)
      : storage(storage.Alias())
      , linear_tid(RowMajorTid(BlockThreads, 1, 1))
  {}

  template <detail::topk::select SelectDirection, bool IsFullTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  select_keys(KeyT (&keys)[items_per_thread], int k, int valid_items, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    NullType values[ItemsPerThread];
    select_topk<SelectDirection, IsFullTile>(keys, values, k, valid_items, begin_bit, end_bit);
  }

  template <detail::topk::select SelectDirection, bool IsFullTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void select_pairs(
    KeyT (&keys)[items_per_thread],
    ValueT (&values)[items_per_thread],
    int k,
    int valid_items,
    int begin_bit = 0,
    int end_bit   = sizeof(KeyT) * 8)
  {
    select_topk<SelectDirection, IsFullTile>(keys, values, k, valid_items, begin_bit, end_bit);
  }
};
} // namespace detail
CUB_NAMESPACE_END
