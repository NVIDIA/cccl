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
#include <cuda/std/__type_traits/conditional.h>
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
  [[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE constexpr bool operator()(SortKeyT sort_key) const noexcept
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
//! becomes the candidate set for the next pass. No data movement occurs during this stage. Only
//! the histogram in shared memory is updated. (2) Partitioning scatters the top-k items (key
//! prefix <= k-th prefix) into shared memory via atomic counters, then each thread reads back
//! its portion. Supports key-only and key-value selection.
//!
//! Keys and values are consumed and returned in a blocked arrangement, i.e., thread `t` holds the tile's items
//! `[t * ItemsPerThread, (t + 1) * ItemsPerThread)`. On a partial tile, the valid items are the tile's first
//! `valid_items` items. The selected items are returned in the tile's first `min(k, valid_items)` slots. The
//! contents of the remaining slots are unspecified.
//!
//! TODO (elstehle): Support a striped arrangement through `select_*_striped_to_striped` overloads. The blocked
//! arrangement is assumed by the partial-tile validity checks in `compute_histograms` and `select_topk`, by
//! `select_topk`'s early return for `k >= valid_items`, and by the slots that `select_topk` gathers the selected
//! items from.
//!
//! @tparam UnrollBitPasses
//!   <b>[optional]</b> When true (default), the radix-pass loop may be fully unrolled. Unrolling provides better
//!   throughput and latency but may come at increased register usage.
//! @tparam MemoizeKeys
//!   <b>[optional]</b> When true (default), a register copy of the original keys is kept and the partitioning stage
//!   scatters from that copy, so selected keys need no untwiddling and no -0.0 restoration state. When false, keys
//!   are untwiddled in place and -0.0 is restored through a bitvector. The copy extends the keys' live ranges across
//!   the radix passes, so disabling it can reduce register pressure.
template <typename KeyT,
          int ThreadsPerBlock,
          int ItemsPerThread,
          typename ValueT      = NullType,
          int RadixBits        = 8,
          bool UnrollBitPasses = true,
          bool MemoizeKeys     = true>
class block_topk_air
{
private:
  // TODO (elstehle): Make this configurable
  // Whether to include all items tied with the k-th key when selecting top-k
  static constexpr bool expand_k_to_include_ties = false;

  static constexpr int threads_per_block = ThreadsPerBlock;
  static constexpr int items_per_thread  = ItemsPerThread;
  static constexpr int tile_items        = threads_per_block * items_per_thread;
  static constexpr int num_buckets       = int{1u << RadixBits};

  // Calculate number of buckets processed per thread
  static constexpr int buckets_per_thread = ::cuda::ceil_div(num_buckets, threads_per_block);
  static constexpr bool keys_only         = ::cuda::std::is_same_v<ValueT, NullType>;

  using histo_counter_t = ::cuda::std::uint32_t;
  using block_scan_t    = BlockScan<histo_counter_t, threads_per_block, BLOCK_SCAN_WARP_SCANS>;

  using traits                 = detail::radix::traits_t<KeyT>;
  using bit_ordered_type       = typename traits::bit_ordered_type;
  using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;

  // ShiftDigitExtractor rather than BFEDigitExtractor: the BFE path is inline PTX, which is opaque to the
  // compiler and prevents it from fusing with other instructions.
  using fundamental_digit_extractor_t = ShiftDigitExtractor<KeyT>;

  struct TempStorage_
  {
    union
    {
      struct
      {
        // Double-buffered: pass p histograms into buffer p%2 while re-zeroing buffer (p+1)%2
        histo_counter_t histogram[2][num_buckets];
        typename block_scan_t::TempStorage scan_temp_storage;
      } passes;

      struct
      {
        union
        {
          KeyT keys[tile_items];
          ValueT values[tile_items];
        } u;
      } exchange;
    } stage;

    struct
    {
      histo_counter_t selected;
      histo_counter_t candidates;
      int bucket;
    } pass_state;

    // Outside the aliased union: preset before the radix passes (ordered by their barriers),
    // so the partitioning stage needs no setup phase or barrier of its own.
    histo_counter_t selected_offset[2];
  };

  /// Shared storage reference
  TempStorage_& storage;

  /// Linear thread index
  int linear_tid;

  // Zero one histogram buffer
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void zero_histogram(histo_counter_t (&histogram)[num_buckets])
  {
    int histo_offset = 0;

    // Loop unrolling is beneficial for performance here
    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + threads_per_block <= num_buckets; histo_offset += threads_per_block)
    {
      histogram[histo_offset + linear_tid] = 0;
    }
    // Finish up with guarded initialization if necessary
    if ((num_buckets % threads_per_block != 0) && (histo_offset + linear_tid < num_buckets))
    {
      histogram[histo_offset + linear_tid] = 0;
    }
  }

  // Compute histogram over keys
  template <detail::topk::select SelectDirection, bool IsFullTile, typename DigitExtractorT, typename FilterOpT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void compute_histograms(
    const bit_ordered_type (&unsigned_keys)[items_per_thread],
    int valid_items,
    DigitExtractorT digit_extractor,
    FilterOpT filter_op,
    histo_counter_t (&histogram)[num_buckets])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const auto item_index      = linear_tid * items_per_thread + i;
      const bit_ordered_type key = unsigned_keys[i];
      if ((IsFullTile || item_index < valid_items) && filter_op(key))
      {
        const auto digit  = static_cast<int>(digit_extractor.Digit(key));
        const auto bucket = (SelectDirection == detail::topk::select::min) ? digit : (num_buckets - 1 - digit);
        atomicAdd_block(&histogram[bucket], histo_counter_t{1});
      }
    }
  }

  // Fused prefix sum over buckets + identification of the bucket that the k-th item falls into.
  // The crossing test runs on the scan's register results (inclusive = exclusive + count).
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  scan_and_choose_bucket(const histo_counter_t (&histogram)[num_buckets], histo_counter_t k)
  {
    histo_counter_t counts[buckets_per_thread]{};
    const int base = linear_tid * buckets_per_thread;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        counts[i] = histogram[bin_idx];
      }
    }

    histo_counter_t exclusive_sums[buckets_per_thread];
    block_scan_t(storage.stage.passes.scan_temp_storage).ExclusiveSum(counts, exclusive_sums);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        const histo_counter_t exclusive = exclusive_sums[i];
        const histo_counter_t inclusive = exclusive + counts[i];
        // If a bug causes less than k candidates in the histogram, the previous pass' pass_state will persist making
        // debugging harder. This assert should catch such bugs. Should there ever be a valid use case for less than k
        // candidates, the pass_state needs to be reset unconditionally.
        _CCCL_ASSERT((bin_idx != num_buckets - 1) || (inclusive >= k),
                     "Less than k candidates have participated in the histogram");

        if (exclusive < k && inclusive >= k)
        {
          storage.pass_state.bucket     = bin_idx;
          storage.pass_state.candidates = counts[i];
          storage.pass_state.selected   = exclusive;
        }
      }
    }
  }

  // One radix pass: histogram over the surviving candidates (re-zeroing the other buffer in the
  // same phase), fused scan+choose, and the pass-state update. Returns true when all remaining
  // candidates are amongst the top-k (early exit).
  template <detail::topk::select SelectDirection, bool IsFullTile, typename DecomposerT>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool run_radix_pass(
    const bit_ordered_type (&unsigned_keys)[items_per_thread],
    int valid_items,
    int& k,
    int& total_selected,
    int& num_candidates,
    bit_ordered_type& kth_key_prefix,
    bit_ordered_type& prefix_mask,
    int pass,
    int pass_begin_bit,
    int pass_bits,
    bool zero_next_histogram,
    DecomposerT decomposer)
  {
    const bit_ordered_type pass_mask = ::cuda::bitmask<bit_ordered_type>(pass_begin_bit, pass_bits);

    histo_counter_t(&histogram)[num_buckets] = storage.stage.passes.histogram[pass % 2];

    // Compute histogram over the current pass's bits, pre-filtered for keys matching the previous pass's prefix mask
    const auto filter_op = compare_key_prefix_op<bit_ordered_type>{prefix_mask, kth_key_prefix};
    const auto digit_extractor =
      traits::template digit_extractor<fundamental_digit_extractor_t>(pass_begin_bit, pass_bits, decomposer);
    compute_histograms<SelectDirection, IsFullTile>(unsigned_keys, valid_items, digit_extractor, filter_op, histogram);
    if (zero_next_histogram)
    {
      // Zero the other buffer for the next pass. It is untouched during the first pass and its
      // last read otherwise preceded the previous pass's state barrier, so this shares the
      // histogram phase instead of needing one of its own
      zero_histogram(storage.stage.passes.histogram[(pass + 1) % 2]);
    }
    __syncthreads();

    // Compute prefix sum over buckets and identify the bucket that the k-th item falls into
    scan_and_choose_bucket(histogram, static_cast<histo_counter_t>(k));
    __syncthreads();

    // Update the current k and length for the next pass
    k -= storage.pass_state.selected;
    num_candidates = storage.pass_state.candidates;
    total_selected += storage.pass_state.selected;

    // Update the kth_key_prefix and prefix_mask for the next pass
    // Basically, we will have valid_items candidates with the prefix kth_key_prefix
    const auto kth_key_digit =
      (SelectDirection == detail::topk::select::min)
        ? storage.pass_state.bucket
        : (num_buckets - 1 - storage.pass_state.bucket);
    kth_key_prefix |= bit_ordered_type(kth_key_digit) << pass_begin_bit;
    prefix_mask |= pass_mask;

    // Short-circuit if all candidates are amongst the top-k
    return num_candidates == k;
  }

  template <typename detail::topk::select SelectDirection, bool IsFullTile, typename DecomposerT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void get_kth_key_prefix(
    bit_ordered_type (&unsigned_keys)[items_per_thread],
    int k,
    int valid_items,
    int& total_selected,
    int& num_candidates,
    bit_ordered_type& kth_key_prefix,
    bit_ordered_type& prefix_mask,
    DecomposerT decomposer = DecomposerT{})
  {
    // Preconditions
    constexpr int max_bit = int(sizeof(KeyT) * 8);
    _CCCL_ASSERT(k > 0 && k <= tile_items, "k must be in (0, tile_items]");
    if constexpr (!IsFullTile)
    {
      _CCCL_ASSERT(valid_items > 0 && valid_items <= tile_items, "valid_items must be in [1, tile_items]");
    }

    // We only consider candidates identified in the previous pass, i.e., ((sortkey & prefix_mask) == kth_prefix)
    // With each pass, we identify a wider prefix of the splitter key
    kth_key_prefix = 0;
    prefix_mask    = 0;

    // The total number of selected items
    total_selected = 0;

    // Zero the first pass's histogram buffer. Every pass but the last re-zeroes the respectively
    // other buffer inside its histogram phase, so no per-pass init phase (and barrier) is needed
    zero_histogram(storage.stage.passes.histogram[0]);
    __syncthreads();

    constexpr int num_passes = ::cuda::ceil_div(max_bit, RadixBits);
    const auto run_pass      = [&](int pass) -> bool {
      const int pass_end_bit   = max_bit - pass * RadixBits;
      const int pass_begin_bit = (::cuda::std::max) (pass_end_bit - RadixBits, 0);
      return run_radix_pass<SelectDirection, IsFullTile>(
        unsigned_keys,
        valid_items,
        k,
        total_selected,
        num_candidates,
        kth_key_prefix,
        prefix_mask,
        pass,
        pass_begin_bit,
        pass_end_bit - pass_begin_bit,
        pass + 1 < num_passes,
        decomposer);
    };

    if constexpr (UnrollBitPasses)
    {
      // Fully unrolled, the per-pass shifts, masks, and histogram-buffer selections become
      // immediates and the compiler can schedule across pass boundaries.
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int pass = 0; pass < num_passes; ++pass)
      {
        if (run_pass(pass))
        {
          break;
        }
      }
    }
    else
    {
      _CCCL_PRAGMA_UNROLL(1)
      for (int pass = 0; pass < num_passes; ++pass)
      {
        if (run_pass(pass))
        {
          break;
        }
      }
    }
    // No trailing barrier is needed before repurposing shared memory: the histograms' last
    // reads precede the final pass's state barrier, and pass_state lives outside the union.
  }

  template <detail::topk::select SelectDirection, bool IsFullTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  select_topk(KeyT (&keys)[items_per_thread], ValueT (&values)[items_per_thread], int k, int valid_items)
  {
    if constexpr (!IsFullTile)
    {
      _CCCL_ASSERT(valid_items > 0 && valid_items <= tile_items, "valid_items must be in [1, tile_items]");
    }

    // TODO (elstehle): Elide this check when k is statically constrained to be positive
    if (k <= 0)
    {
      return;
    }

    // Every valid item is selected. The blocked arrangement already places the valid items in
    // the tile's leading slots, so keys and values can stay where they are.
    // TODO (elstehle): Elide this check when k is statically constrained to be less than the
    // number of items in the tile
    if ((!IsFullTile && k >= valid_items) || k >= tile_items)
    {
      return;
    }

    // Preset the partitioning counters before the radix passes: they live outside the aliased
    // storage union and every pass provides ordering barriers, so the partitioning stage below
    // needs no setup phase or barrier of its own. Tied candidates use a zero-based ticket whose
    // final position is computed as total_selected + ticket.
    if (linear_tid == 0)
    {
      storage.selected_offset[0] = 0;
      storage.selected_offset[1] = 0;
    }

    // Keep a register copy of the original keys: the selected keys are then scattered from the
    // copy, so the keys neither need to be un-twiddled nor does -0.0 need to be tracked and
    // restored (the -0.0 -> +0.0 ranking normalization below is kept, so selection semantics
    // do not depend on MemoizeKeys).
    [[maybe_unused]] KeyT original_keys[items_per_thread];
    if constexpr (MemoizeKeys)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        original_keys[i] = keys[i];
      }
    }

    // TODO (elstehle): Add support for custom decomposers
    identity_decomposer_t decomposer;

    // Get bit-twiddled sortkeys. For float keys, -0.0 is normalized to +0.0 for ranking. When
    // not scattering the original keys, track which keys were -0.0 so we can restore -0.0 in
    // the output via a bitvector.
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
          if constexpr (!MemoizeKeys)
          {
            flip_back_bits[i / 32] |= (1u << (i % 32));
          }
          unsigned_keys[i] = twiddled_zero;
        }
      }
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        unsigned_keys[i] = bit_ordered_conversion::to_bit_ordered(decomposer, unsigned_keys[i]);
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
      unsigned_keys, k, valid_items, total_selected, num_candidates, kth_prefix, prefix_mask, decomposer);

    // Scatter indices of selected items into shared memory (only needed for key-value selection)
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

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const bit_ordered_type key_prefix = unsigned_keys[i] & prefix_mask;

      const bool is_valid = (IsFullTile || linear_tid * items_per_thread + i < valid_items);
      using comparison_t  = ::cuda::std::
        conditional_t<SelectDirection == detail::topk::select::min, ::cuda::std::less<>, ::cuda::std::greater<>>;
      const bool is_selected  = comparison_t{}(key_prefix, kth_prefix);
      const bool is_candidate = key_prefix == kth_prefix;

      // We differentiate between candidates and selected only if not all candidates make it into the top-k items.
      const int item_class = (!select_all_candidates) && is_candidate ? 1 : 0;

      // Without the original-key copy, untwiddle the key in place before storing it to shared
      // memory
      if constexpr (!MemoizeKeys)
      {
        unsigned_keys[i] = bit_ordered_conversion::from_bit_ordered(decomposer, unsigned_keys[i]);
      }

      if (is_valid && (is_selected || is_candidate))
      {
        const auto ticket          = atomicAdd_block(&storage.selected_offset[item_class], histo_counter_t{1});
        const auto selected_offset = (item_class == 1) ? static_cast<histo_counter_t>(total_selected) + ticket : ticket;
        if constexpr (MemoizeKeys)
        {
          storage.stage.exchange.u.keys[selected_offset] = original_keys[i];
        }
        else if constexpr (::cuda::is_floating_point_v<KeyT>)
        {
          storage.stage.exchange.u.keys[selected_offset] =
            (flip_back_bits[i / 32] & (1u << (i % 32))) ? KeyT(-0.0) : ::cuda::std::bit_cast<KeyT>(unsigned_keys[i]);
        }
        else
        {
          storage.stage.exchange.u.keys[selected_offset] = ::cuda::std::bit_cast<KeyT>(unsigned_keys[i]);
        }
        if constexpr (!keys_only)
        {
          scatter_indices[i] = static_cast<int>(selected_offset);
        }
      }
    }

    // Ensure all threads have finished writing to shared memory
    __syncthreads();

    // Gather selected items into thread registers for return. Slots beyond k are left as they
    // are: with MemoizeKeys they still hold bit-twiddled keys, without it the in-place untwiddled
    // ones. Both are unspecified per the contract above, and not restoring them keeps the
    // memoized copy from having to live across the exchange.
    // TODO (elstehle): Revisit whether the slots beyond k should be left untouched (i.e., holding
    // the original keys) rather than unspecified, once this becomes part of a public interface.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const int buffer_idx = linear_tid * items_per_thread + i;
      if (buffer_idx < k)
      {
        keys[i] = storage.stage.exchange.u.keys[buffer_idx];
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
          storage.stage.exchange.u.values[scatter_indices[i]] = values[i];
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
          values[i] = storage.stage.exchange.u.values[buffer_idx];
        }
      }
    }
  }

public:
  struct TempStorage : Uninitialized<TempStorage_>
  {};

  _CCCL_DEVICE_API _CCCL_FORCEINLINE block_topk_air(TempStorage& storage)
      : storage(storage.Alias())
      , linear_tid(RowMajorTid(ThreadsPerBlock, 1, 1))
  {}

  template <detail::topk::select SelectDirection, bool IsFullTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void select_keys(KeyT (&keys)[items_per_thread], int k, int valid_items)
  {
    NullType values[ItemsPerThread];
    select_topk<SelectDirection, IsFullTile>(keys, values, k, valid_items);
  }

  template <detail::topk::select SelectDirection, bool IsFullTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  select_pairs(KeyT (&keys)[items_per_thread], ValueT (&values)[items_per_thread], int k, int valid_items)
  {
    select_topk<SelectDirection, IsFullTile>(keys, values, k, valid_items);
  }
};
} // namespace detail
CUB_NAMESPACE_END
