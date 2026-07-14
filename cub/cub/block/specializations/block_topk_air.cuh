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
//! becomes the candidate set for the next pass. No data movement occurs during this stage—only
//! the histogram in shared memory is updated. (2) Partitioning scatters the top-k items (key
//! prefix <= k-th prefix) into shared memory via atomic counters, then each thread reads back
//! its portion. Supports key-only and key-value selection.
//!
//! Latency-oriented structure (see dsb_experiments/BLOCK_TOPK_AIR_ABLATION.md for the per-change
//! measurements backing these choices; reference point (256 threads, 4 items/thread, k=16,
//! float+int32 pairs, B200)):
//!  * The per-pass prefix sum is fused with the k-th-bucket selection: the crossing test runs on
//!    the scan's register results (exclusive = inclusive - count), so the scanned histogram is
//!    never written back to shared memory and the separate "choose" phase and its barrier are
//!    gone (~-83 cycles/pass, all key types).
//!  * The histograms are double-buffered: both buffers are zeroed once up front and the buffer
//!    for pass p+1 is re-zeroed during pass p's histogram phase (its last read precedes the
//!    preceding pass's state barrier), removing the per-pass init phase and its barrier
//!    (~-200 cycles, all key types; the second buffer unions under the exchange storage, and
//!    register pressure/occupancy measurably improve).
//!  * The partitioning counters are zeroed before the radix passes (they live outside the
//!    aliased storage union) and the tied-candidate position is computed as
//!    total_selected + zero-based ticket, removing the partitioning setup phase and two barriers
//!    (~-50 cycles, all key types).
//!  * For key-value selection, a register copy of the original keys is kept across the radix
//!    stage and scattered directly, so no un-twiddling and no -0.0 restoration bitvector is
//!    needed (the -0.0 -> +0.0 ranking normalization is kept, so selection semantics are
//!    unchanged). Not applied to keys-only selection, where the extra register copy measurably
//!    costs more than the un-twiddle it saves.
//!
//! @tparam KeyT
//!   Key type
//!
//! @tparam ThreadsPerBlock
//!   Number of threads in the block
//!
//! @tparam ItemsPerThread
//!   Number of items per thread
//!
//! @tparam ValueT
//!   <b>[optional]</b> Value type (default: NullType, which indicates keys-only selection)
//!
//! @tparam RadixBits
//!   <b>[optional]</b> Number of radix bits per pass (default: 8; a sweep over {8,10,11,12}
//!   found 8 optimal on sm_100 — wider digits grow the scan and the resets faster than they
//!   save passes)
//!
//! @tparam UnrollBitPasses
//!   <b>[optional]</b> When true (default) and a select_* call covers the full bit range of
//!   KeyT (begin_bit == 0, end_bit == sizeof(KeyT)*8 — the common case), the radix-pass loop is
//!   fully unrolled so all shifts, masks, and histogram-buffer selections become immediates and
//!   the first pass's histogram overlaps the prologue. This is the single largest latency lever
//!   (measured -320 cycles random / -513 cycles on 4-pass inputs, +13% throughput), but it is
//!   also the only change with a real resource cost: +16 registers and one fewer resident
//!   blocks-per-SM tier (e.g. 8 -> 5 blocks/SM at 256 threads on sm_100). Choose false when the
//!   surrounding kernel is register- or occupancy-constrained, or when code size matters more
//!   than latency (the unrolled body is emitted once per pass). Calls with a runtime sub-range
//!   of bits always use the rolled loop, regardless of this parameter.
//!
//! @tparam FuseKeyValueExchange
//!   <b>[optional]</b> Key-value selection only (ignored for keys-only). When true, keys and
//!   values are scattered through shared memory together as pairs and gathered once, instead of
//!   two round trips through a key/value-aliased exchange buffer. Removes two barriers and one
//!   full pass over the items (measured -50..-90 cycles latency, +5% throughput), but the
//!   exchange grows from tile_items * max(sizeof(KeyT), sizeof(ValueT)) to
//!   tile_items * sizeof(pair). The default enables it while the pair is at most 8 bytes (e.g.
//!   4B keys + 4B values), where the measured win is largest; for 16-byte pairs (e.g. 8B keys or
//!   8B values) the win shrinks to ~-30 cycles while the exchange doubles again — enable it
//!   there only if the shared-memory budget allows.
template <typename KeyT,
          int ThreadsPerBlock,
          int ItemsPerThread,
          typename ValueT           = NullType,
          int RadixBits             = 8,
          bool UnrollBitPasses      = true,
          bool FuseKeyValueExchange = (sizeof(KeyT) + sizeof(ValueT) <= 8)>
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
  static constexpr bool fuse_exchange     = FuseKeyValueExchange && !keys_only;

  using histo_counter_t = ::cuda::std::uint32_t;
  using block_scan_t    = BlockScan<histo_counter_t, threads_per_block, BLOCK_SCAN_WARP_SCANS>;

  using traits                 = detail::radix::traits_t<KeyT>;
  using bit_ordered_type       = typename traits::bit_ordered_type;
  using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;

  using fundamental_digit_extractor_t = BFEDigitExtractor<KeyT>;

  struct key_value_pair_
  {
    KeyT key;
    ValueT value;
  };
  struct classic_exchange_
  {
    union
    {
      KeyT keys[tile_items];
      ValueT values[tile_items];
    } u;
  };
  struct fused_exchange_
  {
    key_value_pair_ pairs[tile_items];
  };
  using exchange_t = ::cuda::std::conditional_t<fuse_exchange, fused_exchange_, classic_exchange_>;

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

      exchange_t exchange;
    } stage;

    // Outside the aliased union: written by one thread before the pass barrier, read by all
    // after it — must not overlap the exchange writes that follow the final pass.
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
        atomicAdd(&histogram[bucket], histo_counter_t{1});
      }
    }
  }

  // Fused prefix sum over buckets + identification of the bucket that the k-th item falls into.
  // The crossing test runs on the scan's register results (exclusive = inclusive - count), so
  // the scanned histogram is never written back to shared memory and no separate choose phase
  // (with its barrier) is needed.
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

    histo_counter_t inclusive_sums[buckets_per_thread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      inclusive_sums[i] = counts[i];
    }
    block_scan_t(storage.stage.passes.scan_temp_storage).InclusiveSum(inclusive_sums, inclusive_sums);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        const histo_counter_t inclusive = inclusive_sums[i];
        const histo_counter_t exclusive = inclusive - counts[i];
        // If a bug causes less than k candidates in the histogram, the previous pass' pass_state will persist making
        // debugging harder. This assert should catch such bugs. Should there ever be a valid use case for less than k
        // candidates, the pass_state needs to be reset unconditionally.
        _CCCL_ASSERT((bin_idx != num_buckets - 1) || (inclusive >= k),
                     "Less than k candidates have participated in the histogram");

        if (exclusive < k && inclusive >= k)
        {
          storage.pass_state.bucket     = bin_idx;
          storage.pass_state.candidates = inclusive - exclusive;
          storage.pass_state.selected   = exclusive;
        }
      }
    }
  }

  // One radix pass: histogram over the surviving candidates (re-zeroing the other buffer in the
  // same phase), fused scan+choose, and the pass-state update. Returns true when all remaining
  // candidates are amongst the top-k (early exit).
  template <detail::topk::select SelectDirection, bool IsFullTile, typename DecomposerT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE bool run_radix_pass(
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

    histo_counter_t(&histogram)[num_buckets] = storage.stage.passes.histogram[pass & 1];

    // Compute histogram over the current pass's bits, pre-filtered for keys matching the previous pass's prefix mask
    auto filter_op = compare_key_prefix_op<bit_ordered_type>{prefix_mask, kth_key_prefix};
    auto digit_extractor =
      traits::template digit_extractor<fundamental_digit_extractor_t>(pass_begin_bit, pass_bits, decomposer);
    compute_histograms<SelectDirection, IsFullTile>(unsigned_keys, valid_items, digit_extractor, filter_op, histogram);
    if (zero_next_histogram)
    {
      // Re-zero the other buffer for the next pass; its last read preceded the previous pass's
      // state barrier, so this shares the histogram phase instead of needing one of its own
      zero_histogram(storage.stage.passes.histogram[(pass + 1) & 1]);
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
    int begin_bit,
    int end_bit,
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
    _CCCL_ASSERT(begin_bit >= 0 && begin_bit < max_bit, "begin_bit must be in [0, max_bit)");
    _CCCL_ASSERT(end_bit > begin_bit && end_bit <= max_bit, "end_bit must be in (begin_bit, max_bit]");

    // We only consider candidates identified in the previous pass, i.e., ((sortkey & prefix_mask) == kth_prefix)
    // With each pass, we identify a wider prefix of the splitter key
    kth_key_prefix = 0;
    prefix_mask    = 0;

    // The total number of selected items
    total_selected = 0;

    // Zero both histogram buffers once; later passes re-zero the respectively other buffer
    // inside their histogram phase, so no per-pass init phase (and barrier) is needed
    zero_histogram(storage.stage.passes.histogram[0]);
    zero_histogram(storage.stage.passes.histogram[1]);
    __syncthreads();

    if constexpr (UnrollBitPasses)
    {
      // Fast path for the common full-bit-range call: compile-time pass count and bit offsets.
      // All shifts/masks/buffer selections fold to immediates and the first pass's histogram
      // can overlap the prologue — the single largest latency contribution of this class.
      if (begin_bit == 0 && end_bit == max_bit)
      {
        constexpr int full_num_passes = ::cuda::ceil_div(max_bit, RadixBits);
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int pass = 0; pass < full_num_passes; ++pass)
        {
          const int pass_end_bit   = max_bit - pass * RadixBits;
          const int pass_begin_bit = (pass_end_bit - RadixBits > 0) ? pass_end_bit - RadixBits : 0;
          if (run_radix_pass<SelectDirection, IsFullTile>(
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
                pass > 0 && pass + 1 < full_num_passes,
                decomposer))
          {
            break;
          }
        }
        // No trailing barrier is needed before repurposing shared memory: the histograms' last
        // reads precede the final pass's state barrier, and pass_state lives outside the union.
        return;
      }
    }

    const int total_bits = (::cuda::std::max) (end_bit - begin_bit, 0);
    const int num_passes = ::cuda::ceil_div(total_bits, RadixBits);
    for (int pass = 0; pass < num_passes; ++pass)
    {
      // Bit-range of the current pass
      const int pass_end_bit   = end_bit - pass * RadixBits;
      const int pass_begin_bit = (::cuda::std::max) (pass_end_bit - RadixBits, begin_bit);
      if (run_radix_pass<SelectDirection, IsFullTile>(
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
            pass > 0 && pass + 1 < num_passes,
            decomposer))
      {
        break;
      }
    }
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

    // Preset the partitioning counters before the radix passes: they live outside the aliased
    // storage union and every pass provides ordering barriers, so the partitioning stage below
    // needs no setup phase or barrier of its own. Tied candidates use a zero-based ticket whose
    // final position is computed as total_selected + ticket.
    if (linear_tid == 0)
    {
      storage.selected_offset[0] = 0;
      storage.selected_offset[1] = 0;
    }

    // For key-value selection, keep a register copy of the original keys: the selected keys are
    // then scattered from the copy, so the keys neither need to be un-twiddled nor does -0.0
    // need to be tracked and restored (the -0.0 -> +0.0 ranking normalization below is kept, so
    // selection semantics are unchanged). For keys-only selection the extra register copy costs
    // more than the un-twiddle it saves, so the classic path is used there.
    [[maybe_unused]] KeyT original_keys[items_per_thread];
    if constexpr (!keys_only)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        original_keys[i] = keys[i];
      }
    }

    // TODO (elstehle): Add support for custom decomposers
    identity_decomposer_t decomposer;

    // Get bit-twiddled sortkeys. For float keys, -0.0 is normalized to +0.0 for ranking; for
    // keys-only selection, track which keys were -0.0 so we can restore -0.0 in the output via
    // a bitvector (key-value selection restores from the register copy instead).
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
          if constexpr (keys_only)
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

    // Scatter indices of selected items into shared memory (only needed for key-value selection
    // through the classic key/value-aliased exchange).
    [[maybe_unused]] int scatter_indices[items_per_thread];
    if constexpr (!keys_only && !fuse_exchange)
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

      // Keys-only: untwiddle the key before storing in shared memory (key-value selection
      // scatters the original register copy instead)
      if constexpr (keys_only)
      {
        unsigned_keys[i] = bit_ordered_conversion::from_bit_ordered(decomposer, unsigned_keys[i]);
      }

      if (is_valid && (is_selected || is_candidate))
      {
        const auto ticket          = atomicAdd(&storage.selected_offset[item_class], histo_counter_t{1});
        const auto selected_offset = (item_class == 1) ? static_cast<histo_counter_t>(total_selected) + ticket : ticket;
        if constexpr (fuse_exchange)
        {
          storage.stage.exchange.pairs[selected_offset] = key_value_pair_{original_keys[i], values[i]};
        }
        else if constexpr (keys_only)
        {
          if constexpr (::cuda::is_floating_point_v<KeyT>)
          {
            storage.stage.exchange.u.keys[selected_offset] =
              (flip_back_bits[i / 32] & (1u << (i % 32))) ? KeyT(-0.0) : ::cuda::std::bit_cast<KeyT>(unsigned_keys[i]);
          }
          else
          {
            storage.stage.exchange.u.keys[selected_offset] = ::cuda::std::bit_cast<KeyT>(unsigned_keys[i]);
          }
        }
        else
        {
          storage.stage.exchange.u.keys[selected_offset] = original_keys[i];
          scatter_indices[i]                             = static_cast<int>(selected_offset);
        }
      }
    }

    // Ensure all threads have finished writing to shared memory
    __syncthreads();

    // Gather selected items into thread registers for return.
    if constexpr (fuse_exchange)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        const int buffer_idx = linear_tid * items_per_thread + i;
        if (buffer_idx < k)
        {
          const key_value_pair_ pair = storage.stage.exchange.pairs[buffer_idx];
          keys[i]                    = pair.key;
          values[i]                  = pair.value;
        }
        else
        {
          // The register keys are still bit-twiddled; restore from the original copy
          keys[i] = original_keys[i];
        }
      }
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        const int buffer_idx = linear_tid * items_per_thread + i;
        if (buffer_idx < k)
        {
          keys[i] = storage.stage.exchange.u.keys[buffer_idx];
        }
        else if constexpr (!keys_only)
        {
          // The register keys are still bit-twiddled; restore from the original copy
          keys[i] = original_keys[i];
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
  }

public:
  struct TempStorage : Uninitialized<TempStorage_>
  {};

  _CCCL_DEVICE_API _CCCL_FORCEINLINE block_topk_air(TempStorage& storage)
      : storage(storage.Alias())
      , linear_tid(RowMajorTid(ThreadsPerBlock, 1, 1))
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
