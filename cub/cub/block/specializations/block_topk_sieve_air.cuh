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

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <int ItemsPerThread>
class block_topk_key_states;

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

//! @brief Block-level radix sieve specialization (state-classification only).
//!
template <typename KeyT, int BlockDimX, int RadixBits = 8>
class block_topk_sieve_air
{
private:
  static constexpr int threads_per_block  = BlockDimX;
  static constexpr int num_buckets        = int{1u << RadixBits};
  static constexpr int buckets_per_thread = ::cuda::ceil_div(num_buckets, threads_per_block);

  using histo_counter_t = ::cuda::std::uint32_t;
  using block_scan_t    = BlockScan<histo_counter_t, threads_per_block, BLOCK_SCAN_WARP_SCANS>;

  using traits                 = detail::radix::traits_t<KeyT>;
  using bit_ordered_type       = typename traits::bit_ordered_type;
  using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;
  using bit_ordered_inversion  = typename traits::bit_ordered_inversion_policy;

  using fundamental_digit_extractor_t = BFEDigitExtractor<KeyT>;

  struct TempStorage_
  {
    histo_counter_t histogram[num_buckets];
    typename block_scan_t::TempStorage scan_temp_storage;
    struct
    {
      histo_counter_t selected;
      histo_counter_t candidates;
      int bucket;
    } pass_state;
  };

  TempStorage_& storage;
  int linear_tid;

  // Initialize histogram bins to zero
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void init_histograms()
  {
    // Initialize histogram bin counts to zeros
    int histo_offset = 0;

    // Loop unrolling is beneficial for performance here
    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + threads_per_block <= num_buckets; histo_offset += threads_per_block)
    {
      storage.histogram[histo_offset + linear_tid] = 0;
    }
    // Finish up with guarded initialization if necessary
    if ((num_buckets % threads_per_block != 0) && (histo_offset + linear_tid < num_buckets))
    {
      storage.histogram[histo_offset + linear_tid] = 0;
    }
  }

  // Compute histogram over keys
  template <detail::topk::select SelectDirection,
            bool IsFullTile,
            int ItemsPerThread,
            typename DigitExtractorT,
            typename FilterOpT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void compute_histograms(
    const bit_ordered_type (&unsigned_keys)[ItemsPerThread],
    block_topk_key_states<ItemsPerThread>& states,
    DigitExtractorT digit_extractor,
    FilterOpT filter_op)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      const bit_ordered_type key = unsigned_keys[i];
      if ((IsFullTile || states.is_candidate(i)) && filter_op(key))
      {
        const auto digit  = static_cast<int>(digit_extractor.Digit(key));
        const auto bucket = (SelectDirection == detail::topk::select::min) ? digit : (num_buckets - 1 - digit);
        atomicAdd(&storage.histogram[bucket], histo_counter_t{1});
      }
    }
  }

  // Compute prefix sum over buckets
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void compute_bin_offsets()
  {
    // TODO: If we ever have a block-scan with smem input (vs. rmem), use it here.
    histo_counter_t thread_buckets[buckets_per_thread]{};
    const int base = linear_tid * buckets_per_thread;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        thread_buckets[i] = storage.histogram[bin_idx];
      }
    }

    block_scan_t(storage.scan_temp_storage).InclusiveSum(thread_buckets, thread_buckets);

    // TODO(pauleonix): There is no need to write and read the scanned histogram to/from smem:
    // Keep thread_buckets[0] of each thread from before the scan and compute prev[0] = cur[0] - thread_buckets[0].
    // Alternatively, compute the exclusive scan out-of-place and manually compute the inclusive scan from it and
    // thread_buckets.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        storage.histogram[bin_idx] = thread_buckets[i];
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
        const histo_counter_t prev = (bin_idx == 0) ? 0 : storage.histogram[bin_idx - 1];
        const histo_counter_t cur  = storage.histogram[bin_idx];
        _CCCL_ASSERT(cur >= prev, "Histogram bin count is not monotonically increasing");
        // If a bug causes less than k candidates in the histogram, the previous pass' pass_state will persist making
        // debugging harder. This assert should catch such bugs. Should there ever be a valid use case for less than k
        // candidates, the pass_state needs to be reset unconditionally (e.g. in init_histograms()).
        _CCCL_ASSERT((bin_idx != num_buckets - 1) || (cur >= k),
                     "Less than k candidates have participated in the histogram");

        if (prev < k && cur >= k)
        {
          storage.pass_state.bucket     = bin_idx;
          storage.pass_state.candidates = cur - prev;
          storage.pass_state.selected   = prev;
        }
      }
    }
  }

  template <detail::topk::select SelectDirection, bool IsFullTile, int ItemsPerThread, typename DecomposerT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void get_kth_key_prefix(
    bit_ordered_type (&unsigned_keys)[ItemsPerThread],
    int k,
    block_topk_key_states<ItemsPerThread>& states,
    int begin_bit,
    int end_bit,
    int& total_selected,
    int& num_candidates,
    bit_ordered_type& kth_key_prefix,
    bit_ordered_type& prefix_mask,
    DecomposerT decomposer = DecomposerT{})
  {
    // Preconditions
    [[maybe_unused]] constexpr int tile_items = threads_per_block * ItemsPerThread;
    _CCCL_ASSERT(k > 0 && k <= tile_items, "k must be in (0, tile_items]");
    [[maybe_unused]] constexpr int max_bit = int(sizeof(KeyT) * 8);
    _CCCL_ASSERT(begin_bit >= 0 && begin_bit < max_bit, "begin_bit must be in [0, max_bit)");
    _CCCL_ASSERT(end_bit > begin_bit && end_bit <= max_bit, "end_bit must be in (begin_bit, max_bit]");

    // We only consider candidates identified in the previous pass, i.e., ((sortkey & prefix_mask) == kth_prefix)
    // With each pass, we identify a wider prefix of the splitter key
    kth_key_prefix = 0;
    prefix_mask    = 0;

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
      compute_histograms<SelectDirection, IsFullTile>(unsigned_keys, states, digit_extractor, filter_op);
      __syncthreads();

      // Compute prefix sum over buckets
      compute_bin_offsets();
      __syncthreads();

      // Identify the bucket that the k-th item falls into
      choose_bucket(k);
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
      if (num_candidates == k)
      {
        break;
      }
    }

    // Ensure we can repurpose shared memory after the multi-pass stage
    __syncthreads();
  }

  template <detail::topk::select SelectDirection, bool IsFullTile, int ItemsPerThread>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  select_topk(KeyT (&keys)[ItemsPerThread], block_topk_key_states<ItemsPerThread>& states, int begin_bit, int end_bit)
  {
    // TODO (elstehle): Add support for custom decomposers
    identity_decomposer_t decomposer;

    // Get bit-twiddled sortkeys. For float keys, track which were -0.0 (normalized to +0.0 for ranking) so we can
    // restore -0.0 in the output via a bitvector; no extra key buffer.
    bit_ordered_type(&unsigned_keys)[ItemsPerThread] = reinterpret_cast<bit_ordered_type(&)[ItemsPerThread]>(keys);
    constexpr int flip_back_num_words                = ::cuda::ceil_div(ItemsPerThread, 32);
    [[maybe_unused]] ::cuda::std::uint32_t flip_back_bits[flip_back_num_words] = {};
    if constexpr (::cuda::is_floating_point_v<KeyT>)
    {
      const bit_ordered_type twiddled_minus_zero =
        Traits<KeyT>::TwiddleIn(bit_ordered_type(1) << (8 * sizeof(bit_ordered_type) - 1));
      const bit_ordered_type twiddled_zero = Traits<KeyT>::TwiddleIn(0);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < ItemsPerThread; ++i)
      {
        unsigned_keys[i] = bit_ordered_conversion::to_bit_ordered(decomposer, unsigned_keys[i]);
        if (unsigned_keys[i] == twiddled_minus_zero)
        {
          flip_back_bits[i / 32] |= (1u << (i % 32));
          unsigned_keys[i] = twiddled_zero;
        }
      }
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < ItemsPerThread; ++i)
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
    auto total_selected = IsFullTile ? 0 : states.num_selected();
    // The number of candidates that compare equal to the k-th key's prefix
    auto num_candidates = IsFullTile ? BlockDimX * ItemsPerThread : states.num_candidates();

    // Identify the prefix of the k-th key
    get_kth_key_prefix<SelectDirection, IsFullTile>(
      unsigned_keys,
      states.k() - states.num_selected(),
      states,
      begin_bit,
      end_bit,
      total_selected,
      num_candidates,
      kth_prefix,
      prefix_mask,
      decomposer);

    // Update the states before untwiddling the key
    states.set_num_selected(total_selected);
    states.set_num_candidates(num_candidates);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      if (states.is_candidate(i))
      {
        using comparison_t = ::cuda::std::
          conditional_t<SelectDirection == detail::topk::select::min, ::cuda::std::less<>, ::cuda::std::greater<>>;
        const bit_ordered_type key_prefix = unsigned_keys[i] & prefix_mask;
        if (comparison_t{}(key_prefix, kth_prefix))
        {
          states.set_selected(i);
        }
        else if (comparison_t{}(kth_prefix, key_prefix))
        {
          states.set_rejected(i);
        }
      }
      // Un-twiddle the key
      if constexpr (::cuda::is_floating_point_v<KeyT>)
      {
        const bit_ordered_type twiddled_minus_zero =
          Traits<KeyT>::TwiddleIn(bit_ordered_type(1) << (8 * sizeof(bit_ordered_type) - 1));
        unsigned_keys[i] = (flip_back_bits[i / 32] & (1u << (i % 32))) ? twiddled_minus_zero : unsigned_keys[i];
      }
      unsigned_keys[i] = bit_ordered_conversion::from_bit_ordered(decomposer, unsigned_keys[i]);
    }
  }

public:
  struct TempStorage : Uninitialized<TempStorage_>
  {};

  _CCCL_DEVICE_API _CCCL_FORCEINLINE explicit block_topk_sieve_air(TempStorage& storage)
      : storage(storage.Alias())
      , linear_tid(RowMajorTid(threads_per_block, 1, 1))
  {}

  template <detail::topk::select SelectDirection, bool IsFullTile, int ItemsPerThread>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void refine_keys(
    KeyT (&keys)[ItemsPerThread],
    block_topk_key_states<ItemsPerThread>& states,
    int begin_bit = 0,
    int end_bit   = sizeof(KeyT) * 8)
  {
    select_topk<SelectDirection, IsFullTile>(keys, states, begin_bit, end_bit);
  }
};
} // namespace detail

CUB_NAMESPACE_END
