// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Helpers shared by the radix-select passes of cub::DeviceTopK and cub::DeviceBatchedTopK: pass and start-bit
//! bookkeeping, the k-th key prefix storage, the key bit ordering, and the bin-extraction and candidate-classification
//! operators that the agents are parameterized on.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/detail/numeric_traits.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/climits>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
template <typename KeyT, bool CanTwiddle = detail::radix::can_twiddle<KeyT>>
struct key_prefix_storage_t;

template <typename KeyT>
struct key_prefix_storage_t<KeyT, true>
{
  using bits_t = typename Traits<KeyT>::UnsignedBits;
  bits_t bits;
};

// Calculates the number of passes needed for a type T with BitsPerPass bits processed per pass.
template <typename T>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int calc_num_passes(int bits_per_pass)
{
  return ::cuda::ceil_div<int>(sizeof(T) * CHAR_BIT, bits_per_pass);
}

template <int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int calc_num_passes(const int total_bits)
{
  return ::cuda::ceil_div<int>(total_bits, BitsPerPass);
}

// Calculates the starting bit for a given pass (bit 0 is the least significant (rightmost) bit).
// We process the input from the most to the least significant bit. This way, we can skip some passes in the end.
template <typename T, int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int calc_start_bit(const int pass)
{
  int start_bit = int{sizeof(T)} * CHAR_BIT - (pass + 1) * BitsPerPass;
  if (start_bit < 0)
  {
    start_bit = 0;
  }
  return start_bit;
}

template <int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int calc_start_bit(const int total_bits, const int pass)
{
  int start_bit = total_bits - (pass + 1) * BitsPerPass;
  if (start_bit < 0)
  {
    start_bit = 0;
  }
  return start_bit;
}

// Bit-vector for accumulating prefix digits via funnel shift. Each pass shifts the existing
// contents left by BitsPerPass and ORs the new bucket at the bottom. Sized to hold all
// decomposed bits of KeyT plus headroom for the shift padding of the last pass.
template <typename KeyT>
struct key_prefix_storage_t<KeyT, false>
{
  static constexpr int num_words = ::cuda::ceil_div<int>(sizeof(KeyT) * CHAR_BIT + 31, 32);
  unsigned int words[num_words];

  // Funnel-shifts the entire bit-vector left by `shift` positions and inserts `value` into the
  // vacated low bits. Each word receives carry bits from its lower neighbor (high-to-low order
  // so each word reads its neighbor's original value). The final word is filled from `value`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void shift_or(int shift, unsigned int value)
  {
    _CCCL_ASSERT(shift > 0 && shift < 32, "shift_or requires 0 < shift < 32");
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = num_words - 1; i > 0; --i)
    {
      words[i] = __funnelshift_l(words[i - 1], words[i], shift);
    }
    words[0] = (words[0] << shift) | value;
  }
};

template <typename KeyT, int BitsPerPass>
_CCCL_DEVICE _CCCL_FORCEINLINE void
set_kth_key_bits(key_prefix_storage_t<KeyT>& prefix, const int pass, const int bin_index)
{
  if constexpr (detail::radix::can_twiddle<KeyT>)
  {
    using bits_t        = typename Traits<KeyT>::UnsignedBits;
    const int start_bit = calc_start_bit<KeyT, BitsPerPass>(pass);
    const bits_t bucket = bin_index;
    prefix.bits |= static_cast<bits_t>(bucket) << start_bit;
  }
  else
  {
    prefix.shift_or(BitsPerPass, bin_index);
  }
}

enum class candidate_class
{
  // The given candidate is definitely amongst the top-k items
  selected,
  // The given candidate may or may not be amongst the top-k items
  candidate,
  // The given candidate is definitely not amongst the top-k items
  rejected
};

// Used in the bin ID calculation to exclude bits unrelated to the current pass
template <typename T, int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr unsigned calc_mask(const int pass)
{
  const int num_bits = calc_start_bit<T, BitsPerPass>(pass - 1) - calc_start_bit<T, BitsPerPass>(pass);
  return (1 << num_bits) - 1;
}

// Bit ordering used by the top-k radix passes. Without `NormalizeMinusZero` this is `Traits<T>::TwiddleIn`. With it,
// floating-point keys use a variant that negates (rather than complements) keys with the sign bit set. `TwiddleIn`
// complements negative keys, which leaves -0.0 at `0x7FFF...`, one below +0.0 at `0x8000...`. Negation is that
// complement plus one, so it shifts the whole negative half up by one and -0.0 lands exactly on +0.0 with no separate
// fix-up. Every sign-clear pattern is unchanged and every sign-set pattern moves by exactly +1, so the relative order
// is preserved everywhere (-NaN stays below -inf, +NaN above +inf) and the only new equivalence is the two zeros.
template <typename T, bool NormalizeMinusZero>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE typename Traits<T>::UnsignedBits
twiddle_in_bits(typename Traits<T>::UnsignedBits bits)
{
  using bits_t = typename Traits<T>::UnsignedBits;
  // Every FLOATING_POINT specialization of `Traits` asserts `::cuda::is_floating_point_v<T>`, so this branch covers
  // every key type whose `TwiddleIn` is the floating-point one.
  if constexpr (NormalizeMinusZero && ::cuda::is_floating_point_v<T>)
  {
    // 0 for sign-clear keys, ~0 for sign-set keys. Unsigned arithmetic only.
    const bits_t neg_mask =
      static_cast<bits_t>(bits_t{0} - static_cast<bits_t>(bits >> (CHAR_BIT * sizeof(bits_t) - 1)));
    // `neg_mask | HIGH_BIT` is the mask `Traits<T>::TwiddleIn` xors with (HIGH_BIT for sign-clear keys, ~0 for
    // sign-set keys). Adding `neg_mask` first subtracts one from sign-set keys, so after the xor they are
    // `~(bits - 1) == -bits` while sign-clear keys are `bits | HIGH_BIT` as before. Spelled add-then-xor (rather than
    // xor-then-subtract) so the xor is the last op: the twiddle is a shift, an add and one three-input logic op, and
    // the complement that the max direction applies afterwards folds into that same logic op.
    return static_cast<bits_t>(
      static_cast<bits_t>(bits + neg_mask) ^ static_cast<bits_t>(neg_mask | Traits<T>::HIGH_BIT));
  }
  else
  {
    return Traits<T>::TwiddleIn(bits);
  }
}

// Get the bin ID from the value of element
template <typename T,
          select SelectDirection,
          int BitsPerPass,
          typename DecomposerT,
          bool NormalizeMinusZero,
          bool CanTwiddle = detail::radix::can_twiddle<T>>
struct extract_bin_op_t;

template <typename T, select SelectDirection, int BitsPerPass, typename DecomposerT, bool NormalizeMinusZero>
struct extract_bin_op_t<T, SelectDirection, BitsPerPass, DecomposerT, NormalizeMinusZero, true>
{
  static constexpr bool is_descending = SelectDirection != select::min;
  using bit_ordered_type              = typename Traits<T>::UnsignedBits;

  int pass{};
  int start_bit{};
  unsigned mask{};

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE extract_bin_op_t(int pass, int /*total_bits*/, DecomposerT /*decomposer*/)
      : pass(pass)
      , start_bit(calc_start_bit<T, BitsPerPass>(pass))
      , mask(calc_mask<T, BitsPerPass>(pass))
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int operator()(T key) const
  {
    auto bits = reinterpret_cast<typename Traits<T>::UnsignedBits&>(key);
    bits      = twiddle_in_bits<T, NormalizeMinusZero>(bits);
    if constexpr (SelectDirection != select::min)
    {
      bits = ~bits;
    }
    const int bucket = (bits >> start_bit) & mask;
    return bucket;
  }
};

template <typename T, select SelectDirection, int BitsPerPass, typename DecomposerT, bool NormalizeMinusZero>
struct extract_bin_op_t<T, SelectDirection, BitsPerPass, DecomposerT, NormalizeMinusZero, false>
{
  static constexpr bool is_descending = SelectDirection != select::min;
  using radix_traits_t                = detail::radix::traits_t<T>;
  using bit_ordered_type              = typename radix_traits_t::bit_ordered_type;
  using digit_extractor_t = typename radix_traits_t::template digit_extractor_t<ShiftDigitExtractor<T>, DecomposerT>;

  DecomposerT decomposer{};
  digit_extractor_t digit_extractor;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE extract_bin_op_t(int pass, int total_bits, DecomposerT decomposer)
      : decomposer(decomposer)
      , digit_extractor(radix_traits_t::template digit_extractor<ShiftDigitExtractor<T>>(
          calc_start_bit<BitsPerPass>(total_bits, pass),
          calc_start_bit<BitsPerPass>(total_bits, pass - 1) - calc_start_bit<BitsPerPass>(total_bits, pass),
          decomposer))
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int operator()(T key) const
  {
    bit_ordered_type ordered = key;
    ordered                  = RadixSortTwiddle<is_descending, T>::In(ordered, decomposer);
    return static_cast<int>(digit_extractor.Digit(ordered));
  }
};

// Check if the input element is still a candidate for the target pass.
template <typename T,
          select SelectDirection,
          int BitsPerPass,
          typename DecomposerT,
          bool NormalizeMinusZero,
          bool CanTwiddle = detail::radix::can_twiddle<T>>
struct identify_candidates_op_t;

template <typename T, select SelectDirection, int BitsPerPass, typename DecomposerT, bool NormalizeMinusZero>
struct identify_candidates_op_t<T, SelectDirection, BitsPerPass, DecomposerT, NormalizeMinusZero, true>
{
  using unsigned_bits_t = typename Traits<T>::UnsignedBits;
  using key_prefix_t    = key_prefix_storage_t<T>;
  unsigned_bits_t* kth_key_bits;
  int start_bit;
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
  identify_candidates_op_t(key_prefix_t* kth_key_bits, int pass, int /*total_bits*/, DecomposerT /*decomposer*/)
      : kth_key_bits(&kth_key_bits->bits)
      , start_bit(calc_start_bit<T, BitsPerPass>(pass - 1))
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE candidate_class operator()(T key) const
  {
    auto bits = reinterpret_cast<unsigned_bits_t&>(key);
    bits      = twiddle_in_bits<T, NormalizeMinusZero>(bits); // must match extract_bin_op_t, which built kth_key_bits

    if constexpr (SelectDirection != select::min)
    {
      bits = ~bits;
    }

    bits = (bits >> start_bit) << start_bit;

    return (bits < *kth_key_bits) ? candidate_class::selected
         : (bits == *kth_key_bits)
           ? candidate_class::candidate
           : candidate_class::rejected;
  }
};

template <typename T, select SelectDirection, int BitsPerPass, typename DecomposerT, bool NormalizeMinusZero>
struct identify_candidates_op_t<T, SelectDirection, BitsPerPass, DecomposerT, NormalizeMinusZero, false>
{
  static constexpr bool is_descending = SelectDirection != select::min;
  using radix_traits_t                = detail::radix::traits_t<T>;
  using bit_ordered_type              = typename radix_traits_t::bit_ordered_type;
  using key_prefix_t                  = key_prefix_storage_t<T>;

  key_prefix_t* kth_key_bits{};
  int pass{};
  int total_bits{};
  DecomposerT decomposer{};

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
  identify_candidates_op_t(key_prefix_t* kth_key_bits, int pass, int total_bits, DecomposerT decomposer)
      : kth_key_bits(kth_key_bits)
      , pass(pass)
      , total_bits(total_bits)
      , decomposer(decomposer)
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE candidate_class operator()(T key) const
  {
    if (pass <= 0)
    {
      return candidate_class::candidate;
    }

    bit_ordered_type ordered = key;
    ordered                  = RadixSortTwiddle<is_descending, T>::In(ordered, decomposer);

    // Build the key's prefix using the same funnel shift as set_kth_key_bits
    key_prefix_t key_prefix{};
    for (int prefix_pass = 0; prefix_pass < pass; ++prefix_pass)
    {
      const int start_bit = calc_start_bit<BitsPerPass>(total_bits, prefix_pass);
      const int num_bits =
        calc_start_bit<BitsPerPass>(total_bits, prefix_pass - 1) - calc_start_bit<BitsPerPass>(total_bits, prefix_pass);
      auto extractor =
        radix_traits_t::template digit_extractor<ShiftDigitExtractor<T>>(start_bit, num_bits, decomposer);
      key_prefix.shift_or(BitsPerPass, static_cast<unsigned int>(extractor.Digit(ordered)));
    }

    // Compare word-by-word from MSB to LSB
    const int total_prefix_bits = pass * BitsPerPass;
    const int top_word_idx      = (total_prefix_bits - 1) / 32;
    const int bits_in_top_word  = ((total_prefix_bits - 1) % 32) + 1;

    // Top word may be partially filled
    {
      unsigned int key_w = key_prefix.words[top_word_idx];
      unsigned int kth_w = kth_key_bits->words[top_word_idx];
      if (bits_in_top_word < 32)
      {
        const unsigned int mask = (1u << bits_in_top_word) - 1u;
        key_w &= mask;
        kth_w &= mask;
      }
      if (key_w < kth_w)
      {
        return candidate_class::selected;
      }
      if (key_w > kth_w)
      {
        return candidate_class::rejected;
      }
    }

    // Remaining words are fully populated
    for (int w = top_word_idx - 1; w >= 0; --w)
    {
      if (key_prefix.words[w] < kth_key_bits->words[w])
      {
        return candidate_class::selected;
      }
      if (key_prefix.words[w] > kth_key_bits->words[w])
      {
        return candidate_class::rejected;
      }
    }

    return candidate_class::candidate;
  }
};
} // namespace detail::topk

CUB_NAMESPACE_END
