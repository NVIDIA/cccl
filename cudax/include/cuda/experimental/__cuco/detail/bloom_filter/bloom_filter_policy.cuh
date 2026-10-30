//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_BLOOM_FILTER_BLOOM_FILTER_POLICY_CUH
#define _CUDAX___CUCO_DETAIL_BLOOM_FILTER_BLOOM_FILTER_POLICY_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cccl_config>
#include <cuda/__cmath/pow2.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/bloom_filter_modes.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__bloom_filter_ns
{
//! @brief Sectorized Bloom filter policy with multiplicative-hashing fingerprint generation.
//!
//! Implements the Sectorized Bloom Filter (SBF) variant from "Optimizing Bloom Filters for Modern
//! GPU Architectures" (arXiv:2512.15595).
//!
//! Each key selects exactly one fixed-size block of `_WordsPerBlock` words: the upper 32 bits of
//! the 64-bit hash pick the block via multiply-shift, and the lower 32 bits drive compile-time
//! salt-based multiplicative hashing that distributes `_PatternBits` set bits across that block's
//! words (so a 64-bit hash function is required). Confining a key's probes to one block keeps a
//! lookup to a single contiguous region, minimizing memory transactions; throughput is best while
//! the whole filter fits the GPU cache domain, so sizing it relative to L2 is the main lever.
//!
//! `add` and `contains` take independent vectorization layouts (horizontal = cooperative-group
//! size, vertical = contiguous words per thread per step) because they favor opposite access
//! patterns: `add` spreads a block's words across cooperating threads so atomic writes proceed in
//! parallel (default: fully horizontal), while `contains` assigns at most one 32-byte sector per
//! thread with wide, coalesced loads by default. `_PatternBits` trades false-positive rate
//! against space.
//!
//! @note This class should NOT be used directly. Use
//! `cuda::experimental::cuco::bloom_filter_policy` instead.
//!
//! @tparam _Hash 64-bit hash functor whose call operator returns `cuda::std::uint64_t`
//! @tparam _Word Underlying word type of a filter block. Must be `unsigned int` or
//! `unsigned long long` for native atomic operations
//! @tparam _WordsPerBlock Words per filter block. Must be a power of two and <= 32
//! @tparam _PatternBits Number of fingerprint bits (k in the paper)
//! @tparam _AddHorizontalLayout Cooperative-group size used for `add` (the paper's Theta)
//! @tparam _AddVerticalLayout Contiguous words processed per thread per `add` step (the paper's Phi)
//! @tparam _ContainsHorizontalLayout Cooperative-group size used for `contains`
//! @tparam _ContainsVerticalLayout Contiguous words processed per thread per `contains` step
//! @tparam _ConditionalAdd Whether `add` skips redundant atomic writes
//! @tparam _EarlyExitContains Whether `contains` short-circuits on the first missing slice
template <class _Hash,
          class _Word,
          int _WordsPerBlock,
          int _PatternBits,
          int _AddHorizontalLayout,
          int _AddVerticalLayout,
          int _ContainsHorizontalLayout,
          int _ContainsVerticalLayout,
          ::cuda::experimental::cuco::conditional_add_mode _ConditionalAdd,
          ::cuda::experimental::cuco::early_exit_contains_mode _EarlyExitContains>
class __bloom_filter_policy
{
public:
  using hasher    = _Hash; ///< 64-bit hash functor type
  using word_type = _Word; ///< Underlying filter-block word type

  static constexpr int words_per_block = _WordsPerBlock; ///< Number of words per filter block
  static constexpr int pattern_bits    = _PatternBits; ///< Fingerprint bits per key

  static constexpr int add_horizontal_layout      = _AddHorizontalLayout; ///< Horizontal layout of `add`
  static constexpr int add_vertical_layout        = _AddVerticalLayout; ///< Vertical layout of `add`
  static constexpr int contains_horizontal_layout = _ContainsHorizontalLayout; ///< Horizontal layout of `contains`
  static constexpr int contains_vertical_layout   = _ContainsVerticalLayout; ///< Vertical layout of `contains`

  //! Read-before-atomic on `add` (skips redundant writes)
  static constexpr ::cuda::experimental::cuco::conditional_add_mode conditional_add = _ConditionalAdd;
  //! Short-circuit `contains` on the first missing slice
  static constexpr ::cuda::experimental::cuco::early_exit_contains_mode early_exit_contains = _EarlyExitContains;

private:
  static constexpr int __max_salts                                                = 64;
  static constexpr ::cuda::std::array<::cuda::std::uint32_t, __max_salts> __salts = {
    0x47B6137BU, 0x44974D91U, 0x8824AD5BU, 0xA2B7289DU, 0x705495C7U, 0x2DF1424BU, 0x9EFC4947U, 0x5C6BFB31U,
    0xB24BCDFFU, 0xB6843D6DU, 0x6DB04543U, 0x3A12EFDDU, 0xB0DDD463U, 0x8D22F6E7U, 0xB82F1E53U, 0x7DB9F86BU,
    0xC7AFE639U, 0xFB135CD7U, 0x693256E1U, 0x9466D871U, 0x23D3D02FU, 0x6461D049U, 0x66A91621U, 0xBAA3006FU,
    0x52FB8D99U, 0x3EA88B4FU, 0x0F470CFDU, 0xB1DB79A5U, 0x9809FCD1U, 0xBCED4445U, 0x2EB7C737U, 0x2CEA6803U,
    0x156F1955U, 0x8813C027U, 0xA26819F9U, 0x4C3B57BDU, 0x7DF94487U, 0xB975E769U, 0xB8F20CB5U, 0x5C9E2E77U,
    0x5FB1735FU, 0x3A6F759BU, 0x3C090923U, 0xFCED424DU, 0xA187A6A9U, 0x6F070A41U, 0x2C85233BU, 0x7E62258BU,
    0x2771EF17U, 0x13BBF093U, 0x4FF059E5U, 0xE3CE3D0FU, 0xF1B4789FU, 0x9FBB6173U, 0x6A320CF5U, 0x1BE2C481U,
    0x7BA8222BU, 0x6FD619B3U, 0x7B1BBF0DU, 0x8B8993ADU, 0x448ECA95U, 0x82AB09D9U, 0x2CE53909U, 0x4F548685U};

  static constexpr int __word_bits = ::cuda::std::numeric_limits<word_type>::digits;

public:
  //! Upper bound on the number of filter blocks
  static constexpr ::cuda::std::size_t max_filter_blocks = ::cuda::std::numeric_limits<::cuda::std::uint32_t>::max();
  //! Lower bound on `pattern_bits`: at least one bit per word so every word contributes.
  static constexpr int min_pattern_bits = words_per_block;
  //! Upper bound on `pattern_bits`: the total number of bits in a filter block, capped by the
  //! number of available salts.
  static constexpr int max_pattern_bits = ::cuda::std::min(__word_bits * words_per_block, __max_salts);

private:
  static constexpr int __bit_index_width =
    static_cast<int>(::cuda::std::bit_width(static_cast<::cuda::std::uint32_t>(__word_bits - 1)));

  // Every word gets floor(pattern_bits / words_per_block) salts; the first remainder
  // words get one more. Salt ranges depend only on the word index, so all layouts
  // generate exactly the same fingerprint.
  static constexpr int __bits_per_word = pattern_bits / words_per_block;
  static constexpr int __extra_words   = pattern_bits % words_per_block;

  hasher __hash_;

  static_assert(::cuda::is_power_of_two(words_per_block) && words_per_block <= 32,
                "words_per_block must be a power of two and at most 32");
  static_assert(::cuda::std::is_same_v<word_type, unsigned int>
                  || ::cuda::std::is_same_v<word_type, unsigned long long>,
                "word_type must be unsigned int or unsigned long long");
  static_assert(pattern_bits >= min_pattern_bits, "pattern_bits must be at least words_per_block");
  static_assert(pattern_bits <= max_pattern_bits,
                "pattern_bits must not exceed the total number of bits in a filter block");
  static_assert(add_horizontal_layout > 0 && add_vertical_layout > 0,
                "add layout parameters must be greater than zero");
  static_assert(contains_horizontal_layout > 0 && contains_vertical_layout > 0,
                "contains layout parameters must be greater than zero");
  // Require exact tiling. With `words_per_block` a power of two, this is equivalent to requiring
  // both `add_horizontal_layout` and `add_vertical_layout` to be powers of two with product
  // <= `words_per_block`. The internal loop count uses integer division on the product; non-
  // dividing layouts would leave trailing words uninserted on add while contains still expects
  // non-zero patterns there, producing false negatives for every inserted key.
  static_assert(words_per_block % (add_horizontal_layout * add_vertical_layout) == 0,
                "add_horizontal_layout * add_vertical_layout must evenly divide words_per_block");
  static_assert(words_per_block % (contains_horizontal_layout * contains_vertical_layout) == 0,
                "contains_horizontal_layout * contains_vertical_layout must evenly divide words_per_block");

public:
  //! @brief Constructs a Bloom filter policy.
  //!
  //! @param __hash Hash function used to generate fingerprints
  _CCCL_HOST_DEVICE_API constexpr __bloom_filter_policy(hasher __hash = {})
      : __hash_{__hash}
  {}

  //! @brief Gets the hash function.
  //!
  //! @return The hash function
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr hasher hash_function() const
  {
    return __hash_;
  }

  //! @brief Splits the 64-bit hash of a key into its upper and lower 32 bits.
  //!
  //! The upper half is used for block selection (via multiply-shift); the lower half drives the
  //! per-word fingerprint pattern via salt-based multiplicative hashing.
  //!
  //! @tparam _Key Key type
  //!
  //! @param __key Key to hash
  //!
  //! @return `{upper 32 bits, lower 32 bits}` of the 64-bit hash
  template <class _Key>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::pair<::cuda::std::uint32_t, ::cuda::std::uint32_t>
  split_hash(const _Key& __key) const
  {
    // The `split_hash()` design requires a 64-bit hash split into upper 32 bits (block selection
    // via multiply-shift) and lower 32 bits (pattern generation via salt-based multiplicative
    // hashing). This is a permanent design requirement, not a temporary limitation.
    static_assert(::cuda::std::is_same_v<decltype(__hash_(__key)), ::cuda::std::uint64_t>,
                  "bloom_filter_policy requires a 64-bit hash function");
    const auto __hash_value = __hash_(__key);
    return {static_cast<::cuda::std::uint32_t>(__hash_value >> 32), static_cast<::cuda::std::uint32_t>(__hash_value)};
  }

  //! @brief Determines the filter block a key maps to via fast multiply-shift modulo.
  //!
  //! @param __upper_hash_value Upper 32 bits of the key's hash
  //! @param __num_blocks Number of blocks in the filter
  //!
  //! @return Block index in `[0, __num_blocks)`
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::uint32_t
  block_index(::cuda::std::uint32_t __upper_hash_value, ::cuda::std::size_t __num_blocks) const noexcept
  {
    _CCCL_ASSERT(__num_blocks > 0 && __num_blocks <= max_filter_blocks, "invalid number of filter blocks");
    return static_cast<::cuda::std::uint32_t>(
      (static_cast<::cuda::std::uint64_t>(__upper_hash_value) * static_cast<::cuda::std::uint64_t>(__num_blocks))
      >> 32);
  }

  //! @brief Generates the per-word fingerprint pattern for a key when the horizontal layout is 1.
  //!
  //! @tparam _LoopIndex Outer-loop iteration index when `words_per_block / _VerticalLayout > 1`
  //! @tparam _VerticalLayout Number of contiguous words this call produces
  //!
  //! @param __lower_hash_value Lower 32 bits of the key's hash
  //!
  //! @return Array of `_VerticalLayout` words
  template <int _LoopIndex, int _VerticalLayout>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::array<word_type, _VerticalLayout>
  array_pattern(::cuda::std::uint32_t __lower_hash_value) const noexcept
  {
    return __pattern_impl<_LoopIndex, _VerticalLayout>(__lower_hash_value);
  }

  //! @brief Generates the per-word fingerprint pattern for a key when the horizontal layout is > 1.
  //!
  //! @tparam _LoopIndex Outer-loop iteration index
  //! @tparam _HorizontalLayout Cooperative-group size cooperating on a single key
  //! @tparam _VerticalLayout Number of contiguous words this call produces
  //!
  //! @param __lower_hash_value Lower 32 bits of the key's hash
  //! @param __thread_index Caller's rank within the cooperative group
  //!
  //! @return Array of `_VerticalLayout` words owned by the calling thread
  template <int _LoopIndex, int _HorizontalLayout, int _VerticalLayout>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::array<word_type, _VerticalLayout>
  array_pattern(::cuda::std::uint32_t __lower_hash_value, ::cuda::std::uint32_t __thread_index) const noexcept
  {
    return __pattern_impl<_LoopIndex, _HorizontalLayout, _VerticalLayout>(__lower_hash_value, __thread_index);
  }

private:
  // Computes the bit pattern for a vertical layout of words. The term `virtual thread` refers to an
  // ordering of the vertical layouts, namely
  //   virtual_thread_index = _LoopIndex * _HorizontalLayout + thread_index,
  // where `_LoopIndex` is the index of the outermost loop in the range
  //   [0, words_per_block / (_HorizontalLayout * _VerticalLayout)).

  // Precondition: <add/contains>_horizontal_layout == 1
  template <int _LoopIndex, int _VerticalLayout>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::array<word_type, _VerticalLayout>
  __pattern_impl(::cuda::std::uint32_t __hash) const noexcept
  {
    using __pattern_array_t = ::cuda::std::array<word_type, _VerticalLayout>;

    constexpr int __num_iterations = words_per_block / _VerticalLayout;
    static_assert(_LoopIndex < __num_iterations, "the loop index cannot exceed the number of loop iterations");

    __pattern_array_t __pattern_array{};
    __set_words<_VerticalLayout * _LoopIndex, 0>(__hash, __pattern_array);
    return __pattern_array;
  }

  // Precondition: <add/contains>_horizontal_layout > 1
  template <int _LoopIndex, int _HorizontalLayout, int _VerticalLayout>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::array<word_type, _VerticalLayout>
  __pattern_impl(::cuda::std::uint32_t __hash, ::cuda::std::uint32_t __thread_index) const noexcept
  {
    using __pattern_array_t = ::cuda::std::array<word_type, _VerticalLayout>;

    constexpr int __num_iterations = words_per_block / (_HorizontalLayout * _VerticalLayout);
    static_assert(_LoopIndex < __num_iterations, "the loop index cannot exceed the number of loop iterations");

    // [__lower_bound, __upper_bound) defines the range of virtual thread indices for this loop
    // iteration.
    constexpr int __lower_bound = _LoopIndex * _HorizontalLayout;
    constexpr int __upper_bound = __lower_bound + _HorizontalLayout;

    __pattern_array_t __pattern_array{};
    if constexpr (__num_iterations == 1)
    {
      __thread_dispatch<_VerticalLayout, __lower_bound, __upper_bound>(__hash, __thread_index, __pattern_array);
    }
    else
    {
      const ::cuda::std::uint32_t __virtual_thread_index =
        static_cast<::cuda::std::uint32_t>(_LoopIndex * _HorizontalLayout) + __thread_index;
      __thread_dispatch<_VerticalLayout, __lower_bound, __upper_bound>(__hash, __virtual_thread_index, __pattern_array);
    }
    return __pattern_array;
  }

  // Dispatches a dynamic virtual thread index to a static virtual thread index by building a
  // compile-time decision tree over the range [_LowerBound, _UpperBound) for the virtual thread
  // index. This method is only used when <add/contains>_horizontal_layout > 1.
  template <int _VerticalLayout, int _LowerBound, int _UpperBound, class _PatternArrayT>
  _CCCL_HOST_DEVICE_API constexpr void __thread_dispatch(
    ::cuda::std::uint32_t __hash, ::cuda::std::uint32_t __thread_index, _PatternArrayT& __pattern_array) const noexcept
  {
    static_assert(_LowerBound < _UpperBound, "the lower bound must be less than the upper bound");

    if constexpr (_LowerBound + 1 == _UpperBound)
    {
      // Base case: __thread_index == _LowerBound
      __set_words<_LowerBound * _VerticalLayout, 0>(__hash, __pattern_array);
    }
    else
    {
      // Recursive case: __thread_index > _LowerBound
      constexpr int __mid = (_LowerBound + _UpperBound) / 2;
      if (__thread_index < static_cast<::cuda::std::uint32_t>(__mid))
      {
        __thread_dispatch<_VerticalLayout, _LowerBound, __mid>(__hash, __thread_index, __pattern_array);
      }
      else
      {
        __thread_dispatch<_VerticalLayout, __mid, _UpperBound>(__hash, __thread_index, __pattern_array);
      }
    }
  }

  template <int _WordIndex, int _ArrayIndex, ::cuda::std::size_t _Size>
  _CCCL_HOST_DEVICE_API constexpr void
  __set_words(::cuda::std::uint32_t __hash, ::cuda::std::array<word_type, _Size>& __pattern_array) const noexcept
  {
    if constexpr (_ArrayIndex < static_cast<int>(_Size))
    {
      constexpr int __salt_start = _WordIndex * __bits_per_word + ::cuda::std::min(_WordIndex, __extra_words);
      constexpr int __salt_end   = (_WordIndex + 1) * __bits_per_word + ::cuda::std::min(_WordIndex + 1, __extra_words);
      __set_bits<__salt_start, __salt_end, _ArrayIndex>(__hash, __pattern_array);
      __set_words<_WordIndex + 1, _ArrayIndex + 1>(__hash, __pattern_array);
    }
  }

  //! @brief Sets bits in the pattern array using salts starting from `_SaltIndex`.
  template <int _SaltIndex, int _SaltEndIndex, int _PatternArrayIndex, class _PatternArrayT>
  _CCCL_HOST_DEVICE_API constexpr void
  __set_bits(::cuda::std::uint32_t __hash, _PatternArrayT& __pattern_array) const noexcept
  {
    if constexpr (_SaltIndex < _SaltEndIndex)
    {
      // Select the top `__bit_index_width` bits from the salted hash to determine the bit index.
      const ::cuda::std::uint32_t __bit_index =
        (::cuda::std::get<static_cast<::cuda::std::size_t>(_SaltIndex)>(__salts) * __hash) >> (32 - __bit_index_width);

      // Set the bit in the pattern array.
      ::cuda::std::get<static_cast<::cuda::std::size_t>(_PatternArrayIndex)>(__pattern_array) |=
        static_cast<word_type>(word_type{1} << __bit_index);

      // Recurse.
      __set_bits<_SaltIndex + 1, _SaltEndIndex, _PatternArrayIndex>(__hash, __pattern_array);
    }
  }
};
} // namespace cuda::experimental::cuco::__bloom_filter_ns

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_BLOOM_FILTER_BLOOM_FILTER_POLICY_CUH
