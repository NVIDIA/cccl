//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANDOM_SEED_SEQ_H
#define _CUDA_STD___RANDOM_SEED_SEQ_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/copy_n.h>
#include <cuda/std/__algorithm/fill.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

/// @class seed_seq
///
/// Generate unbiased seeds by filling the output range [begin, end) with 32-bit
/// unsigned integer values, based on the (possibly biased) seeds stored in the
/// internal buffer. If `begin == end`, the function does nothing. Otherwise,
/// the values are produced by the standard seed sequence mixing algorithm.
///
/// If `std::iterator_traits<RandomIt>::value_type` is not an unsigned integer
/// type, or its width is less than 32 bits, the program is ill-formed.
/// If `RandomIt` does not meet the requirements of a LegacyRandomAccessIterator
/// or is not mutable, the behavior is undefined.
class _CCCL_TYPE_VISIBILITY_DEFAULT seed_seq
{
public:
  using result_type = uint_least32_t;

  /// @brief Default-construct an empty seed sequence.
  ///
  /// `size()` will return 0.
  _CCCL_HIDE_FROM_ABI constexpr seed_seq() noexcept = default;

  /// @brief Construct from an iterator range of seed values.
  /// @tparam _InputIt Input iterator type yielding values convertible to `result_type`.
  /// @param __begin Iterator pointing to the first seed value.
  /// @param __end Iterator one-past-the-last seed value.
  template <class _InputIt>
  _CCCL_API _CCCL_CONSTEXPR_CXX20 seed_seq(_InputIt __begin, _InputIt __end)
  {
    auto __n = ::cuda::std::distance(__begin, __end);
    if (__n <= 0)
    {
      return;
    }
    __data_ = new result_type[__n]{};
    __size_ = static_cast<size_t>(__n);
    _CCCL_TRY
    {
      for (size_t __i = 0; __i < __size_; ++__i, ++__begin)
      {
        __data_[__i] = static_cast<uint32_t>(*__begin);
      }
    }
    _CCCL_CATCH_ALL
    {
      delete[] __data_;
      _CCCL_RETHROW;
    }
  }

  /// @brief Construct from an initializer list of seed values.
  /// @tparam _InitT Element type convertible to `result_type`.
  /// @param __il The list of seed values.
  _CCCL_TEMPLATE(typename _InitT)
  _CCCL_REQUIRES(__cccl_is_integer_v<_InitT>)
  _CCCL_API _CCCL_CONSTEXPR_CXX20 seed_seq(initializer_list<_InitT> __il)
      : seed_seq(__il.begin(), __il.end())
  {}

  /// @brief seed_seq is not copyable.
  seed_seq(const seed_seq&) = delete;

  _CCCL_API _CCCL_CONSTEXPR_CXX20 ~seed_seq()
  {
    delete[] __data_;
  }

  /// @brief seed_seq is not copy-assignable.
  seed_seq& operator=(const seed_seq&) = delete;

  /// @brief Returns the number of seed values stored.
  /// @return Number of internal seed values (may be 0).
  [[nodiscard]] _CCCL_API constexpr size_t size() const noexcept
  {
    return __size_;
  }

  /// @brief Copy stored seed values into the output iterator `__dest`.
  /// @tparam _OutputIt Output iterator type accepting `result_type` values.
  /// @param __dest Destination iterator where stored values will be written.
  template <class _OutputIt>
  _CCCL_API constexpr void param(_OutputIt __dest) const
  {
    ::cuda::std::copy_n(__data_, __size_, __dest);
  }

  /// @brief Generate unbiased seeds by filling the output range [begin, end) with 32-bit unsigned integer values, based
  /// on the (possibly biased) seeds stored in v. The generation algorithm is adapted from the initialization sequence
  /// of the Mersenne Twister generator by Makoto Matsumoto and Takuji Nishimura, incorporating the improvements made by
  /// Mutsuo Saito in 2007.
  /// @tparam _RandomIt Random-access iterator to writable storage for `result_type`.
  /// @param __begin Iterator to the beginning of the destination range.
  /// @param __end Iterator one-past-the-end of the destination range.
  template <class _RandomIt>
  _CCCL_API constexpr void generate(_RandomIt __begin, _RandomIt __end) const
  {
    if (__begin == __end)
    {
      return;
    }
    // https://en.cppreference.com/w/cpp/numeric/random/seed_seq/generate.html
    const result_type __z = static_cast<result_type>(__size_);
    const result_type __n = static_cast<result_type>(::cuda::std::distance(__begin, __end));
    const result_type __m = ::cuda::std::max(__z + 1, __n);
    const result_type __t = (__n >= 623) ? 11 : (__n >= 68) ? 7 : (__n >= 39) ? 5 : (__n >= 7) ? 3 : (__n - 1) / 2;
    const result_type __p = (__n - __t) / 2;
    const result_type __q = __p + __t;

    // 1.
    ::cuda::std::fill(__begin, __end, result_type{0x8b8b8b8b});

    // 2.
    for (size_t __k = 0; __k < __m; ++__k)
    {
      result_type __k_mod_n   = __k % __n;
      result_type __k_p_mod_n = (__k + __p) % __n;
      result_type __k_q_mod_n = (__k + __q) % __n;
      // 2.1
      const result_type __r1 =
        1664525 * __generate_T(__begin[__k_mod_n] ^ __begin[__k_p_mod_n] ^ __begin[(__k - 1) % __n]);
      // 2.2
      result_type __r2 = (__k == 0)   ? __r1 + __z
                       : (__k <= __z) ? __r1 + __k_mod_n + __data_[__k - 1]
                                      : __r1 + __k_mod_n;
      // 2.3
      __begin[__k_p_mod_n] += __r1;
      __begin[__k_p_mod_n] &= 0xFFFFFFFF;
      __begin[__k_q_mod_n] += __r2;
      __begin[__k_q_mod_n] &= 0xFFFFFFFF;
      __begin[__k_mod_n] = __r2 & 0xFFFFFFFF;
    }

    // 3.
    for (size_t __k = __m; __k < __m + __n; ++__k)
    {
      result_type __k_mod_n   = __k % __n;
      result_type __k_p_mod_n = (__k + __p) % __n;
      result_type __k_q_mod_n = (__k + __q) % __n;
      // 3.1
      const result_type __r3 =
        1566083941 * __generate_T(__begin[__k_mod_n] + __begin[__k_p_mod_n] + __begin[(__k - 1) % __n]);
      // 3.2
      const result_type __r4 = __r3 - __k_mod_n;
      // 3.3
      __begin[__k_p_mod_n] ^= __r3;
      __begin[__k_p_mod_n] &= 0xFFFFFFFF;
      // 3.4
      __begin[__k_q_mod_n] ^= __r4;
      __begin[__k_q_mod_n] &= 0xFFFFFFFF;
      // 3.5
      __begin[__k_mod_n] = __r4 & 0xFFFFFFFF;
    }
  }

private:
  [[nodiscard]] _CCCL_API static constexpr result_type __generate_T(result_type __x) noexcept
  {
    return (__x ^ (__x >> 27));
  }
  result_type* __data_ = nullptr;
  size_t __size_       = 0;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_SEED_SEQ_H
