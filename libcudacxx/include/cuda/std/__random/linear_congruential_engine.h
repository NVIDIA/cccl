//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANDOM_LINEAR_CONGRUENTIAL_ENGINE_H
#define _CUDA_STD___RANDOM_LINEAR_CONGRUENTIAL_ENGINE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/ios.h>
#include <cuda/std/__random/is_seed_sequence.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/climits>
#include <cuda/std/cstdint>

#if !_CCCL_COMPILER(NVRTC)
#  include <ios>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <uint64_t __A,
          uint64_t __C,
          uint64_t __M,
          uint64_t _Mp,
          bool _MightOverflow = (__A != 0 && __M != 0 && __M - 1 > (_Mp - __C) / __A),
          bool _OverflowOk    = ((__M | (__M - 1)) > __M), // m = 2^n
          bool _SchrageOk     = (__A != 0 && __M != 0 && __M % __A <= __M / __A)> // r <= q
struct __lce_alg_picker
{
  static_assert(__A != 0 || __M != 0 || !_MightOverflow || _OverflowOk || _SchrageOk,
                "The current values of a, c, and m cannot generate a number "
                "within bounds of linear_congruential_engine.");

  static constexpr const bool __use_schrage = _MightOverflow && !_OverflowOk && _SchrageOk;
};

template <uint64_t __A,
          uint64_t __C,
          uint64_t __M,
          uint64_t _Mp,
          bool _UseSchrage = __lce_alg_picker<__A, __C, __M, _Mp>::__use_schrage>
struct __lce_ta;

// 64

template <uint64_t __A, uint64_t __C, uint64_t __M>
struct __lce_ta<__A, __C, __M, ~uint64_t{0}, true>
{
  using result_type = uint64_t;
  [[nodiscard]] _CCCL_API static constexpr result_type next(result_type __x) noexcept
  {
    // Schrage's algorithm
    constexpr result_type __q = __M / __A;
    constexpr result_type __r = __M % __A;
    const result_type __t0    = __A * (__x % __q);
    const result_type __t1    = __r * (__x / __q);
    __x                       = __t0 + (__t0 < __t1) * __M - __t1;
    __x += __C - (__x >= __M - __C) * __M;
    return __x;
  }
};

template <uint64_t __A, uint64_t __M>
struct __lce_ta<__A, 0, __M, ~uint64_t{0}, true>
{
  using result_type = uint64_t;
  [[nodiscard]] _CCCL_API static constexpr result_type next(result_type __x) noexcept
  {
    // Schrage's algorithm
    constexpr result_type __q = __M / __A;
    constexpr result_type __r = __M % __A;
    const result_type __t0    = __A * (__x % __q);
    const result_type __t1    = __r * (__x / __q);
    __x                       = __t0 + (__t0 < __t1) * __M - __t1;
    return __x;
  }
};

template <uint64_t __A, uint64_t __C, uint64_t __M>
struct __lce_ta<__A, __C, __M, ~uint64_t{0}, false>
{
  using result_type = uint64_t;
  [[nodiscard]] _CCCL_API static constexpr result_type next(result_type __x) noexcept
  {
    return (__A * __x + __C) % __M;
  }
};

template <uint64_t __A, uint64_t __C>
struct __lce_ta<__A, __C, 0, ~uint64_t{0}, false>
{
  using result_type = uint64_t;
  [[nodiscard]] _CCCL_API static constexpr result_type next(result_type __x) noexcept
  {
    return __A * __x + __C;
  }
};

// 32

template <uint64_t _Ap, uint64_t _Cp, uint64_t _Mp>
struct __lce_ta<_Ap, _Cp, _Mp, ~uint32_t{0}, true>
{
  using result_type = uint32_t;
  [[nodiscard]] _CCCL_API static constexpr result_type next(result_type __x) noexcept
  {
    constexpr auto __A = static_cast<result_type>(_Ap);
    constexpr auto __C = static_cast<result_type>(_Cp);
    constexpr auto __M = static_cast<result_type>(_Mp);
    // Schrage's algorithm
    constexpr result_type __q = __M / __A;
    constexpr result_type __r = __M % __A;
    const result_type __t0    = __A * (__x % __q);
    const result_type __t1    = __r * (__x / __q);
    __x                       = __t0 + (__t0 < __t1) * __M - __t1;
    __x += __C - (__x >= __M - __C) * __M;
    return __x;
  }
};

template <uint64_t _Ap, uint64_t _Mp>
struct __lce_ta<_Ap, 0, _Mp, ~uint32_t{0}, true>
{
  using result_type = uint32_t;
  [[nodiscard]] _CCCL_API static constexpr result_type next(result_type __x) noexcept
  {
    constexpr result_type __A = static_cast<result_type>(_Ap);
    constexpr result_type __M = static_cast<result_type>(_Mp);
    // Schrage's algorithm
    constexpr result_type __q = __M / __A;
    constexpr result_type __r = __M % __A;
    const result_type __t0    = __A * (__x % __q);
    const result_type __t1    = __r * (__x / __q);
    __x                       = __t0 + (__t0 < __t1) * __M - __t1;
    return __x;
  }
};

template <uint64_t _Ap, uint64_t _Cp, uint64_t _Mp>
struct __lce_ta<_Ap, _Cp, _Mp, ~uint32_t{0}, false>
{
  using result_type = uint32_t;
  [[nodiscard]] _CCCL_API static constexpr result_type next(result_type __x) noexcept
  {
    constexpr result_type __A = static_cast<result_type>(_Ap);
    constexpr result_type __C = static_cast<result_type>(_Cp);
    constexpr result_type __M = static_cast<result_type>(_Mp);
    return (__A * __x + __C) % __M;
  }
};

template <uint64_t _Ap, uint64_t _Cp>
struct __lce_ta<_Ap, _Cp, 0, ~uint32_t{0}, false>
{
  using result_type = uint32_t;
  [[nodiscard]] _CCCL_API static constexpr result_type next(result_type __x) noexcept
  {
    constexpr result_type __A = static_cast<result_type>(_Ap);
    constexpr result_type __C = static_cast<result_type>(_Cp);
    return __A * __x + __C;
  }
};

// 16

template <uint64_t __A, uint64_t __C, uint64_t __M, bool __b>
struct __lce_ta<__A, __C, __M, static_cast<uint16_t>(~0), __b>
{
  using result_type = uint16_t;
  [[nodiscard]] _CCCL_API static constexpr result_type next(result_type __x) noexcept
  {
    return static_cast<result_type>(__lce_ta<__A, __C, __M, ~uint32_t{0}>::next(__x));
  }
};

template <class _UIntType, _UIntType __A, _UIntType __C, _UIntType __M>
class _CCCL_TYPE_VISIBILITY_DEFAULT linear_congruential_engine;

template <class _UIntType, _UIntType __A, _UIntType __C, _UIntType __M>
class _CCCL_TYPE_VISIBILITY_DEFAULT linear_congruential_engine
{
public:
  // types
  using result_type = _UIntType;

private:
  result_type __x_{};

  static constexpr const result_type _Mp = result_type(~0);

  static_assert(__M == 0 || __A < __M, "linear_congruential_engine invalid parameters");
  static_assert(__M == 0 || __C < __M, "linear_congruential_engine invalid parameters");
  static_assert(is_unsigned_v<_UIntType>, "_UIntType must be uint32_t type");

public:
  static constexpr const result_type _Min = __C == 0u ? 1u : 0u;
  static constexpr const result_type _Max = __M - _UIntType(1u);
  static_assert(_Min < _Max, "linear_congruential_engine invalid parameters");

  // engine characteristics
  static constexpr const result_type multiplier = __A;
  static constexpr const result_type increment  = __C;
  static constexpr const result_type modulus    = __M;
  [[nodiscard]] _CCCL_API static constexpr result_type min() noexcept
  {
    return _Min;
  }
  [[nodiscard]] _CCCL_API static constexpr result_type max() noexcept
  {
    return _Max;
  }
  static constexpr const result_type default_seed = 1u;

  // constructors and seeding functions
  _CCCL_API constexpr linear_congruential_engine() noexcept
      : linear_congruential_engine(default_seed)
  {}
  _CCCL_API explicit constexpr linear_congruential_engine(result_type __s) noexcept
  {
    seed(__s);
  }

  template <class _Sseq, enable_if_t<__is_seed_sequence<_Sseq, linear_congruential_engine>, int> = 0>
  _CCCL_API explicit constexpr linear_congruential_engine(_Sseq& __q) noexcept
  {
    seed(__q);
  }
  _CCCL_API constexpr void seed(result_type __s = default_seed) noexcept
  {
    seed(integral_constant<bool, __M == 0>(), integral_constant<bool, __C == 0>(), __s);
  }
  template <class _Sseq, enable_if_t<__is_seed_sequence<_Sseq, linear_congruential_engine>, int> = 0>
  _CCCL_API constexpr void seed(_Sseq& __q) noexcept
  {
    __seed(__q,
           integral_constant<uint32_t,
                             1 + (__M == 0 ? (sizeof(result_type) * CHAR_BIT - 1) / 32 : (__M > 0x100000000ull))>());
  }

  // generating functions
  _CCCL_API constexpr result_type operator()() noexcept
  {
    return __x_ = static_cast<result_type>(__lce_ta<__A, __C, __M, _Mp>::next(__x_));
  }

  _CCCL_API constexpr void discard(uint64_t __z) noexcept
  {
    constexpr bool __can_overflow = (__A != 0 && __M != 0 && __M - 1 > (_Mp - __C) / __A);
    // Fallback implementation
    if constexpr (__can_overflow)
    {
      for (; __z; --__z)
      {
        (void) operator()();
      }
    }
    else
    {
      uint64_t __acc_mult                  = 1;
      [[maybe_unused]] uint64_t __acc_plus = 0;
      uint64_t __cur_mult                  = multiplier;
      [[maybe_unused]] uint64_t __cur_plus = increment;
      while (__z > 0)
      {
        if (__z & 1)
        {
          __acc_mult = (__acc_mult * __cur_mult) % modulus;
          if constexpr (increment != 0)
          {
            __acc_plus = (__acc_plus * __cur_mult + __cur_plus) % modulus;
          }
        }
        if constexpr (increment != 0)
        {
          __cur_plus = ((__cur_mult + 1) * __cur_plus) % modulus;
        }
        __cur_mult = (__cur_mult * __cur_mult) % modulus;
        __z >>= 1;
      }
      __x_ = (__acc_mult * __x_ + __acc_plus) % modulus;
    }
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const linear_congruential_engine& __x, const linear_congruential_engine& __y) noexcept
  {
    return __x.__x_ == __y.__x_;
  }
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const linear_congruential_engine& __x, const linear_congruential_engine& __y) noexcept
  {
    return !(__x == __y);
  }

#if !_CCCL_COMPILER(NVRTC)
  template <typename _CharT, typename _Traits>
  _CCCL_API friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const linear_congruential_engine& __e)
  {
    using _Ostream                            = ::std::basic_ostream<_CharT, _Traits>;
    const typename _Ostream::fmtflags __flags = __os.flags();
    __os.flags(_Ostream::dec | _Ostream::left);
    __os.fill(__os.widen(' '));
    __os.flags(__flags);
    return __os << __e.__x_;
  }
  template <typename _CharT, typename _Traits>
  _CCCL_API friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, linear_congruential_engine& __e)
  {
    using _Istream                            = ::std::basic_istream<_CharT, _Traits>;
    const typename _Istream::fmtflags __flags = __is.flags();
    __is.flags(_Istream::dec | _Istream::skipws);
    _UIntType __t;
    __is >> __t;
    if (!__is.fail())
    {
      __e.__x_ = __t;
    }
    __is.flags(__flags);
    return __is;
  }
#endif // !_CCCL_COMPILER(NVRTC)

private:
  _CCCL_API constexpr void seed(true_type, true_type, result_type __s) noexcept
  {
    __x_ = __s == 0 ? 1 : __s;
  }
  _CCCL_API constexpr void seed(true_type, false_type, result_type __s) noexcept
  {
    __x_ = __s;
  }
  _CCCL_API constexpr void seed(false_type, true_type, result_type __s) noexcept
  {
    __x_ = __s % __M == 0 ? 1 : __s % __M;
  }
  _CCCL_API constexpr void seed(false_type, false_type, result_type __s) noexcept
  {
    __x_ = __s % __M;
  }

  template <class _Sseq>
  _CCCL_API constexpr void __seed(_Sseq& __q, integral_constant<uint32_t, 1>) noexcept;
  template <class _Sseq>
  _CCCL_API constexpr void __seed(_Sseq& __q, integral_constant<uint32_t, 2>) noexcept;
};

template <class _UIntType, _UIntType __A, _UIntType __C, _UIntType __M>
template <class _Sseq>
_CCCL_API constexpr void
linear_congruential_engine<_UIntType, __A, __C, __M>::__seed(_Sseq& __q, integral_constant<uint32_t, 1>) noexcept
{
  constexpr uint32_t __k = 1;
  uint32_t __ar[__k + 3] = {};
  __q.generate(__ar, __ar + __k + 3);
  result_type __s = static_cast<result_type>(__ar[3] % __M);
  __x_            = __C == 0 && __s == 0 ? result_type(1) : __s;
}

template <class _UIntType, _UIntType __A, _UIntType __C, _UIntType __M>
template <class _Sseq>
_CCCL_API constexpr void
linear_congruential_engine<_UIntType, __A, __C, __M>::__seed(_Sseq& __q, integral_constant<uint32_t, 2>) noexcept
{
  constexpr uint32_t __k = 2;
  uint32_t __ar[__k + 3] = {};
  __q.generate(__ar, __ar + __k + 3);
  result_type __s = static_cast<result_type>((__ar[3] + ((uint64_t) __ar[4] << 32)) % __M);
  __x_            = __C == 0 && __s == 0 ? result_type(1) : __s;
}

using minstd_rand0 = linear_congruential_engine<uint_fast32_t, 16807, 0, 2147483647>;
using minstd_rand  = linear_congruential_engine<uint_fast32_t, 48271, 0, 2147483647>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_LINEAR_CONGRUENTIAL_ENGINE_H
