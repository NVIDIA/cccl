//===----------------------------------------------------------------------===//
//
// Part of libcu++, the _Common++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___CMATH_IS_OVERFLOW_H
#define _CUDA___CMATH_IS_OVERFLOW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

_CCCL_TEMPLATE(class _Ap, class _Bp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Ap) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Bp))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_add_overflow([[maybe_unused]] const _Ap __a, //
                                                                       [[maybe_unused]] const _Bp __b) noexcept
{
  using _Common = _CUDA_VSTD::common_type_t<_Ap, _Bp>;
  if constexpr (sizeof(_Common) > sizeof(_Ap) && sizeof(_Common) > sizeof(_Bp)) // promotion -> fast path
  {
    return false;
  }
  else
  {
    constexpr auto __max_v = _CUDA_VSTD::numeric_limits<_Common>::max();
    constexpr auto __min_v = _CUDA_VSTD::numeric_limits<_Common>::min();
    auto __a1              = static_cast<_Common>(__a);
    auto __b1              = static_cast<_Common>(__b);
    if (_CUDA_VSTD::is_unsigned_v<_Common> || _CUDA_VSTD::is_unsigned_v<_Bp> || __b1 >= 0) // check for overflow
    { // a + b > max
      return __a1 > __max_v - __b1;
    }
    // b < 0 --> check for underflow, a + b < min
    return _CUDA_VSTD::is_unsigned_v<_Ap> ? false : __a1 < __min_v - __b1;
  }
}

_CCCL_TEMPLATE(class _Ap, class _Bp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Ap) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Bp))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_sub_overflow([[maybe_unused]] const _Ap __a, //
                                                                       [[maybe_unused]] const _Bp __b) noexcept
{
  // we cannot call is_add_overflow(a, -b) because b could be equal to min
  using _Common = _CUDA_VSTD::common_type_t<_Ap, _Bp>;
  if constexpr (sizeof(_Common) > sizeof(_Ap) && sizeof(_Common) > sizeof(_Bp)) // promotion -> fast path
  {
    return false;
  }
  else
  {
    constexpr auto __max_v = _CUDA_VSTD::numeric_limits<_Common>::max();
    constexpr auto __min_v = _CUDA_VSTD::numeric_limits<_Common>::min();
    auto __a1              = static_cast<_Common>(__a);
    auto __b1              = static_cast<_Common>(__b);
    if (_CUDA_VSTD::is_unsigned_v<_Common> || _CUDA_VSTD::is_unsigned_v<_Bp> || __b1 >= 0) // check for underflow
    { // a - b < min
      return __a1 < __min_v + __b1;
    }
    // b < 0 --> check for overflow, a - b > max
    return __a1 > __max_v + __b1;
    // we cannot use is_unsigned_v<_Ap> here because b could be equal to min
  }
}

_CCCL_TEMPLATE(class _Ap, class _Bp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Ap) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Bp))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_mul_overflow([[maybe_unused]] const _Ap __a, //
                                                                       [[maybe_unused]] const _Bp __b) noexcept
{
  using _Common = _CUDA_VSTD::common_type_t<_Ap, _Bp>;
  if constexpr (sizeof(_Common) > sizeof(_Ap) && sizeof(_Common) > sizeof(_Bp)) // promotion -> fast path
  {
    return false;
  }
  else
  {
    constexpr auto __max_v = _CUDA_VSTD::numeric_limits<_Common>::max();
    auto __a1              = static_cast<_Common>(__a);
    auto __b1              = static_cast<_Common>(__b);
    if (__b1 == 0)
    {
      return false;
    }
    if constexpr (_CUDA_VSTD::is_unsigned_v<_Common>
                  || (_CUDA_VSTD::is_unsigned_v<_Ap> && _CUDA_VSTD::is_unsigned_v<_Bp>) )
    {
      return __a1 > __max_v / __b1;
    }
    else // signed
    {
      constexpr auto __min_v = _CUDA_VSTD::numeric_limits<_Common>::min();
      auto __a_ge_zero       = _CUDA_VSTD::is_unsigned_v<_Ap> || __a1 >= 0;
      auto __b_ge_zero       = _CUDA_VSTD::is_unsigned_v<_Bp> || __b1 >= 0;
      // a >= 0 && b >= 0 -> a > max / b     -
      // a < 0  && b < 0  -> a < max / b +++                  need to handle  b == min
      // a < 0  && b >= 0 -> a < min / b
      // a >= 0 && b < 0  -> a > min / b *** --> b < min / a  to avoid min / -1 overflow, a != 0
      if (__a_ge_zero)
      {
        return (__b_ge_zero) ? (__a1 > __max_v / __b1) // a >= 0 && b >= 0
                             : (__a1 == 0 ? false : __b1 < __min_v / __a1); // a >= 0 && b < 0
      }
      // a < 0
      return __b1 >= 0 ? (__a1 < __min_v / __b1) // a < 0 && b >= 0
                       : (__b1 == __min_v ? true : __a1 < __max_v / __b1); // a < 0 && b < 0
    }
  }
}

_CCCL_TEMPLATE(class _Ap, class _Bp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Ap) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Bp))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_div_overflow([[maybe_unused]] const _Ap __a, //
                                                                       [[maybe_unused]] const _Bp __b) noexcept
{
  using _Common = _CUDA_VSTD::common_type_t<_Ap, _Bp>;
  if (__b == 0)
  {
    return true;
  }
  if constexpr (sizeof(_Common) > sizeof(_Ap) || _CUDA_VSTD::is_unsigned_v<_Ap> || _CUDA_VSTD::is_unsigned_v<_Bp>)
  {
    return false;
  }
  else
  {
    auto __a1              = static_cast<_Common>(__a);
    auto __b1              = static_cast<_Common>(__b);
    constexpr auto __min_v = _CUDA_VSTD::numeric_limits<_Common>::min();
    return (__a1 == __min_v && __b1 == _Common{-1});
  }
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___CMATH_IS_OVERFLOW_H
