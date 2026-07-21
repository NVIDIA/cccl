//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHARCONV_TO_CHARS_H
#define _CUDA_STD___CHARCONV_TO_CHARS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/ilog.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/__cmath/uabs.h>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__charconv/chars_format.h>
#include <cuda/std/__charconv/to_chars_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

[[nodiscard]] _CCCL_API constexpr char __to_chars_value_to_char(int __v, int __base) noexcept
{
  _CCCL_ASSERT(__v >= 0 && __v < __base, "value must be in the range [0, base)");
  const int __offset = (__base < 10 || __v < 10) ? '0' : ('a' - 10);
  return static_cast<char>(__offset + __v);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr int __to_chars_int_width(_Tp __v, int __base) noexcept
{
  using _Up = ::cuda::std::conditional_t<sizeof(_Tp) >= sizeof(uint32_t), make_unsigned_t<_Tp>, uint32_t>;

  auto __uv = static_cast<_Up>(__v);

  const auto __ubase   = static_cast<_Up>(__base);
  const auto __ubase_2 = __ubase * __ubase;
  const auto __ubase_3 = __ubase_2 * __ubase;
  const auto __ubase_4 = __ubase_2 * __ubase_2;

  int __r = 0;
  while (true)
  {
    if (__uv < __ubase)
    {
      __r += 1;
      break;
    }
    else if (__uv < __ubase_2)
    {
      __r += 2;
      break;
    }
    else if (__uv < __ubase_3)
    {
      __r += 3;
      break;
    }
    else if (__uv < __ubase_4)
    {
      __r += 4;
      break;
    }

    __uv /= __ubase_4;
    __r += 4;
  }
  return __r;
}

template <int _Base, class _Tp>
[[nodiscard]] _CCCL_API constexpr int __to_chars_int_width(_Tp __v) noexcept
{
  if constexpr (::cuda::is_power_of_two(_Base))
  {
    // For bases that are powers of 2, we can count leading zeros to compute the width more efficiently.
    constexpr auto __base_ilog2 = ::cuda::ilog2(_Base);

    // If value == 0 still need one digit, so we always set the least significant bit.
    return ::cuda::ceil_div(__num_bits_v<_Tp> - ::cuda::std::countl_zero(static_cast<_Tp>(__v | 1)), __base_ilog2);
  }
  else if constexpr (_Base == 10)
  {
    return (__v > 1) ? ::cuda::ceil_ilog10(__v) : 1;
  }
  else
  {
    return ::cuda::std::__to_chars_int_width(__v, _Base);
  }
}

template <class _Tp>
_CCCL_API constexpr void __to_chars_int_generic(char* __last, _Tp __value, int __base) noexcept
{
  do
  {
    const int __c = __value % __base;
    *--__last     = ::cuda::std::__to_chars_value_to_char(__c, __base);
    __value /= static_cast<_Tp>(__base);
  } while (__value != 0);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr to_chars_result
to_chars(char* __first, char* __last, _Tp __value, int __base = 10) noexcept
{
  _CCCL_ASSERT(__base >= 2 && __base <= 36, "base must be in the range [2, 36]");
  _CCCL_ASSERT(__first <= __last, "output range must be a valid range");

  if constexpr (is_signed_v<_Tp>)
  {
    if (__value < _Tp{0} && __first < __last)
    {
      *__first++ = '-';
    }
    return ::cuda::std::to_chars(__first, __last, ::cuda::uabs(__value), __base);
  }
  else
  {
    const ptrdiff_t __cap = __last - __first;

    int __n{};
    switch (__base)
    {
      case 2:
        __n = ::cuda::std::__to_chars_int_width<2>(__value);
        break;
      case 4:
        __n = ::cuda::std::__to_chars_int_width<4>(__value);
        break;
      case 8:
        __n = ::cuda::std::__to_chars_int_width<8>(__value);
        break;
      case 10:
        __n = ::cuda::std::__to_chars_int_width<10>(__value);
        break;
      case 16:
        __n = ::cuda::std::__to_chars_int_width<16>(__value);
        break;
      case 32:
        __n = ::cuda::std::__to_chars_int_width<32>(__value);
        break;
      default:
        __n = ::cuda::std::__to_chars_int_width(__value, __base);
        break;
    }

    if (__n > __cap)
    {
      return {__last, errc::value_too_large};
    }

    char* __new_last = __first + __n;

    ::cuda::std::__to_chars_int_generic(__new_last, __value, __base);

    return {__new_last, errc{}};
  }
}

[[nodiscard]] _CCCL_API constexpr to_chars_result
to_chars(char* __first, char* __last, char __value, int __base = 10) noexcept
{
  if constexpr (is_signed_v<char>)
  {
    return ::cuda::std::to_chars(__first, __last, static_cast<signed char>(__value), __base);
  }
  else
  {
    return ::cuda::std::to_chars(__first, __last, static_cast<unsigned char>(__value), __base);
  }
}

_CCCL_API constexpr to_chars_result to_chars(char*, char*, bool, int = 10) noexcept = delete;

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_floating_point_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr to_chars_result to_chars(char* __first, char* __last, _Tp __value) noexcept
{
  static_assert(::cuda::std::__always_false_v<_Tp>,
                "cuda::std::to_chars for floating point types is not yet implemented");
  (void) __first;
  (void) __last;
  (void) __value;
  return {};
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_floating_point_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr to_chars_result
to_chars(char* __first, char* __last, _Tp __value, chars_format __fmt) noexcept
{
  static_assert(::cuda::std::__always_false_v<_Tp>,
                "cuda::std::to_chars for floating point types is not yet implemented");
  (void) __first;
  (void) __last;
  (void) __value;
  (void) __fmt;
  return {};
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_floating_point_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr to_chars_result
to_chars(char* __first, char* __last, _Tp __value, chars_format __fmt, int __prec) noexcept
{
  static_assert(::cuda::std::__always_false_v<_Tp>,
                "cuda::std::to_chars for floating point types is not yet implemented");
  (void) __first;
  (void) __last;
  (void) __value;
  (void) __fmt;
  (void) __prec;
  return {};
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHARCONV_TO_CHARS_H
