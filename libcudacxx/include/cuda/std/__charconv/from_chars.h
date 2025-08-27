//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHARCONV_FROM_CHARS_H
#define _CUDA_STD___CHARCONV_FROM_CHARS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/neg.h>
#include <cuda/__cmath/uabs.h>
#include <cuda/std/__charconv/from_chars_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct __from_chars_char_to_value_result
{
  bool __valid_;
  int __value_;
};

[[nodiscard]] _CCCL_API constexpr __from_chars_char_to_value_result
__from_chars_char_to_value(char __c, int __base) noexcept
{
  if (__base <= 10)
  {
    return {'0' <= __c && __c < '0' + __base, __c - '0'};
  }
  else if ('0' <= __c && __c <= '9')
  {
    return {true, __c - '0'};
  }
  else if ('a' <= __c && __c < 'a' + __base - 10)
  {
    return {true, __c - 'a' + 10};
  }
  else
  {
    return {'A' <= __c && __c < 'A' + __base - 10, __c - 'A' + 10};
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr from_chars_result
__from_chars_int_generic(const char* __first, const char* __last, _Tp& __value, int __base) noexcept
{
  bool __overflow  = false;
  const char* __it = __first;
  for (; __it != __last; ++__it)
  {
    const auto __digit = ::cuda::std::__from_chars_char_to_value(*__it, __base);
    if (!__digit.__valid_)
    {
      break;
    }
    if (!__overflow)
    {
      const auto __new_value = static_cast<_Tp>(__value * _Tp(__base) + _Tp(__digit.__value_));
      if (__new_value < __value)
      {
        __overflow = true;
      }
      __value = __new_value;
    }
  }
  return {__it, (__overflow) ? errc::result_out_of_range : ((__it == __first) ? errc::invalid_argument : errc{})};
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr from_chars_result
from_chars(const char* __first, const char* __last, _Tp& __value, int __base = 10) noexcept
{
  _CCCL_ASSERT(__base >= 2 && __base <= 36, "base must be in the range [2, 36]");
  _CCCL_ASSERT(__first <= __last, "input range must be a valid range");

  make_unsigned_t<_Tp> __result{};
  from_chars_result __ret{};

  if constexpr (is_signed_v<_Tp>)
  {
    bool __neg = (__first < __last && *__first == '-');
    __ret      = ::cuda::std::__from_chars_int_generic(__first + __neg, __last, __result, __base);
    if (__ret.ec == errc{})
    {
      const auto __max = ::cuda::uabs((__neg) ? numeric_limits<_Tp>::min() : numeric_limits<_Tp>::max());
      if (__result > __max)
      {
        __ret.ec = errc::result_out_of_range;
      }
      if (__neg)
      {
        __result = ::cuda::neg(__result);
      }
    }
  }
  else
  {
    __ret = ::cuda::std::__from_chars_int_generic(__first, __last, __result, __base);
  }

  if (__ret.ec == errc{})
  {
    __value = static_cast<_Tp>(__result);
  }
  else if (__ret.ec == errc::invalid_argument)
  {
    __ret.ptr = __first;
  }
  return __ret;
}

[[nodiscard]] _CCCL_API constexpr from_chars_result
from_chars(const char* __first, const char* __last, char& __value, int __base = 10) noexcept
{
  using _Tp = conditional_t<is_signed_v<char>, signed char, unsigned char>;
  _Tp __value_tmp{};
  const auto __ret = ::cuda::std::from_chars(__first, __last, __value_tmp, __base);
  if (__ret.ec == errc{})
  {
    __value = static_cast<char>(__value_tmp);
  }
  return __ret;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHARCONV_FROM_CHARS_H
