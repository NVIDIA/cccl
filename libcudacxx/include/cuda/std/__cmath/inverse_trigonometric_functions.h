// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_INVERSE_TRIGONOMETRIC_FUNCTIONS_H
#define _LIBCUDACXX___CMATH_INVERSE_TRIGONOMETRIC_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__internal/nvfp_types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/promote.h>

#include <nv/target>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// acos

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float acos(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ACOSF)
  return _CCCL_BUILTIN_ACOSF(__x);
#else // ^^^ _CCCL_BUILTIN_ACOSF ^^^ / vvv !_CCCL_BUILTIN_ACOSF vvv
  return ::acosf(__x);
#endif // !_CCCL_BUILTIN_ACOSF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float acosf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ACOSF)
  return _CCCL_BUILTIN_ACOSF(__x);
#else // ^^^ _CCCL_BUILTIN_ACOSF ^^^ / vvv !_CCCL_BUILTIN_ACOSF vvv
  return ::acosf(__x);
#endif // !_CCCL_BUILTIN_ACOSF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double acos(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ACOS)
  return _CCCL_BUILTIN_ACOS(__x);
#else // ^^^ _CCCL_BUILTIN_ACOS ^^^ / vvv !_CCCL_BUILTIN_ACOS vvv
  return ::acos(__x);
#endif // !_CCCL_BUILTIN_ACOS
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double acos(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ACOSL)
  return _CCCL_BUILTIN_ACOSL(__x);
#  else // ^^^ _CCCL_BUILTIN_ACOSL ^^^ / vvv !_CCCL_BUILTIN_ACOSL vvv
  return ::acosl(__x);
#  endif // !_CCCL_BUILTIN_ACOSL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double acosl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ACOSL)
  return _CCCL_BUILTIN_ACOSL(__x);
#  else // ^^^ _CCCL_BUILTIN_ACOSL ^^^ / vvv !_CCCL_BUILTIN_ACOSL vvv
  return ::acosl(__x);
#  endif // !_CCCL_BUILTIN_ACOSL
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half acos(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::acosf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 acos(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::acosf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double acos(_Integer __x) noexcept
{
  return _CUDA_VSTD::acos((double) __x);
}

// asin

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float asin(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ASINF)
  return _CCCL_BUILTIN_ASINF(__x);
#else // ^^^ _CCCL_BUILTIN_ASINF ^^^ / vvv !_CCCL_BUILTIN_ASINF vvv
  return ::asinf(__x);
#endif // !_CCCL_BUILTIN_ASINF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float asinf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ASINF)
  return _CCCL_BUILTIN_ASINF(__x);
#else // ^^^ _CCCL_BUILTIN_ASINF ^^^ / vvv !_CCCL_BUILTIN_ASINF vvv
  return ::asinf(__x);
#endif // !_CCCL_BUILTIN_ASINF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double asin(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ASIN)
  return _CCCL_BUILTIN_ASIN(__x);
#else // ^^^ _CCCL_BUILTIN_ASIN ^^^ / vvv !_CCCL_BUILTIN_ASIN vvv
  return ::asin(__x);
#endif // !_CCCL_BUILTIN_ASIN
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double asin(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ASINL)
  return _CCCL_BUILTIN_ASINL(__x);
#  else // ^^^ _CCCL_BUILTIN_ASINL ^^^ / vvv !_CCCL_BUILTIN_ASINL vvv
  return ::asinl(__x);
#  endif // !_CCCL_BUILTIN_ASINL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double asinl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ASINL)
  return _CCCL_BUILTIN_ASINL(__x);
#  else // ^^^ _CCCL_BUILTIN_ASINL ^^^ / vvv !_CCCL_BUILTIN_ASINL vvv
  return ::asinl(__x);
#  endif // !_CCCL_BUILTIN_ASINL
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half asin(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::asinf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 asin(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::asinf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double asin(_Integer __x) noexcept
{
  return _CUDA_VSTD::asin((double) __x);
}

// atan

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float atan(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ATANF)
  return _CCCL_BUILTIN_ATANF(__x);
#else // ^^^ _CCCL_BUILTIN_ATANF ^^^ / vvv !_CCCL_BUILTIN_ATANF vvv
  return ::atanf(__x);
#endif // !_CCCL_BUILTIN_ATANF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float atanf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ATANF)
  return _CCCL_BUILTIN_ATANF(__x);
#else // ^^^ _CCCL_BUILTIN_ATANF ^^^ / vvv !_CCCL_BUILTIN_ATANF vvv
  return ::atanf(__x);
#endif // !_CCCL_BUILTIN_ATANF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double atan(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ATAN)
  return _CCCL_BUILTIN_ATAN(__x);
#else // ^^^ _CCCL_BUILTIN_ATAN ^^^ / vvv !_CCCL_BUILTIN_ATAN vvv
  return ::atan(__x);
#endif // !_CCCL_BUILTIN_ATAN
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double atan(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ATANL)
  return _CCCL_BUILTIN_ATANL(__x);
#  else // ^^^ _CCCL_BUILTIN_ATANL ^^^ / vvv !_CCCL_BUILTIN_ATANL vvv
  return ::atanl(__x);
#  endif // !_CCCL_BUILTIN_ATANL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double atanl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ATANL)
  return _CCCL_BUILTIN_ATANL(__x);
#  else // ^^^ _CCCL_BUILTIN_ATANL ^^^ / vvv !_CCCL_BUILTIN_ATANL vvv
  return ::atanl(__x);
#  endif // !_CCCL_BUILTIN_ATANL
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half atan(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::atanf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 atan(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::atanf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double atan(_Integer __x) noexcept
{
  return _CUDA_VSTD::atan((double) __x);
}

// atan2

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float atan2(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_ATAN2F)
  return _CCCL_BUILTIN_ATAN2F(__x, __y);
#else // ^^^ _CCCL_BUILTIN_ATAN2F ^^^ // vvv !_CCCL_BUILTIN_ATAN2F vvv
  return ::atan2f(__x, __y);
#endif // !_CCCL_BUILTIN_ATAN2F
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float atan2f(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_ATAN2F)
  return _CCCL_BUILTIN_ATAN2F(__x, __y);
#else // ^^^ _CCCL_BUILTIN_ATAN2F ^^^ // vvv !_CCCL_BUILTIN_ATAN2F vvv
  return ::atan2f(__x, __y);
#endif // !_CCCL_BUILTIN_ATAN2F
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double atan2(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_ATAN2)
  return _CCCL_BUILTIN_ATAN2(__x, __y);
#else // ^^^ _CCCL_BUILTIN_ATAN2 ^^^ // vvv !_CCCL_BUILTIN_ATAN2 vvv
  return ::atan2(__x, __y);
#endif // !_CCCL_BUILTIN_ATAN2
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double atan2(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_ATAN2L)
  return _CCCL_BUILTIN_ATAN2L(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_ATAN2L ^^^ // vvv !_CCCL_BUILTIN_ATAN2L vvv
  return ::atan2l(__x, __y);
#  endif // !_CCCL_BUILTIN_ATAN2L
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double atan2l(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_ATAN2L)
  return _CCCL_BUILTIN_ATAN2L(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_ATAN2L ^^^ // vvv !_CCCL_BUILTIN_ATAN2L vvv
  return ::atan2l(__x, __y);
#  endif // !_CCCL_BUILTIN_ATAN2L
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half atan2(__half __x, __half __y) noexcept
{
  return __float2half(_CUDA_VSTD::atan2f(__half2float(__x), __half2float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 atan2(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::atan2f(__bfloat162float(__x), __bfloat162float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1, class _A2, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1) && _CCCL_TRAIT(is_arithmetic, _A2), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __promote_t<_A1, _A2> atan2(_A1 __x, _A2 __y) noexcept
{
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(_CCCL_TRAIT(is_same, _A1, __result_type) && _CCCL_TRAIT(is_same, _A2, __result_type)), "");
  return _CUDA_VSTD::atan2((__result_type) __x, (__result_type) __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_INVERSE_TRIGONOMETRIC_FUNCTIONS_H
