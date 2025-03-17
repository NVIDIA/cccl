// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_ROOTS_H
#define _LIBCUDACXX___CMATH_ROOTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_integral.h>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// sqrt

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float sqrt(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_SQRTF)
  return _CCCL_BUILTIN_SQRTF(__x);
#else // ^^^ _CCCL_BUILTIN_SQRTF ^^^ // vvv !_CCCL_BUILTIN_SQRTF vvv
  return ::sqrtf(__x);
#endif // !_CCCL_BUILTIN_SQRTF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float sqrtf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_SQRTF)
  return _CCCL_BUILTIN_SQRTF(__x);
#else // ^^^ _CCCL_BUILTIN_SQRTF ^^^ // vvv !_CCCL_BUILTIN_SQRTF vvv
  return ::sqrtf(__x);
#endif // !_CCCL_BUILTIN_SQRTF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double sqrt(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_SQRT)
  return _CCCL_BUILTIN_SQRT(__x);
#else // ^^^ _CCCL_BUILTIN_SQRT ^^^ // vvv !_CCCL_BUILTIN_SQRT vvv
  return ::sqrt(__x);
#endif // !_CCCL_BUILTIN_SQRT
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double sqrt(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_SQRTL)
  return _CCCL_BUILTIN_SQRTL(__x);
#  else // ^^^ _CCCL_BUILTIN_SQRTL ^^^ // vvv !_CCCL_BUILTIN_SQRTL vvv
  return ::sqrtl(__x);
#  endif // !_CCCL_BUILTIN_SQRTL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double sqrtl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_SQRTL)
  return _CCCL_BUILTIN_SQRTL(__x);
#  else // ^^^ _CCCL_BUILTIN_SQRTL ^^^ // vvv !_CCCL_BUILTIN_SQRTL vvv
  return ::sqrtl(__x);
#  endif // !_CCCL_BUILTIN_SQRTL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half sqrt(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hsqrt(__x);), (return __float2half(_CUDA_VSTD::sqrt(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 sqrt(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hsqrt(__x);), (return __float2bfloat16(_CUDA_VSTD::sqrt(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double sqrt(_Integer __x) noexcept
{
  return _CUDA_VSTD::sqrt((double) __x);
}

// cbrt

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float cbrt(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_CBRTF)
  return _CCCL_BUILTIN_CBRTF(__x);
#else // ^^^ _CCCL_BUILTIN_CBRTF ^^^ // vvv !_CCCL_BUILTIN_CBRTF vvv
  return ::cbrtf(__x);
#endif // !_CCCL_BUILTIN_CBRTF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float cbrtf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_CBRTF)
  return _CCCL_BUILTIN_CBRTF(__x);
#else // ^^^ _CCCL_BUILTIN_CBRTF ^^^ // vvv !_CCCL_BUILTIN_CBRTF vvv
  return ::cbrtf(__x);
#endif // !_CCCL_BUILTIN_CBRTF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double cbrt(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_CBRT)
  return _CCCL_BUILTIN_CBRT(__x);
#else // ^^^ _CCCL_BUILTIN_CBRT ^^^ // vvv !_CCCL_BUILTIN_CBRT vvv
  return ::cbrt(__x);
#endif // !_CCCL_BUILTIN_CBRT
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double cbrt(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_CBRTL)
  return _CCCL_BUILTIN_CBRTL(__x);
#  else // ^^^ _CCCL_BUILTIN_CBRTL ^^^ // vvv !_CCCL_BUILTIN_CBRTL vvv
  return ::cbrtl(__x);
#  endif // !_CCCL_BUILTIN_CBRTL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double cbrtl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_CBRTL)
  return _CCCL_BUILTIN_CBRTL(__x);
#  else // ^^^ _CCCL_BUILTIN_CBRTL ^^^ // vvv !_CCCL_BUILTIN_CBRTL vvv
  return ::cbrtl(__x);
#  endif // !_CCCL_BUILTIN_CBRTL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half cbrt(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::cbrt(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 cbrt(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::cbrt(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double cbrt(_Integer __x) noexcept
{
  return _CUDA_VSTD::cbrt((double) __x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_ROOTS_H
