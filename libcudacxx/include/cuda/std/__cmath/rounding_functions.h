// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_ROUNDING_FUNCTIONS_H
#define _LIBCUDACXX___CMATH_ROUNDING_FUNCTIONS_H

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
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/promote.h>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// ceil

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float ceil(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_CEILF)
  return _CCCL_BUILTIN_CEILF(__x);
#else // ^^^ _CCCL_BUILTIN_CEILF ^^^ // vvv !_CCCL_BUILTIN_CEILF vvv
  return ::ceilf(__x);
#endif // !_CCCL_BUILTIN_CEILF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float ceilf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_CEILF)
  return _CCCL_BUILTIN_CEILF(__x);
#else // ^^^ _CCCL_BUILTIN_CEILF ^^^ // vvv !_CCCL_BUILTIN_CEILF vvv
  return ::ceilf(__x);
#endif // !_CCCL_BUILTIN_CEILF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double ceil(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_CEIL)
  return _CCCL_BUILTIN_CEIL(__x);
#else // ^^^ _CCCL_BUILTIN_CEIL ^^^ // vvv !_CCCL_BUILTIN_CEIL vvv
  return ::ceil(__x);
#endif // !_CCCL_BUILTIN_CEIL
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double ceil(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_CEILL)
  return _CCCL_BUILTIN_CEILL(__x);
#  else // ^^^ _CCCL_BUILTIN_CEILL ^^^ // vvv !_CCCL_BUILTIN_CEILL vvv
  return ::ceill(__x);
#  endif // !_CCCL_BUILTIN_CEILL
}
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double ceill(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_CEILL)
  return _CCCL_BUILTIN_CEILL(__x);
#  else // ^^^ _CCCL_BUILTIN_CEILL ^^^ // vvv !_CCCL_BUILTIN_CEILL vvv
  return ::ceill(__x);
#  endif // !_CCCL_BUILTIN_CEILL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half ceil(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hceil(__x);), (return __float2half(_CUDA_VSTD::ceil(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 ceil(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hceil(__x);), (return __float2bfloat16(_CUDA_VSTD::ceil(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double ceil(_Integer __x) noexcept
{
  return _CUDA_VSTD::ceil((double) __x);
}

// floor

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float floor(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_FLOORF)
  return _CCCL_BUILTIN_FLOORF(__x);
#else // ^^^ _CCCL_BUILTIN_FLOORF ^^^ // vvv !_CCCL_BUILTIN_FLOORF vvv
  return ::floorf(__x);
#endif // !_CCCL_BUILTIN_FLOORF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float floorf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_FLOORF)
  return _CCCL_BUILTIN_FLOORF(__x);
#else // ^^^ _CCCL_BUILTIN_FLOORF ^^^ // vvv !_CCCL_BUILTIN_FLOORF vvv
  return ::floorf(__x);
#endif // !_CCCL_BUILTIN_FLOORF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double floor(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_FLOOR)
  return _CCCL_BUILTIN_FLOOR(__x);
#else // ^^^ _CCCL_BUILTIN_FLOOR ^^^ // vvv !_CCCL_BUILTIN_FLOOR vvv
  return ::floor(__x);
#endif // !_CCCL_BUILTIN_FLOOR
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double floor(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_FLOORL)
  return _CCCL_BUILTIN_FLOORL(__x);
#  else // ^^^ _CCCL_BUILTIN_FLOORL ^^^ // vvv !_CCCL_BUILTIN_FLOORL vvv
  return ::floorl(__x);
#  endif // !_CCCL_BUILTIN_FLOORL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double floorl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_FLOORL)
  return _CCCL_BUILTIN_FLOORL(__x);
#  else // ^^^ _CCCL_BUILTIN_FLOORL ^^^ // vvv !_CCCL_BUILTIN_FLOORL vvv
  return ::floorl(__x);
#  endif // !_CCCL_BUILTIN_FLOORL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half floor(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hfloor(__x);), (return __float2half(_CUDA_VSTD::floor(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 floor(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hfloor(__x);), (return __float2bfloat16(_CUDA_VSTD::floor(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double floor(_Integer __x) noexcept
{
  return _CUDA_VSTD::floor((double) __x);
}

// llrint

_LIBCUDACXX_HIDE_FROM_ABI long long llrint(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LLRINTF)
  return _CCCL_BUILTIN_LLRINTF(__x);
#else // ^^^ _CCCL_BUILTIN_LLRINTF ^^^ // vvv !_CCCL_BUILTIN_LLRINTF vvv
  return ::llrintf(__x);
#endif // !_CCCL_BUILTIN_LLRINTF
}

_LIBCUDACXX_HIDE_FROM_ABI long long llrintf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LLRINTF)
  return _CCCL_BUILTIN_LLRINTF(__x);
#else // ^^^ _CCCL_BUILTIN_LLRINTF ^^^ // vvv !_CCCL_BUILTIN_LLRINTF vvv
  return ::llrintf(__x);
#endif // !_CCCL_BUILTIN_LLRINTF
}

_LIBCUDACXX_HIDE_FROM_ABI long long llrint(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LLRINT)
  return _CCCL_BUILTIN_LLRINT(__x);
#else // ^^^ _CCCL_BUILTIN_LLRINT ^^^ // vvv !_CCCL_BUILTIN_LLRINT vvv
  return ::llrint(__x);
#endif // !_CCCL_BUILTIN_LLRINT
}

#if _CCCL_HAS_LONG_DOUBLE()
_LIBCUDACXX_HIDE_FROM_ABI long long llrint(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LLRINTL)
  return _CCCL_BUILTIN_LLRINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_LLRINTL ^^^ // vvv !_CCCL_BUILTIN_LLRINTL vvv
  return ::llrintl(__x);
#  endif // !_CCCL_BUILTIN_LLRINTL
}

_LIBCUDACXX_HIDE_FROM_ABI long long llrintl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LLRINTL)
  return _CCCL_BUILTIN_LLRINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_LLRINTL ^^^ // vvv !_CCCL_BUILTIN_LLRINTL vvv
  return ::llrintl(__x);
#  endif // !_CCCL_BUILTIN_LLRINTL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long long llrint(__half __x) noexcept
{
  return _CUDA_VSTD::llrintf(__half2float(__x));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long long llrint(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::llrintf(__bfloat162float(__x));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_LIBCUDACXX_HIDE_FROM_ABI long long llrint(_Integer __x) noexcept
{
  return _CUDA_VSTD::llrint((double) __x);
}

// llround

_LIBCUDACXX_HIDE_FROM_ABI long long llround(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LLROUNDF)
  return _CCCL_BUILTIN_LLROUNDF(__x);
#else // ^^^ _CCCL_BUILTIN_LLROUNDF ^^^ // vvv !_CCCL_BUILTIN_LLROUNDF vvv
  return ::llroundf(__x);
#endif // !_CCCL_BUILTIN_LLROUNDF
}

_LIBCUDACXX_HIDE_FROM_ABI long long llroundf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LLROUNDF)
  return _CCCL_BUILTIN_LLROUNDF(__x);
#else // ^^^ _CCCL_BUILTIN_LLROUNDF ^^^ // vvv !_CCCL_BUILTIN_LLROUNDF vvv
  return ::llroundf(__x);
#endif // !_CCCL_BUILTIN_LLROUNDF
}

_LIBCUDACXX_HIDE_FROM_ABI long long llround(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LLROUND)
  return _CCCL_BUILTIN_LLROUND(__x);
#else // ^^^ _CCCL_BUILTIN_LLROUND ^^^ // vvv !_CCCL_BUILTIN_LLROUND vvv
  return ::llround(__x);
#endif // !_CCCL_BUILTIN_LLROUND
}

#if _CCCL_HAS_LONG_DOUBLE()
_LIBCUDACXX_HIDE_FROM_ABI long long llround(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LLROUNDL)
  return _CCCL_BUILTIN_LLROUNDL(__x);
#  else // ^^^ _CCCL_BUILTIN_LLROUNDL ^^^ // vvv !_CCCL_BUILTIN_LLROUNDL vvv
  return ::llroundl(__x);
#  endif // !_CCCL_BUILTIN_LLROUNDL
}

_LIBCUDACXX_HIDE_FROM_ABI long long llroundl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LLROUNDL)
  return _CCCL_BUILTIN_LLROUNDL(__x);
#  else // ^^^ _CCCL_BUILTIN_LLROUNDL ^^^ // vvv !_CCCL_BUILTIN_LLROUNDL vvv
  return ::llroundl(__x);
#  endif // !_CCCL_BUILTIN_LLROUNDL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long long llround(__half __x) noexcept
{
  return _CUDA_VSTD::llroundf(__half2float(__x));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long long llround(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::llroundf(__bfloat162float(__x));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_LIBCUDACXX_HIDE_FROM_ABI long long llround(_Integer __x) noexcept
{
  return _CUDA_VSTD::llround((double) __x);
}

// lrint

_LIBCUDACXX_HIDE_FROM_ABI long lrint(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LRINTF)
  return _CCCL_BUILTIN_LRINTF(__x);
#else // ^^^ _CCCL_BUILTIN_LRINTF ^^^ // vvv !_CCCL_BUILTIN_LRINTF vvv
  return ::lrintf(__x);
#endif // !_CCCL_BUILTIN_LRINTF
}

_LIBCUDACXX_HIDE_FROM_ABI long lrintf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LRINTF)
  return _CCCL_BUILTIN_LRINTF(__x);
#else // ^^^ _CCCL_BUILTIN_LRINTF ^^^ // vvv !_CCCL_BUILTIN_LRINTF vvv
  return ::lrintf(__x);
#endif // !_CCCL_BUILTIN_LRINTF
}

_LIBCUDACXX_HIDE_FROM_ABI long lrint(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LRINT)
  return _CCCL_BUILTIN_LRINT(__x);
#else // ^^^ _CCCL_BUILTIN_LRINT ^^^ // vvv !_CCCL_BUILTIN_LRINT vvv
  return ::lrint(__x);
#endif // !_CCCL_BUILTIN_LRINT
}

#if _CCCL_HAS_LONG_DOUBLE()
_LIBCUDACXX_HIDE_FROM_ABI long lrint(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LRINTL)
  return _CCCL_BUILTIN_LRINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_LRINTL ^^^ // vvv !_CCCL_BUILTIN_LRINTL vvv
  return ::lrintl(__x);
#  endif // !_CCCL_BUILTIN_LRINTL
}

_LIBCUDACXX_HIDE_FROM_ABI long lrintl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LRINTL)
  return _CCCL_BUILTIN_LRINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_LRINTL ^^^ // vvv !_CCCL_BUILTIN_LRINTL vvv
  return ::lrintl(__x);
#  endif // !_CCCL_BUILTIN_LRINTL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long lrint(__half __x) noexcept
{
  return _CUDA_VSTD::lrintf(__half2float(__x));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long lrint(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::lrintf(__bfloat162float(__x));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_LIBCUDACXX_HIDE_FROM_ABI long lrint(_Integer __x) noexcept
{
  return _CUDA_VSTD::lrint((double) __x);
}

// lround

_LIBCUDACXX_HIDE_FROM_ABI long lround(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LROUNDF)
  return _CCCL_BUILTIN_LROUNDF(__x);
#else // ^^^ _CCCL_BUILTIN_LROUNDF ^^^ // vvv !_CCCL_BUILTIN_LROUNDF vvv
  return ::lroundf(__x);
#endif // !_CCCL_BUILTIN_LROUNDF
}

_LIBCUDACXX_HIDE_FROM_ABI long lroundf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LROUNDF)
  return _CCCL_BUILTIN_LROUNDF(__x);
#else // ^^^ _CCCL_BUILTIN_LROUNDF ^^^ // vvv !_CCCL_BUILTIN_LROUNDF vvv
  return ::lroundf(__x);
#endif // !_CCCL_BUILTIN_LROUNDF
}

_LIBCUDACXX_HIDE_FROM_ABI long lround(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LROUND)
  return _CCCL_BUILTIN_LROUND(__x);
#else // ^^^ _CCCL_BUILTIN_LROUND ^^^ // vvv !_CCCL_BUILTIN_LROUND vvv
  return ::lround(__x);
#endif // !_CCCL_BUILTIN_LROUND
}

#if _CCCL_HAS_LONG_DOUBLE()
_LIBCUDACXX_HIDE_FROM_ABI long lround(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LROUNDL)
  return _CCCL_BUILTIN_LROUNDL(__x);
#  else // ^^^ _CCCL_BUILTIN_LROUNDL ^^^ // vvv !_CCCL_BUILTIN_LROUNDL vvv
  return ::lroundl(__x);
#  endif // !_CCCL_BUILTIN_LROUNDL
}

_LIBCUDACXX_HIDE_FROM_ABI long lroundl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LROUNDL)
  return _CCCL_BUILTIN_LROUNDL(__x);
#  else // ^^^ _CCCL_BUILTIN_LROUNDL ^^^ // vvv !_CCCL_BUILTIN_LROUNDL vvv
  return ::lroundl(__x);
#  endif // !_CCCL_BUILTIN_LROUNDL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long lround(__half __x) noexcept
{
  return _CUDA_VSTD::lroundf(__half2float(__x));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long lround(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::lroundf(__bfloat162float(__x));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_LIBCUDACXX_HIDE_FROM_ABI long lround(_Integer __x) noexcept
{
  return _CUDA_VSTD::lround((double) __x);
}

// nearbyint

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float nearbyint(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_NEARBYINTF)
  return _CCCL_BUILTIN_NEARBYINTF(__x);
#else // ^^^ _CCCL_BUILTIN_NEARBYINTF ^^^ // vvv !_CCCL_BUILTIN_NEARBYINTF vvv
  return ::nearbyintf(__x);
#endif // !_CCCL_BUILTIN_NEARBYINTF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float nearbyintf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_NEARBYINTF)
  return _CCCL_BUILTIN_NEARBYINTF(__x);
#else // ^^^ _CCCL_BUILTIN_NEARBYINTF ^^^ // vvv !_CCCL_BUILTIN_NEARBYINTF vvv
  return ::nearbyintf(__x);
#endif // !_CCCL_BUILTIN_NEARBYINTF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double nearbyint(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_NEARBYINT)
  return _CCCL_BUILTIN_NEARBYINT(__x);
#else // ^^^ _CCCL_BUILTIN_NEARBYINT ^^^ // vvv !_CCCL_BUILTIN_NEARBYINT vvv
  return ::nearbyint(__x);
#endif // !_CCCL_BUILTIN_NEARBYINT
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double nearbyint(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_NEARBYINTL)
  return _CCCL_BUILTIN_NEARBYINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_NEARBYINTL ^^^ // vvv !_CCCL_BUILTIN_NEARBYINTL vvv
  return ::nearbyintl(__x);
#  endif // !_CCCL_BUILTIN_NEARBYINTL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double nearbyintl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_NEARBYINTL)
  return _CCCL_BUILTIN_NEARBYINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_NEARBYINTL ^^^ // vvv !_CCCL_BUILTIN_NEARBYINTL vvv
  return ::nearbyintl(__x);
#  endif // !_CCCL_BUILTIN_NEARBYINTL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half nearbyint(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::nearbyintf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 nearbyint(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::nearbyintf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double nearbyint(_Integer __x) noexcept
{
  return _CUDA_VSTD::nearbyint((double) __x);
}

// nextafter

_LIBCUDACXX_HIDE_FROM_ABI float nextafter(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_NEXTAFTERF)
  return _CCCL_BUILTIN_NEXTAFTERF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_NEXTAFTERF ^^^ // vvv !_CCCL_BUILTIN_NEXTAFTERF vvv
  return ::nextafterf(__x, __y);
#endif // !_CCCL_BUILTIN_NEXTAFTERF
}

_LIBCUDACXX_HIDE_FROM_ABI float nextafterf(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_NEXTAFTERF)
  return _CCCL_BUILTIN_NEXTAFTERF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_NEXTAFTERF ^^^ // vvv !_CCCL_BUILTIN_NEXTAFTERF vvv
  return ::nextafterf(__x, __y);
#endif // !_CCCL_BUILTIN_NEXTAFTERF
}

_LIBCUDACXX_HIDE_FROM_ABI double nextafter(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_NEXTAFTER)
  return _CCCL_BUILTIN_NEXTAFTER(__x, __y);
#else // ^^^ _CCCL_BUILTIN_NEXTAFTER ^^^ // vvv !_CCCL_BUILTIN_NEXTAFTER vvv
  return ::nextafter(__x, __y);
#endif // !_CCCL_BUILTIN_NEXTAFTER
}

#if _CCCL_HAS_LONG_DOUBLE()
_LIBCUDACXX_HIDE_FROM_ABI long double nextafter(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTAFTERL)
  return _CCCL_BUILTIN_NEXTAFTERL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTAFTERL ^^^ // vvv !_CCCL_BUILTIN_NEXTAFTERL vvv
  return ::nextafterl(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTAFTERL
}

_LIBCUDACXX_HIDE_FROM_ABI long double nextafterl(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTAFTERL)
  return _CCCL_BUILTIN_NEXTAFTERL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTAFTERL ^^^ // vvv !_CCCL_BUILTIN_NEXTAFTERL vvv
  return ::nextafterl(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTAFTERL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half nextafter(__half __x, __half __y) noexcept
{
  return __float2half(_CUDA_VSTD::nextafterf(__half2float(__x), __half2float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 nextafter(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::nextafterf(__bfloat162float(__x), __bfloat162float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1, class _A2, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1) && _CCCL_TRAIT(is_arithmetic, _A2), int> = 0>
_LIBCUDACXX_HIDE_FROM_ABI __promote_t<_A1, _A2> nextafter(_A1 __x, _A2 __y) noexcept
{
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(_CCCL_TRAIT(is_same, _A1, __result_type) && _CCCL_TRAIT(is_same, _A2, __result_type)), "");
  return _CUDA_VSTD::nextafter(static_cast<__result_type>(__x), static_cast<__result_type>(__y));
}

// nexttoward

#if _CCCL_HAS_LONG_DOUBLE()
_LIBCUDACXX_HIDE_FROM_ABI float nexttoward(float __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTTOWARDF)
  return _CCCL_BUILTIN_NEXTTOWARDF(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTTOWARDF ^^^ // vvv !_CCCL_BUILTIN_NEXTTOWARDF vvv
  return ::nexttowardf(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTTOWARDF
}

_LIBCUDACXX_HIDE_FROM_ABI float nexttowardf(float __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTTOWARDF)
  return _CCCL_BUILTIN_NEXTTOWARDF(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTTOWARDF ^^^ // vvv !_CCCL_BUILTIN_NEXTTOWARDF vvv
  return ::nexttowardf(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTTOWARDF
}

_LIBCUDACXX_HIDE_FROM_ABI double nexttoward(double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTTOWARD)
  return _CCCL_BUILTIN_NEXTTOWARD(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTTOWARD ^^^ // vvv !_CCCL_BUILTIN_NEXTTOWARD vvv
  return ::nexttoward(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTTOWARD
}

_LIBCUDACXX_HIDE_FROM_ABI long double nexttoward(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTTOWARDL)
  return _CCCL_BUILTIN_NEXTTOWARDL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTTOWARDL ^^^ // vvv !_CCCL_BUILTIN_NEXTTOWARDL vvv
  return ::nexttowardl(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTTOWARDL
}

_LIBCUDACXX_HIDE_FROM_ABI long double nexttowardl(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTTOWARDL)
  return _CCCL_BUILTIN_NEXTTOWARDL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTTOWARDL ^^^ // vvv !_CCCL_BUILTIN_NEXTTOWARDL vvv
  return ::nexttowardl(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTTOWARDL
}

#  if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half nexttoward(__half __x, long double __y) noexcept
{
  return __float2half(_CUDA_VSTD::nexttowardf(__half2float(__x), __y));
}
#  endif // _LIBCUDACXX_HAS_NVFP16()

#  if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 nexttoward(__nv_bfloat16 __x, long double __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::nexttowardf(__bfloat162float(__x), __y));
}
#  endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_LIBCUDACXX_HIDE_FROM_ABI double nexttoward(_Integer __x, long double __y) noexcept
{
  return _CUDA_VSTD::nexttoward((double) __x, __y);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

// rint

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float rint(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_RINTF)
  return _CCCL_BUILTIN_RINTF(__x);
#else // ^^^ _CCCL_BUILTIN_RINTF ^^^ // vvv !_CCCL_BUILTIN_RINTF vvv
  return ::rintf(__x);
#endif // !_CCCL_BUILTIN_RINTF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float rintf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_RINTF)
  return _CCCL_BUILTIN_RINTF(__x);
#else // ^^^ _CCCL_BUILTIN_RINTF ^^^ // vvv !_CCCL_BUILTIN_RINTF vvv
  return ::rintf(__x);
#endif // !_CCCL_BUILTIN_RINTF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double rint(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_RINT)
  return _CCCL_BUILTIN_RINT(__x);
#else // ^^^ _CCCL_BUILTIN_RINT ^^^ // vvv !_CCCL_BUILTIN_RINT vvv
  return ::rint(__x);
#endif // !_CCCL_BUILTIN_RINT
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double rint(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_RINTL)
  return _CCCL_BUILTIN_RINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_RINTL ^^^ // vvv !_CCCL_BUILTIN_RINTL vvv
  return ::rintl(__x);
#  endif // !_CCCL_BUILTIN_RINTL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double rintl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_RINTL)
  return _CCCL_BUILTIN_RINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_RINTL ^^^ // vvv !_CCCL_BUILTIN_RINTL vvv
  return ::rintl(__x);
#  endif // !_CCCL_BUILTIN_RINTL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half rint(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hrint(__x);), (return __float2half(_CUDA_VSTD::rint(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 rint(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hrint(__x);), (return __float2bfloat16(_CUDA_VSTD::rint(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double rint(_Integer __x) noexcept
{
  return _CUDA_VSTD::rint((double) __x);
}

// round

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float round(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ROUNDF)
  return _CCCL_BUILTIN_ROUNDF(__x);
#else // ^^^ _CCCL_BUILTIN_ROUNDF ^^^ // vvv !_CCCL_BUILTIN_ROUNDF vvv
  return ::roundf(__x);
#endif // !_CCCL_BUILTIN_ROUNDF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float roundf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ROUNDF)
  return _CCCL_BUILTIN_ROUNDF(__x);
#else // ^^^ _CCCL_BUILTIN_ROUNDF ^^^ // vvv !_CCCL_BUILTIN_ROUNDF vvv
  return ::roundf(__x);
#endif // !_CCCL_BUILTIN_ROUNDF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double round(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ROUND)
  return _CCCL_BUILTIN_ROUND(__x);
#else // ^^^ _CCCL_BUILTIN_ROUND ^^^ // vvv !_CCCL_BUILTIN_ROUND vvv
  return ::round(__x);
#endif // !_CCCL_BUILTIN_ROUND
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double round(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ROUNDL)
  return _CCCL_BUILTIN_ROUNDL(__x);
#  else // ^^^ _CCCL_BUILTIN_ROUNDL ^^^ // vvv !_CCCL_BUILTIN_ROUNDL vvv
  return ::roundl(__x);
#  endif // !_CCCL_BUILTIN_ROUNDL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double roundl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ROUNDL)
  return _CCCL_BUILTIN_ROUNDL(__x);
#  else // ^^^ _CCCL_BUILTIN_ROUNDL ^^^ // vvv !_CCCL_BUILTIN_ROUNDL vvv
  return ::roundl(__x);
#  endif // !_CCCL_BUILTIN_ROUNDL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half round(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::roundf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 round(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::roundf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double round(_Integer __x) noexcept
{
  return _CUDA_VSTD::round((double) __x);
}

// trunc

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float trunc(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_TRUNCF)
  return _CCCL_BUILTIN_TRUNCF(__x);
#else // ^^^ _CCCL_BUILTIN_TRUNCF ^^^ // vvv !_CCCL_BUILTIN_TRUNCF vvv
  return ::truncf(__x);
#endif // !_CCCL_BUILTIN_TRUNCF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float truncf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_TRUNCF)
  return _CCCL_BUILTIN_TRUNCF(__x);
#else // ^^^ _CCCL_BUILTIN_TRUNCF ^^^ // vvv !_CCCL_BUILTIN_TRUNCF vvv
  return ::truncf(__x);
#endif // !_CCCL_BUILTIN_TRUNCF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double trunc(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_TRUNC)
  return _CCCL_BUILTIN_TRUNC(__x);
#else // ^^^ _CCCL_BUILTIN_TRUNC ^^^ // vvv !_CCCL_BUILTIN_TRUNC vvv
  return ::trunc(__x);
#endif // !_CCCL_BUILTIN_TRUNC
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double trunc(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_TRUNCL)
  return _CCCL_BUILTIN_TRUNCL(__x);
#  else // ^^^ _CCCL_BUILTIN_TRUNCL ^^^ // vvv !_CCCL_BUILTIN_TRUNCL vvv
  return ::truncl(__x);
#  endif // !_CCCL_BUILTIN_TRUNCL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double truncl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_TRUNCL)
  return _CCCL_BUILTIN_TRUNCL(__x);
#  else // ^^^ _CCCL_BUILTIN_TRUNCL ^^^ // vvv !_CCCL_BUILTIN_TRUNCL vvv
  return ::truncl(__x);
#  endif // !_CCCL_BUILTIN_TRUNCL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half trunc(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::htrunc(__x);), (return __float2half(_CUDA_VSTD::trunc(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 trunc(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::htrunc(__x);), (return __float2bfloat16(_CUDA_VSTD::trunc(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double trunc(_Integer __x) noexcept
{
  return _CUDA_VSTD::trunc((double) __x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_ROUNDING_FUNCTIONS_H
