// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_FPCLASSIFY_H
#define _LIBCUDACXX___CMATH_FPCLASSIFY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__cmath/common.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>

#if _CCCL_COMPILER(NVRTC)
#  ifndef FP_NAN
#    define FP_NAN 0
#  endif // ! FP_NAN
#  ifndef FP_INFINITE
#    define FP_INFINITE 1
#  endif // ! FP_INFINITE
#  ifndef FP_ZERO
#    define FP_ZERO 2
#  endif // ! FP_ZERO
#  ifndef FP_SUBNORMAL
#    define FP_SUBNORMAL 3
#  endif // ! FP_SUBNORMAL
#  ifndef FP_NORMAL
#    define FP_NORMAL 4
#  endif // ! FP_NORMAL
#else // ^^^ _CCCL_COMPILER(NVRTC) ^^^ ^/  vvv !_CCCL_COMPILER(NVRTC) vvv
#  include <math.h>
#endif // !_CCCL_COMPILER(NVRTC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct _CCCL_FLOAT_BITS
{
#if defined(_LIBCUDACXX_LITTLE_ENDIAN)
  unsigned int man  : 23;
  unsigned int exp  : 8;
  unsigned int sign : 1;
#else // ^^^ _LIBCUDACXX_LITTLE_ENDIAN ^^^ / vvv _LIBCUDACXX_BIG_ENDIAN vvv
  unsigned int sign : 1;
  unsigned int exp  : 8;
  unsigned int man  : 23;
#endif // _LIBCUDACXX_BIG_ENDIAN
};

struct _CCCL_DOUBLE_BITS
{
#if defined(_LIBCUDACXX_LITTLE_ENDIAN)
  unsigned int manl : 32;
  unsigned int manh : 20;
  unsigned int exp  : 11;
  unsigned int sign : 1;
#else // ^^^ _LIBCUDACXX_LITTLE_ENDIAN ^^^ / vvv _LIBCUDACXX_BIG_ENDIAN vvv
  unsigned int sign : 1;
  unsigned int exp  : 11;
  unsigned int manh : 20;
  unsigned int manl : 32;
#endif // _LIBCUDACXX_BIG_ENDIAN
};

#if defined(_LIBCUDACXX_HAS_NVFP16)
struct _CCCL_HALF_BITS
{
#  if defined(_LIBCUDACXX_LITTLE_ENDIAN)
  unsigned short man  : 10;
  unsigned short exp  : 5;
  unsigned short sign : 1;
#  else // ^^^ _LIBCUDACXX_LITTLE_ENDIAN ^^^ / vvv _LIBCUDACXX_BIG_ENDIAN vvv
  unsigned short sign : 1;
  unsigned short exp  : 5;
  unsigned short man  : 10;
#  endif // _LIBCUDACXX_BIG_ENDIAN
};
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
struct _CCCL_NVBFLOAT_BITS
{
#  if defined(_LIBCUDACXX_LITTLE_ENDIAN)
  unsigned short man  : 7;
  unsigned short exp  : 8;
  unsigned short sign : 1;
#  else // ^^^ _LIBCUDACXX_LITTLE_ENDIAN ^^^ / vvv _LIBCUDACXX_BIG_ENDIAN vvv
  unsigned short sign : 1;
  unsigned short exp  : 8;
  unsigned short man  : 7;
#  endif // _LIBCUDACXX_BIG_ENDIAN
};
#endif // _LIBCUDACXX_HAS_NVBF16

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int fpclassify(float __x) noexcept
{
  _CCCL_FLOAT_BITS __bits = _CUDA_VSTD::bit_cast<_CCCL_FLOAT_BITS>(__x);
  if (__bits.exp == 0)
  {
    return __bits.man == 0 ? FP_ZERO : FP_SUBNORMAL;
  }
  if (__bits.exp == 255)
  {
    return __bits.man == 0 ? FP_INFINITE : FP_NAN;
  }
  return (FP_NORMAL);
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int fpclassify(double __x) noexcept
{
  _CCCL_DOUBLE_BITS __bits = _CUDA_VSTD::bit_cast<_CCCL_DOUBLE_BITS>(__x);
  if (__bits.exp == 0)
  {
    return (__bits.manl | __bits.manh) == 0 ? FP_ZERO : FP_SUBNORMAL;
  }
  if (__bits.exp == 2047)
  {
    return (__bits.manl | __bits.manh) == 0 ? FP_INFINITE : FP_NAN;
  }
  return (FP_NORMAL);
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int fpclassify(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_FPCLASSIFY)
  return _CCCL_BUILTIN_FPCLASSIFY(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, __x);
#  else // ^^^ _CCCL_BUILTIN_SIGNBIT ^^^ / vvv !_CCCL_BUILTIN_SIGNBIT vvv
  return ::fpclassify(__x);
#  endif // !_CCCL_BUILTIN_SIGNBIT
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int fpclassify(__half __x) noexcept
{
  _CCCL_HALF_BITS __bits = _CUDA_VSTD::bit_cast<_CCCL_HALF_BITS>(__x);
  if (__bits.exp == 0)
  {
    return __bits.man == 0 ? FP_ZERO : FP_SUBNORMAL;
  }
  if (__bits.exp == 31)
  {
    return __bits.man == 0 ? FP_INFINITE : FP_NAN;
  }
  return (FP_NORMAL);
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int fpclassify(__nv_bfloat16 __x) noexcept
{
  _CCCL_NVBFLOAT_BITS __bits = _CUDA_VSTD::bit_cast<_CCCL_NVBFLOAT_BITS>(__x);
  if (__bits.exp == 0)
  {
    return __bits.man == 0 ? FP_ZERO : FP_SUBNORMAL;
  }
  if (__bits.exp == 255)
  {
    return __bits.man == 0 ? FP_INFINITE : FP_NAN;
  }
  return (FP_NORMAL);
}
#endif // _LIBCUDACXX_HAS_NVBF16

template <class _A1, enable_if_t<_CCCL_TRAIT(is_integral, _A1), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int fpclassify(_A1 __x) noexcept
{
  return (__x == 0) ? FP_ZERO : FP_NORMAL;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_FPCLASSIFY_H
