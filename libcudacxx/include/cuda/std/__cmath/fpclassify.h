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
#include <cuda/std/__concepts/concept_macros.h>
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

#if defined(_CCCL_BUILTIN_FPCLASSIFY)
#  define _CCCL_CONSTEXPR_FPCLASSIFY       constexpr
#  define _CCCL_HAS_CONSTEXPR_FPCLASSIFY() 1
#else // ^^^ _CCCL_BUILTIN_FPCLASSIFY ^^^ / vvv !_CCCL_BUILTIN_FPCLASSIFY vvv
#  define _CCCL_CONSTEXPR_FPCLASSIFY
#  define _CCCL_HAS_CONSTEXPR_FPCLASSIFY() 0
#endif // !_CCCL_BUILTIN_FPCLASSIFY

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

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_FPCLASSIFY int fpclassify(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_FPCLASSIFY)
  return _CCCL_BUILTIN_FPCLASSIFY(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, __x);
#else // ^^^ _CCCL_BUILTIN_FPCLASSIFY ^^^ / vvv !_CCCL_BUILTIN_FPCLASSIFY vvv
  _CCCL_FLOAT_BITS __bits = _CUDA_VSTD::bit_cast<_CCCL_FLOAT_BITS>(__x);
  if (__bits.exp == 0)
  {
    return __bits.man == 0 ? FP_ZERO : FP_SUBNORMAL;
  }
  if (__bits.exp == 255)
  {
    return __bits.man == 0 ? FP_INFINITE : FP_NAN;
  }
  return FP_NORMAL;
#endif // !_CCCL_BUILTIN_FPCLASSIFY
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_FPCLASSIFY int fpclassify(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_FPCLASSIFY)
  return _CCCL_BUILTIN_FPCLASSIFY(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, __x);
#else // ^^^ _CCCL_BUILTIN_FPCLASSIFY ^^^ / vvv !_CCCL_BUILTIN_FPCLASSIFY vvv
  _CCCL_DOUBLE_BITS __bits = _CUDA_VSTD::bit_cast<_CCCL_DOUBLE_BITS>(__x);
  if (__bits.exp == 0)
  {
    return (__bits.manl | __bits.manh) == 0 ? FP_ZERO : FP_SUBNORMAL;
  }
  if (__bits.exp == 2047)
  {
    return (__bits.manl | __bits.manh) == 0 ? FP_INFINITE : FP_NAN;
  }
  return FP_NORMAL;
#endif // !_CCCL_BUILTIN_FPCLASSIFY
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_FPCLASSIFY int fpclassify(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_FPCLASSIFY)
  return _CCCL_BUILTIN_FPCLASSIFY(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, __x);
#  else // ^^^ _CCCL_BUILTIN_SIGNBIT ^^^ / vvv !_CCCL_BUILTIN_SIGNBIT vvv
  return ::fpclassify(__x);
#  endif // !_CCCL_BUILTIN_SIGNBIT
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int fpclassify(__half __x) noexcept
{
  const auto __storage = _CUDA_VSTD::__nv_fp_get_storage(__x);
  if ((__storage & __nv_fp16_exp_mask) == 0)
  {
    return (__storage & __nv_fp16_mant_mask) ? FP_SUBNORMAL : FP_ZERO;
  }
  if ((__storage & __nv_fp16_exp_mask) == __nv_fp16_exp_mask)
  {
    return (__storage & __nv_fp16_mant_mask) ? FP_NAN : FP_INFINITE;
  }
  return FP_NORMAL;
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int fpclassify(__nv_bfloat16 __x) noexcept
{
  const auto __storage = _CUDA_VSTD::__nv_fp_get_storage(__x);
  if ((__storage & __nv_bf16_exp_mask) == 0)
  {
    return (__storage & __nv_bf16_mant_mask) ? FP_SUBNORMAL : FP_ZERO;
  }
  if ((__storage & __nv_bf16_exp_mask) == __nv_bf16_exp_mask)
  {
    return (__storage & __nv_bf16_mant_mask) ? FP_NAN : FP_INFINITE;
  }
  return FP_NORMAL;
}
#endif // _LIBCUDACXX_HAS_NVBF16

#if _CCCL_HAS_NVFP8_E4M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int fpclassify(__nv_fp8_e4m3 __x) noexcept
{
  if ((__x.__x & __nv_fp8_e4m3_exp_mask) == 0)
  {
    return (__x.__x & __nv_fp8_e4m3_mant_mask) ? FP_SUBNORMAL : FP_ZERO;
  }
  return ((__x.__x & __nv_fp8_e4m3_exp_mant_mask) == __nv_fp8_e4m3_exp_mant_mask) ? FP_NAN : FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int fpclassify(__nv_fp8_e5m2 __x) noexcept
{
  if ((__x.__x & __nv_fp8_e5m2_exp_mask) == 0)
  {
    return (__x.__x & __nv_fp8_e5m2_mant_mask) ? FP_SUBNORMAL : FP_ZERO;
  }
  if ((__x.__x & __nv_fp8_e5m2_exp_mask) == __nv_fp8_e5m2_exp_mask)
  {
    return (__x.__x & __nv_fp8_e5m2_mant_mask) ? FP_NAN : FP_INFINITE;
  }
  return FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int fpclassify(__nv_fp8_e8m0 __x) noexcept
{
  return ((__x.__x & __nv_fp8_e8m0_exp_mask) == __nv_fp8_e8m0_exp_mask) ? FP_NAN : FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int fpclassify(__nv_fp6_e2m3 __x) noexcept
{
  if ((__x.__x & __nv_fp6_e2m3_exp_mask) == 0)
  {
    return (__x.__x & __nv_fp6_e2m3_mant_mask) ? FP_SUBNORMAL : FP_ZERO;
  }
  return FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int fpclassify(__nv_fp6_e3m2 __x) noexcept
{
  if ((__x.__x & __nv_fp6_e3m2_exp_mask) == 0)
  {
    return (__x.__x & __nv_fp6_e3m2_mant_mask) ? FP_SUBNORMAL : FP_ZERO;
  }
  return FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int fpclassify(__nv_fp4_e2m1 __x) noexcept
{
  if ((__x.__x & __nv_fp4_e2m1_exp_mask) == 0)
  {
    return (__x.__x & __nv_fp4_e2m1_mant_mask) ? FP_SUBNORMAL : FP_ZERO;
  }
  return FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP4_E2M1()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int fpclassify(_Tp __x) noexcept
{
  return (__x == 0) ? FP_ZERO : FP_NORMAL;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_FPCLASSIFY_H
