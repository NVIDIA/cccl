// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_ISNAN_H
#define _LIBCUDACXX___CMATH_ISNAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__cmath/common.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integral.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_ISNAN)
#  define _CCCL_CONSTEXPR_ISNAN constexpr
#else // ^^^ _CCCL_BUILTIN_ISNAN ^^^ / vvv !_CCCL_BUILTIN_ISNAN vvv
#  define _CCCL_CONSTEXPR_ISNAN
#endif // !_CCCL_BUILTIN_ISNAN

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISNAN bool isnan(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISNAN)
  return _CCCL_BUILTIN_ISNAN(__x);
#else
  return ::isnan(__x);
#endif // defined(_CCCL_BUILTIN_ISNAN)
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISNAN bool isnan(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISNAN)
  return _CCCL_BUILTIN_ISNAN(__x);
#else
  return ::isnan(__x);
#endif // defined(_CCCL_BUILTIN_ISNAN)
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISNAN bool isnan(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ISNAN)
  return _CCCL_BUILTIN_ISNAN(__x);
#  else
  return ::isnan(__x);
#  endif // defined(_CCCL_BUILTIN_ISNAN)
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isnan(__half __x) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    return ::__hisnan(__x);
  }

  const auto __storage = _CUDA_VSTD::__nv_fp_get_storage(__x);
  return ((__storage & __nv_fp16_exp_mask) == __nv_fp16_exp_mask) && (__storage & __nv_fp16_mant_mask);
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isnan(__nv_bfloat16 __x) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    return ::__hisnan(__x);
  }

  const auto __storage = _CUDA_VSTD::__nv_fp_get_storage(__x);
  return ((__storage & __nv_bf16_exp_mask) == __nv_bf16_exp_mask) && (__storage & __nv_bf16_mant_mask);
}
#endif // _LIBCUDACXX_HAS_NVBF16

#if _CCCL_HAS_NVFP8_E4M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isnan(__nv_fp8_e4m3 __x) noexcept
{
  return (__x.__x & __nv_fp8_e4m3_exp_mant_mask) == __nv_fp8_e4m3_exp_mant_mask;
}
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isnan(__nv_fp8_e5m2 __x) noexcept
{
  return ((__x.__x & __nv_fp8_e5m2_exp_mask) == __nv_fp8_e5m2_exp_mask) && (__x.__x & __nv_fp8_e5m2_mant_mask);
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isnan(__nv_fp8_e8m0 __x) noexcept
{
  return (__x.__x & __nv_fp8_e8m0_exp_mask) == __nv_fp8_e8m0_exp_mask;
}
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isnan(__nv_fp6_e2m3) noexcept
{
  return false;
}
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isnan(__nv_fp6_e3m2) noexcept
{
  return false;
}
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isnan(__nv_fp4_e2m1) noexcept
{
  return false;
}
#endif // _CCCL_HAS_NVFP4_E2M1()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isnan(_Tp) noexcept
{
  return false;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_ISNAN_H
