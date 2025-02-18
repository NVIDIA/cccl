// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_ISFINITE_H
#define _LIBCUDACXX___CMATH_ISFINITE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/common.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integral.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_ISFINITE)
#  define _CCCL_CONSTEXPR_ISFINITE constexpr
#else // ^^^ _CCCL_BUILTIN_ISFINITE ^^^ / vvv !_CCCL_BUILTIN_ISFINITE vvv
#  define _CCCL_CONSTEXPR_ISFINITE
#endif // !_CCCL_BUILTIN_ISFINITE

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISFINITE bool isfinite(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISFINITE)
  return _CCCL_BUILTIN_ISFINITE(__x);
#else
  return ::isfinite(__x);
#endif // defined(_CCCL_BUILTIN_ISFINITE)
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISFINITE bool isfinite(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISFINITE)
  return _CCCL_BUILTIN_ISFINITE(__x);
#else
  return ::isfinite(__x);
#endif // defined(_CCCL_BUILTIN_ISFINITE)
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISFINITE bool isfinite(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ISFINITE)
  return _CCCL_BUILTIN_ISFINITE(__x);
#  else
  return ::isfinite(__x);
#  endif // defined(_CCCL_BUILTIN_ISFINITE)
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(__half __x) noexcept
{
  return (_CUDA_VSTD::__cccl_nvfp_get_storage(__x) & __cccl_nvfp16_exp_mask) != __cccl_nvfp16_exp_mask;
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(__nv_bfloat16 __x) noexcept
{
  return (_CUDA_VSTD::__cccl_nvfp_get_storage(__x) & __cccl_nvbf16_exp_mask) != __cccl_nvbf16_exp_mask;
}
#endif // _LIBCUDACXX_HAS_NVBF16

#if _CCCL_HAS_NVFP8_E4M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(__nv_fp8_e4m3 __x) noexcept
{
  return (__x.__x & __cccl_nvfp8_e4m3_exp_mant_mask) != __cccl_nvfp8_e4m3_exp_mant_mask;
}
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(__nv_fp8_e5m2 __x) noexcept
{
  return (__x.__x & __cccl_nvfp8_e5m2_exp_mask) != __cccl_nvfp8_e5m2_exp_mask;
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(__nv_fp8_e8m0 __x) noexcept
{
  return __x.__x != __cccl_nvfp8_e8m0_exp_mask;
}
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(__nv_fp6_e2m3) noexcept
{
  return true;
}
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(__nv_fp6_e3m2) noexcept
{
  return true;
}
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(__nv_fp4_e2m1) noexcept
{
  return true;
}
#endif // _CCCL_HAS_NVFP4_E2M1()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(_Tp) noexcept
{
  return true;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_ISFINITE_H
