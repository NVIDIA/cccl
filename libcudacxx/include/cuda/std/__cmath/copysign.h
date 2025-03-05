//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_COPYSIGN_H
#define _LIBCUDACXX___CMATH_COPYSIGN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/fp_utils.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/cstdint>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float copysign(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_COPYSIGNF)
  return _CCCL_BUILTIN_COPYSIGNF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_COPYSIGN ^^^ / vvv !_CCCL_BUILTIN_COPYSIGN vvv
  return ::copysignf(__x, __y);
#endif // !_CCCL_BUILTIN_COPYSIGN
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double copysign(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_COPYSIGN)
  return _CCCL_BUILTIN_COPYSIGN(__x, __y);
#else // ^^^ _CCCL_BUILTIN_COPYSIGN ^^^ / vvv !_CCCL_BUILTIN_COPYSIGN vvv
  return ::copysign(__x, __y);
#endif // !_CCCL_BUILTIN_COPYSIGN
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double copysign(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_COPYSIGNL)
  return _CCCL_BUILTIN_COPYSIGNL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_COPYSIGNL ^^^ / vvv !_CCCL_BUILTIN_COPYSIGNL vvv
  return ::copysignl(__x, __y);
#  endif // !_CCCL_BUILTIN_COPYSIGNL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __half copysign(__half __x, __half __y) noexcept
{
  const auto __val = (_CUDA_VSTD::__cccl_fp_get_storage(__x) & __cccl_nvfp16_exp_mant_mask)
                   | (_CUDA_VSTD::__cccl_fp_get_storage(__y) & __cccl_nvfp16_sign_mask);
  return _CUDA_VSTD::__cccl_make_nvfp16_from_storage(static_cast<uint16_t>(__val));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_bfloat16 copysign(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  const auto __val = (_CUDA_VSTD::__cccl_fp_get_storage(__x) & __cccl_nvbf16_exp_mant_mask)
                   | (_CUDA_VSTD::__cccl_fp_get_storage(__y) & __cccl_nvbf16_sign_mask);
  return _CUDA_VSTD::__cccl_make_nvbf16_from_storage(static_cast<uint16_t>(__val));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp8_e4m3 copysign(__nv_fp8_e4m3 __x, __nv_fp8_e4m3 __y) noexcept
{
  __x.__x = static_cast<__nv_fp8_storage_t>(
    (__x.__x & __cccl_nvfp8_e4m3_exp_mant_mask) | (__y.__x & __cccl_nvfp8_e4m3_sign_mask));
  return __x;
}
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp8_e5m2 copysign(__nv_fp8_e5m2 __x, __nv_fp8_e5m2 __y) noexcept
{
  __x.__x = static_cast<__nv_fp8_storage_t>(
    (__x.__x & __cccl_nvfp8_e5m2_exp_mant_mask) | (__y.__x & __cccl_nvfp8_e5m2_sign_mask));
  return __x;
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp8_e8m0 copysign(__nv_fp8_e8m0 __x, __nv_fp8_e8m0) noexcept
{
  return __x;
}
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp6_e2m3 copysign(__nv_fp6_e2m3 __x, __nv_fp6_e2m3 __y) noexcept
{
  __x.__x = static_cast<__nv_fp6_storage_t>(
    (__x.__x & __cccl_nvfp6_e2m3_exp_mant_mask) | (__y.__x & __cccl_nvfp6_e2m3_sign_mask));
  return __x;
}
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp6_e3m2 copysign(__nv_fp6_e3m2 __x, __nv_fp6_e3m2 __y) noexcept
{
  __x.__x = static_cast<__nv_fp6_storage_t>(
    (__x.__x & __cccl_nvfp6_e3m2_exp_mant_mask) | (__y.__x & __cccl_nvfp6_e3m2_sign_mask));
  return __x;
}
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp4_e2m1 copysign(__nv_fp4_e2m1 __x, __nv_fp4_e2m1 __y) noexcept
{
  __x.__x = static_cast<__nv_fp4_storage_t>(
    (__x.__x & __cccl_nvfp4_e2m1_exp_mant_mask) | (__y.__x & __cccl_nvfp4_e2m1_sign_mask));
  return __x;
}
#endif // _CCCL_HAS_NVFP4_E2M1()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr double copysign(_Tp __x, _Tp __y) noexcept
{
  if constexpr (_CCCL_TRAIT(is_signed, _Tp))
  {
    const auto __x_dbl = static_cast<double>(__x);
    if (__y < 0)
    {
      return (__x < 0) ? __x_dbl : -__x_dbl;
    }
    else
    {
      return (__x < 0) ? -__x_dbl : __x_dbl;
    }
  }
  else
  {
    _LIBCUDACXX_UNUSED_VAR(__y);
    return static_cast<double>(__x);
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_COPYSIGN_H
