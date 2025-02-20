//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_SIGNBIT_H
#define _LIBCUDACXX___CMATH_SIGNBIT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/copysign.h>
#include <cuda/std/__cmath/fp_utils.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool signbit(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_SIGNBITF) && !_CCCL_CUDA_COMPILER(NVCC)
  return _CCCL_BUILTIN_SIGNBITF(__x);
#elif defined(_CCCL_BUILTIN_SIGNBITF)
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    return _CCCL_BUILTIN_SIGNBIT(__x);
  }
  // nvcc has bug in builtin signbit, todo report and add nvbug number
  return _CUDA_VSTD::copysign(1.0f, __x) < 0.0f;
#elif _LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST()
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, (return ::signbit(__x);))
  }
  return _CUDA_VSTD::__cccl_fp_get_storage(__x) & __cccl_fp32_sign_mask;
#else // ^^^ _LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST() ^^^ / vvv !_LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST() vvv
  return ::signbit(__x);
#endif // !_CCCL_BUILTIN_SIGNBITF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool signbit(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_SIGNBIT) && !_CCCL_CUDA_COMPILER(NVCC)
  return _CCCL_BUILTIN_SIGNBIT(__x);
#elif defined(_CCCL_BUILTIN_SIGNBIT)
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    return _CCCL_BUILTIN_SIGNBIT(__x);
  }
  // nvcc has bug in builtin signbit, todo report and add nvbug number
  return _CUDA_VSTD::copysign(1.0, __x) < 0.0;
#elif _LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST()
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, (return ::signbit(__x);))
  }
  return _CUDA_VSTD::__cccl_fp_get_storage(__x) & __cccl_fp64_sign_mask;
#else // ^^^ _LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST() ^^^ / vvv !_LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST() vvv
  return ::signbit(__x);
#endif // !_CCCL_BUILTIN_SIGNBIT
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool signbit(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_SIGNBITL)
  return _CCCL_BUILTIN_SIGNBITL(__x);
#  else // ^^^ _CCCL_BUILTIN_SIGNBITL ^^^ / vvv !_CCCL_BUILTIN_SIGNBITL vvv
  return ::signbit(__x);
#  endif // !_CCCL_BUILTIN_SIGNBITL
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool signbit(__half __x) noexcept
{
  return _CUDA_VSTD::__cccl_fp_get_storage(__x) & __cccl_nvfp16_sign_mask;
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool signbit(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::__cccl_fp_get_storage(__x) & __cccl_nvbf16_sign_mask;
}
#endif // _LIBCUDACXX_HAS_NVBF16

#if _CCCL_HAS_NVFP8_E4M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool signbit(__nv_fp8_e4m3 __x) noexcept
{
  return __x.__x & __cccl_nvfp8_e4m3_sign_mask;
}
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool signbit(__nv_fp8_e5m2 __x) noexcept
{
  return __x.__x & __cccl_nvfp8_e5m2_sign_mask;
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool signbit(__nv_fp8_e8m0) noexcept
{
  return false;
}
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool signbit(__nv_fp6_e2m3 __x) noexcept
{
  return __x.__x & __cccl_nvfp6_e2m3_sign_mask;
}
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool signbit(__nv_fp6_e3m2 __x) noexcept
{
  return __x.__x & __cccl_nvfp6_e3m2_sign_mask;
}
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool signbit(__nv_fp4_e2m1 __x) noexcept
{
  return __x.__x & __cccl_nvfp4_e2m1_sign_mask;
}
#endif // _CCCL_HAS_NVFP4_E2M1()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool signbit(_Tp __x) noexcept
{
  if constexpr (_CCCL_TRAIT(is_signed, _Tp))
  {
    return __x < 0;
  }
  else
  {
    return false;
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_SIGNBIT_H
