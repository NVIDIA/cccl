//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
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

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/is_integral.h>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __isfinite_impl(_Tp __x) noexcept
{
  static_assert(_CCCL_TRAIT(is_floating_point, _Tp), "Only standard floating-point types are supported");
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    return ::isfinite(__x);
  }
  return !_CUDA_VSTD::isnan(__x) && !_CUDA_VSTD::isinf(__x);
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISFINITE)
  return _CCCL_BUILTIN_ISFINITE(__x);
#else // ^^^ _CCCL_BUILTIN_ISFINITE ^^^ / vvv !_CCCL_BUILTIN_ISFINITE vvv
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    return ::isfinite(__x);
  }
#  if _LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST()
  return (_CUDA_VSTD::__fp_get_storage(__x) & __fp_exp_mask_v<float>) != __fp_exp_mask_v<float>;
#  else // ^^^ _LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST() ^^^ / vvv !_LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST() vvv
  return _CUDA_VSTD::__isfinite_impl(__x);
#  endif // ^^^ !_LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST() ^^^
#endif // ^^^ !_CCCL_BUILTIN_ISFINITE ^^^
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISFINITE)
  return _CCCL_BUILTIN_ISFINITE(__x);
#else // ^^^ _CCCL_BUILTIN_ISFINITE ^^^ / vvv !_CCCL_BUILTIN_ISFINITE vvv
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    return ::isfinite(__x);
  }
#  if _LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST()
  return (_CUDA_VSTD::__fp_get_storage(__x) & __fp_exp_mask_v<double>) != __fp_exp_mask_v<double>;
#  else // ^^^ _LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST() ^^^ / vvv !_LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST() vvv
  return _CUDA_VSTD::__isfinite_impl(__x);
#  endif // ^^^ !_LIBCUDACXX_HAS_CONSTEXPR_BIT_CAST() ^^^
#endif // ^^^ !_CCCL_BUILTIN_ISFINITE ^^^
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ISFINITE)
  return _CCCL_BUILTIN_ISFINITE(__x);
#  else
  return _CUDA_VSTD::__isfinite_impl(__x);
#  endif // defined(_CCCL_BUILTIN_ISFINITE)
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _CCCL_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(__half __x) noexcept
{
  return (_CUDA_VSTD::__fp_get_storage(__x) & __fp_exp_mask_v<__half>) != __fp_exp_mask_v<__half>;
}
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(__nv_bfloat16 __x) noexcept
{
  return (_CUDA_VSTD::__fp_get_storage(__x) & __fp_exp_mask_v<__nv_bfloat16>) != __fp_exp_mask_v<__nv_bfloat16>;
}
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(__nv_fp8_e4m3 __x) noexcept
{
  return (__x.__x & __fp_exp_mant_mask_v<__nv_fp8_e4m3>) != __fp_exp_mant_mask_v<__nv_fp8_e4m3>;
}
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(__nv_fp8_e5m2 __x) noexcept
{
  return (__x.__x & __fp_exp_mask_v<__nv_fp8_e5m2>) != __fp_exp_mask_v<__nv_fp8_e5m2>;
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(__nv_fp8_e8m0 __x) noexcept
{
  return __x.__x != __fp_exp_mask_v<__nv_fp8_e8m0>;
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
