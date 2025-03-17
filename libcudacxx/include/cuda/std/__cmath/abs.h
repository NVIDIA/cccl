//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_ABS_H
#define _LIBCUDACXX___CMATH_ABS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/signbit.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/cstdint>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// fabs

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float fabs(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_FABSF)
  return _CCCL_BUILTIN_FABSF(__x);
#else // ^^^ _CCCL_BUILTIN_FABSF ^^^ / vvv !_CCCL_BUILTIN_FABSF vvv
  return ::fabsf(__x);
#endif // ^^^ !_CCCL_BUILTIN_FABSF ^^^
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float fabsf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_FABSF)
  return _CCCL_BUILTIN_FABSF(__x);
#else // ^^^ _CCCL_BUILTIN_FABSF ^^^ / vvv !_CCCL_BUILTIN_FABSF vvv
  return ::fabsf(__x);
#endif // ^^^ !_CCCL_BUILTIN_FABSF ^^^
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double fabs(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_FABS)
  return _CCCL_BUILTIN_FABS(__x);
#else // ^^^ _CCCL_BUILTIN_FABS ^^^ / vvv !_CCCL_BUILTIN_FABS vvv
  return ::fabs(__x);
#endif // ^^^ !_CCCL_BUILTIN_FABS ^^^
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double fabs(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_FABSL)
  return _CCCL_BUILTIN_FABSL(__x);
#  else // ^^^ _CCCL_BUILTIN_FABSL ^^^ / vvv !_CCCL_BUILTIN_FABSL vvv
  return ::fabsl(__x);
#  endif // ^^^ !_CCCL_BUILTIN_FABSL ^^^
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double fabsl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_FABSL)
  return _CCCL_BUILTIN_FABSL(__x);
#  else // ^^^ _CCCL_BUILTIN_FABSL ^^^ / vvv !_CCCL_BUILTIN_FABSL vvv
  return ::fabsl(__x);
#  endif // ^^^ !_CCCL_BUILTIN_FABSL ^^^
}
#endif // _CCCL_HAS_LONG_DOUBLE()

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __fabs_impl(_Tp __x) noexcept
{
  const auto __val = _CUDA_VSTD::__fp_get_storage(__x) & __fp_exp_mant_mask_v<_Tp>;
  return _CUDA_VSTD::__fp_from_storage<_Tp>(static_cast<__fp_storage_t<_Tp>>(__val));
}

#if _CCCL_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __half fabs(__half __x) noexcept
{
  // We cannot use `abs.f16` because it is not IEEE 754 compliant, see docs
  return _CUDA_VSTD::__fabs_impl(__x);
}
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_bfloat16 fabs(__nv_bfloat16 __x) noexcept
{
  // We cannot use `abs.bf16` because it is not IEEE 754 compliant, see docs
  return _CUDA_VSTD::__fabs_impl(__x);
}
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp8_e4m3 fabs(__nv_fp8_e4m3 __x) noexcept
{
  return _CUDA_VSTD::__fabs_impl(__x);
}
#endif // _CCCL_HAS_NVFP8_E4M#()

#if _CCCL_HAS_NVFP8_E5M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp8_e5m2 fabs(__nv_fp8_e5m2 __x) noexcept
{
  return _CUDA_VSTD::__fabs_impl(__x);
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp8_e8m0 fabs(__nv_fp8_e8m0 __x) noexcept
{
  return __x;
}
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp6_e2m3 fabs(__nv_fp6_e2m3 __x) noexcept
{
  return _CUDA_VSTD::__fabs_impl(__x);
}
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp6_e3m2 fabs(__nv_fp6_e3m2 __x) noexcept
{
  return _CUDA_VSTD::__fabs_impl(__x);
}
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp4_e2m1 fabs(__nv_fp4_e2m1 __x) noexcept
{
  return _CUDA_VSTD::__fabs_impl(__x);
}
#endif // _CCCL_HAS_NVFP4_E2M1()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double fabs(_Tp __val) noexcept
{
  return _CUDA_VSTD::fabs(static_cast<double>(__val));
}

// abs

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float abs(float __val) noexcept
{
  return _CUDA_VSTD::fabsf(__val);
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double abs(double __val) noexcept
{
  return _CUDA_VSTD::fabs(__val);
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double abs(long double __val) noexcept
{
  return _CUDA_VSTD::fabsl(__val);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __half abs(__half __x) noexcept
{
  return _CUDA_VSTD::fabs(__x);
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_bfloat16 abs(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::fabs(__x);
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp8_e4m3 abs(__nv_fp8_e4m3 __x) noexcept
{
  return _CUDA_VSTD::fabs(__x);
}
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp8_e5m2 abs(__nv_fp8_e5m2 __x) noexcept
{
  return _CUDA_VSTD::fabs(__x);
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp8_e8m0 abs(__nv_fp8_e8m0 __x) noexcept
{
  return _CUDA_VSTD::fabs(__x);
}
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp6_e2m3 abs(__nv_fp6_e2m3 __x) noexcept
{
  return _CUDA_VSTD::fabs(__x);
}
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp6_e3m2 abs(__nv_fp6_e3m2 __x) noexcept
{
  return _CUDA_VSTD::fabs(__x);
}
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp4_e2m1 abs(__nv_fp4_e2m1 __x) noexcept
{
  return _CUDA_VSTD::fabs(__x);
}
#endif // _CCCL_HAS_NVFP4_E2M1()

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_ABS_H
