//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_RSQRT_H
#define _CUDA___CMATH_RSQRT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/roots.h>
#include <cuda/std/__cmath/signbit.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _Tp>
[[nodiscard]] _CCCL_API inline _Tp __cccl_rsqrt_generic(_Tp __v) noexcept
{
  return _Tp{1.0} / _CUDA_VSTD::sqrt(__v);
}

//! @brief Computes the reciprocal square root of a value
//! @param __v The value to compute the reciprocal square root of.
//! @pre \p __v must be a floating-point or an integral type.
//! @return The reciprocal square root of the value.
//! @note rsqrt(+inf) returns +0.0
//!       rsqrt(+-0.0) returns +-inf
//!       rsqrt(x) returns NaN of unspecified sign if x < 0.0
//!       rsqrt(+-NaN) returns NaN of unspecified sign
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CUDA_VSTD::__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API inline _CUDA_VSTD::conditional_t<_CUDA_VSTD::is_integral_v<_Tp>, double, _Tp>
rsqrt(_Tp __v) noexcept
{
  if constexpr (_CUDA_VSTD::is_integral_v<_Tp>)
  {
    return ::cuda::rsqrt(static_cast<double>(__v));
  }
#if _LIBCUDACXX_HAS_NVFP16()
  else if constexpr (_CUDA_VSTD::is_same_v<_Tp, ::__half>)
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return ::hrsqrt(__v);),
                      (return ::__float2half(::cuda::__cccl_rsqrt_generic(::__half2float(__v)));))
  }
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  else if constexpr (_CUDA_VSTD::is_same_v<_Tp, ::__bfloat16>)
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return ::hrsqrt(__v);),
                      (return ::__float2bfloat16(::cuda::__cccl_rsqrt_generic(::__bfloat162float(__v)));))
  }
#endif // _LIBCUDACXX_HAS_NVBF16()
  else if constexpr (_CUDA_VSTD::is_same_v<_Tp, float>)
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::rsqrtf(__v);), (return ::cuda::__cccl_rsqrt_generic(__v);))
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Tp, double>)
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::rsqrt(__v);), (return ::cuda::__cccl_rsqrt_generic(__v);))
  }
  else
  {
    return ::cuda::__cccl_rsqrt_generic(__v);
  }
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_RSQRT_H
