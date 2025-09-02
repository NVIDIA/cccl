//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___COMPLEX_MATH_H
#define _CUDA_STD___COMPLEX_MATH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/hypot.h>
#include <cuda/std/__cmath/inverse_trigonometric_functions.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__cmath/signbit.h>
#include <cuda/std/__cmath/trigonometric_functions.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/__complex/vector_support.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// abs

template <class _Tp>
[[nodiscard]] _CCCL_API inline _Tp abs(const complex<_Tp>& __c)
{
  return ::cuda::std::hypot(__c.real(), __c.imag());
}

// norm

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp norm(const complex<_Tp>& __c)
{
  if (::cuda::std::isinf(__c.real()))
  {
    return ::cuda::std::abs(__c.real());
  }
  if (::cuda::std::isinf(__c.imag()))
  {
    return ::cuda::std::abs(__c.imag());
  }
  return __c.real() * __c.real() + __c.imag() * __c.imag();
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr __cccl_complex_value_type<_Tp> norm(_Tp __re)
{
  return static_cast<__cccl_complex_value_type<_Tp>>(__re) * __re;
}

// conj

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> conj(const complex<_Tp>& __c)
{
  return complex<_Tp>(__c.real(), -__c.imag());
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr __cccl_complex_complex_type<_Tp> conj(_Tp __re)
{
  return __cccl_complex_complex_type<_Tp>(__re);
}

// proj

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> proj(const complex<_Tp>& __c)
{
  complex<_Tp> __r = __c;
  if (::cuda::std::isinf(__c.real()) || ::cuda::std::isinf(__c.imag()))
  {
    __r = complex<_Tp>(numeric_limits<_Tp>::infinity(), ::cuda::std::copysign(_Tp(0), __c.imag()));
  }
  return __r;
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((is_floating_point_v<_Tp> || __is_extended_floating_point_v<_Tp>) )
[[nodiscard]] _CCCL_API inline __cccl_complex_complex_type<_Tp> proj(_Tp __re)
{
  if (::cuda::std::isinf(__re))
  {
    __re = ::cuda::std::abs(__re);
  }
  return complex<_Tp>(__re);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_integral_v<_Tp>)
[[nodiscard]] _CCCL_API inline __cccl_complex_complex_type<_Tp> proj(_Tp __re)
{
  return __cccl_complex_complex_type<_Tp>(__re);
}

// polar

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> polar(const _Tp& __rho, const _Tp& __theta = _Tp())
{
  if (::cuda::std::isnan(__rho) || ::cuda::std::signbit(__rho))
  {
    return complex<_Tp>(numeric_limits<_Tp>::quiet_NaN(), numeric_limits<_Tp>::quiet_NaN());
  }
  if (::cuda::std::isnan(__theta))
  {
    if (::cuda::std::isinf(__rho))
    {
      return complex<_Tp>(__rho, __theta);
    }
    return complex<_Tp>(__theta, __theta);
  }
  if (::cuda::std::isinf(__theta))
  {
    if (::cuda::std::isinf(__rho))
    {
      return complex<_Tp>(__rho, numeric_limits<_Tp>::quiet_NaN());
    }
    return complex<_Tp>(numeric_limits<_Tp>::quiet_NaN(), numeric_limits<_Tp>::quiet_NaN());
  }
  _Tp __x = __rho * ::cuda::std::cos(__theta);
  if (::cuda::std::isnan(__x))
  {
    __x = 0;
  }
  _Tp __y = __rho * ::cuda::std::sin(__theta);
  if (::cuda::std::isnan(__y))
  {
    __y = 0;
  }
  return complex<_Tp>(__x, __y);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___COMPLEX_MATH_H
