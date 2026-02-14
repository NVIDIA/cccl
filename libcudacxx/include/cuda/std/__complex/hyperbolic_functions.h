//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___COMPLEX_HYPERBOLIC_FUNCTIONS_H
#define _CUDA_STD___COMPLEX_HYPERBOLIC_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/sincos.h>
#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/copysign.h>
#include <cuda/std/__cmath/hyperbolic_functions.h>
#include <cuda/std/__cmath/isfinite.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__cmath/trigonometric_functions.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// sinh

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> sinh(const complex<_Tp>& __x)
{
  if (::cuda::std::isinf(__x.real()) && !::cuda::std::isfinite(__x.imag()))
  {
    return complex<_Tp>(__x.real(), numeric_limits<_Tp>::quiet_NaN());
  }
  if (__x.real() == _Tp(0) && !::cuda::std::isfinite(__x.imag()))
  {
    return complex<_Tp>(__x.real(), numeric_limits<_Tp>::quiet_NaN());
  }
  if (__x.imag() == _Tp(0) && !::cuda::std::isfinite(__x.real()))
  {
    return __x;
  }
  const auto [__im_sin, __im_cos] = ::cuda::sincos(__x.imag());
  return complex<_Tp>(::cuda::std::sinh(__x.real()) * __im_cos, ::cuda::std::cosh(__x.real()) * __im_sin);
}

// cosh

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> cosh(const complex<_Tp>& __x)
{
  if (::cuda::std::isinf(__x.real()) && !::cuda::std::isfinite(__x.imag()))
  {
    return complex<_Tp>(::cuda::std::abs(__x.real()), numeric_limits<_Tp>::quiet_NaN());
  }
  if (__x.real() == _Tp(0) && !::cuda::std::isfinite(__x.imag()))
  {
    return complex<_Tp>(numeric_limits<_Tp>::quiet_NaN(), __x.real());
  }
  if (__x.real() == _Tp(0) && __x.imag() == _Tp(0))
  {
    return complex<_Tp>(_Tp(1), __x.imag());
  }
  if (__x.imag() == _Tp(0) && !::cuda::std::isfinite(__x.real()))
  {
    return complex<_Tp>(::cuda::std::abs(__x.real()), __x.imag());
  }
  const auto [__im_sin, __im_cos] = ::cuda::sincos(__x.imag());
  return complex<_Tp>(::cuda::std::cosh(__x.real()) * __im_cos, ::cuda::std::sinh(__x.real()) * __im_sin);
}

// tanh

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> tanh(const complex<_Tp>& __x)
{
  if (::cuda::std::isinf(__x.real()))
  {
    if (!::cuda::std::isfinite(__x.imag()))
    {
      return complex<_Tp>(::cuda::std::copysign(_Tp(1), __x.real()), _Tp(0));
    }
    return complex<_Tp>(::cuda::std::copysign(_Tp(1), __x.real()),
                        ::cuda::std::copysign(_Tp(0), ::cuda::std::sin(_Tp(2) * __x.imag())));
  }
  if (::cuda::std::isnan(__x.real()) && __x.imag() == _Tp(0))
  {
    return __x;
  }
  _Tp __2r(_Tp(2) * __x.real());
  _Tp __2i(_Tp(2) * __x.imag());
  const auto [__2i_sin, __2i_cos] = ::cuda::sincos(__2i);
  _Tp __d(::cuda::std::cosh(__2r) + __2i_cos);
  _Tp __2rsh(::cuda::std::sinh(__2r));
  if (::cuda::std::isinf(__2rsh) && ::cuda::std::isinf(__d))
  {
    return complex<_Tp>(__2rsh > _Tp(0) ? _Tp(1) : _Tp(-1), __2i > _Tp(0) ? _Tp(0) : _Tp(-0.));
  }
  return complex<_Tp>(__2rsh / __d, __2i_sin / __d);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___COMPLEX_HYPERBOLIC_FUNCTIONS_H
