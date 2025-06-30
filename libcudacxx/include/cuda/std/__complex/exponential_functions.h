//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_EXPONENTIAL_FUNCTIONS_H
#define _LIBCUDACXX___COMPLEX_EXPONENTIAL_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/copysign.h>
#include <cuda/std/__cmath/isfinite.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__cmath/trigonometric_functions.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/__complex/exponential_functions.h>
#include <cuda/std/__complex/logarithms.h>
#include <cuda/std/__complex/vector_support.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// exp

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> exp(const complex<_Tp>& __x)
{
  _Tp __i = __x.imag();
  if (__i == _Tp(0))
  {
    return complex<_Tp>(_CUDA_VSTD::exp(__x.real()), _CUDA_VSTD::copysign(_Tp(0), __x.imag()));
  }
  if (_CUDA_VSTD::isinf(__x.real()))
  {
    if (__x.real() < _Tp(0))
    {
      if (!_CUDA_VSTD::isfinite(__i))
      {
        __i = _Tp(1);
      }
    }
    else if (__i == _Tp(0) || !_CUDA_VSTD::isfinite(__i))
    {
      if (_CUDA_VSTD::isinf(__i))
      {
        __i = numeric_limits<_Tp>::quiet_NaN();
      }
      return complex<_Tp>(__x.real(), __i);
    }
  }
  _Tp __e = _CUDA_VSTD::exp(__x.real());
  return complex<_Tp>(__e * _CUDA_VSTD::cos(__i), __e * _CUDA_VSTD::sin(__i));
}

// pow

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> pow(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
  return _CUDA_VSTD::exp(__y * _CUDA_VSTD::log(__x));
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244)

template <class _Tp, class _Up>
[[nodiscard]] _CCCL_API inline complex<common_type_t<_Tp, _Up>> pow(const complex<_Tp>& __x, const complex<_Up>& __y)
{
  using __result_type = complex<common_type_t<_Tp, _Up>>;
  return _CUDA_VSTD::pow(__result_type(__x), __result_type(__y));
}

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES((!__is_complex_v<_Up>) )
[[nodiscard]] _CCCL_API inline complex<common_type_t<_Tp, _Up>> pow(const complex<_Tp>& __x, const _Up& __y)
{
  using __result_type = complex<common_type_t<_Tp, _Up>>;
  return _CUDA_VSTD::pow(__result_type(__x), __result_type(__y));
}

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES((!__is_complex_v<_Tp>) )
[[nodiscard]] _CCCL_API inline complex<common_type_t<_Tp, _Up>> pow(const _Tp& __x, const complex<_Up>& __y)
{
  using __result_type = complex<common_type_t<_Tp, _Up>>;
  return _CUDA_VSTD::pow(__result_type(__x, 0), __result_type(__y));
}

_CCCL_DIAG_POP

// __sqr, computes pow(x, 2)

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> __sqr(const complex<_Tp>& __x)
{
  return complex<_Tp>((__x.real() - __x.imag()) * (__x.real() + __x.imag()), _Tp(2) * __x.real() * __x.imag());
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___COMPLEX_EXPONENTIAL_FUNCTIONS_H
