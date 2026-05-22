//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___COMPLEX_INVERSE_TRIGONOMETRIC_FUNCTIONS_H
#define _CUDA_STD___COMPLEX_INVERSE_TRIGONOMETRIC_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/fma.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__cmath/signbit.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/__complex/inverse_hyperbolic_functions.h>
#include <cuda/std/__complex/nvbf16.h>
#include <cuda/std/__complex/nvfp16.h>
#include <cuda/std/limits>
#include <cuda/std/numbers>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// asin

template <class _Tp>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline complex<_Tp> asin(const complex<_Tp>& __x)
{
  complex<_Tp> __z = ::cuda::std::asinh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// acos

template <class _Tp>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline complex<_Tp> acos(const complex<_Tp>& __x)
{
  // Get both real and imag parts >= +0.0
  const bool __real_neg = ::cuda::std::signbit(__x.real());
  const bool __imag_neg = ::cuda::std::signbit(__x.imag());

  const complex<_Tp> __x_first_quadrant{::cuda::std::fabs(__x.real()), ::cuda::std::fabs(__x.imag())};

  const complex<_Tp> __acosh_out = ::cuda::std::acosh(__x_first_quadrant);

  _Tp __ans_real = __acosh_out.imag();
  _Tp __ans_imag = -__acosh_out.real();

  if (__real_neg)
  {
    // Values that multiply to an extra-accurate value of pi.
    constexpr _Tp __pi_hi = is_same_v<_Tp, float> ? 1.866378903f : static_cast<_Tp>(1.56226695361364598113596002804);
    constexpr _Tp __pi_lo = is_same_v<_Tp, float> ? 1.683255553f : static_cast<_Tp>(2.01091922627118435684678843245);
    __ans_real            = ::cuda::std::fma(__pi_hi, __pi_lo, -__ans_real);
  }

  if (__imag_neg)
  {
    __ans_imag = -__ans_imag;
  }

  return complex<_Tp>{__ans_real, __ans_imag};
}

// We have performance issues with some trigonometric functions with extended floating point types
#if _LIBCUDACXX_HAS_NVFP16()
template <>
_CCCL_HOST_DEVICE_API inline complex<__half> acos(const complex<__half>& __x)
{
  return complex<__half>{::cuda::std::acos(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
template <>
_CCCL_HOST_DEVICE_API inline complex<__nv_bfloat16> acos(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{::cuda::std::acos(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVBF16()

// atan

template <class _Tp>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline complex<_Tp> atan(const complex<_Tp>& __x)
{
  complex<_Tp> __z = ::cuda::std::atanh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___COMPLEX_INVERSE_TRIGONOMETRIC_FUNCTIONS_H
