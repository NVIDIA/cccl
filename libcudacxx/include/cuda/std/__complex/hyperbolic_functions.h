//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
#include <cuda/std/numbers>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// sinh

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> sinh(const complex<_Tp>& __x) noexcept
{
  // Need to distinguish +-0.0, use signbit rather than >
  const bool __x_neg = ::cuda::std::signbit(__x.real());

  const _Tp __imag_x = __x_neg ? -__x.imag() : __x.imag();
  _Tp __real_x_abs   = ::cuda::std::fabs(__x.real());

  // Purge some special cases that don't pass through correctly:
  if ((!::cuda::std::isfinite(__imag_x))
      && ((__real_x_abs == _Tp{0}) || (__real_x_abs == numeric_limits<_Tp>::infinity())))
  {
    return complex<_Tp>{__x.real(), numeric_limits<_Tp>::quiet_NaN()};
  }
  if (::cuda::std::isnan(__real_x_abs) && (__imag_x == _Tp{0}))
  {
    return __x;
  }

  const auto [__sin, __cos] = ::cuda::sincos(__imag_x);

  // Using ans = {sinh(re) * cos(im), cosh(re) * sin(im)}
  // results in intermediate overflows where eg cosh(re) overflows, but cosh(re) * sin(im) does not.
  // We scale our inputs to avoid this.

  // Rather than examining exponents, we save computation by pre-selecting 2
  // separate intervals that cover our problematic overflow interval.
  // Use sinh(x) ~= cosh(x) ~= exp(x)/2 for large x.
  // Need two intervals to ensure the reduced values do not break the sinh(x) ~= cosh(x) ~= exp(x)/2 estimate.

  // We need eg cosh(x) * sin(y) to not have intermediate overflow. We find bounds where the final result will overflow
  // no matter the (non-zero) value of y.
  // sin(y) can attain the smallest denormal value, we calculate the largest value of x where cosh(x) * sin(y) is still
  // finite.
  // This is attained at acosh(x) * MIN_DENORMAL_FLOAT = MAX_FLOAT.
  // aka x ~= (MAX_FLOAT_EXPONENT - MIN_DENORMAL_FLOAT_EXPONENT + 1) * log(2)
  // aka x ~= (1024 + 1074 + 1) * log(2) in double for example.
  // ~= 1454.916... for double
  // ~= 192.695.... for float

  // we also find where sinh(x) ~=cosh(x) ~= exp(x)/2 overflows normally..
  // This is at exp(x)/2 = MAX_FLOAT,
  // aka x ~= (MAX_FLOAT_EXPONENT + 1) * log(2)
  // This is:
  // ~= (1024 + 1) * log(2) ~= 710.475... for double
  // ~= (128  + 1) * log(2) ~= 89.416...  for float

  constexpr int32_t __mant_nbits = __fp_mant_nbits_v<__fp_format_of_v<_Tp>>;
  constexpr int32_t __exp_max    = __fp_exp_max_v<__fp_format_of_v<_Tp>>;

  // Massage these values a little to cover the edges.
  // These values are small enough we can cast to int and back as a poor mans constexpr floor.
  // (we want __ans_always_overflow_bound to be slightly larger than the real value, we add 1)
  constexpr _Tp __ln2 = __numbers<_Tp>::__ln2();

  constexpr _Tp __sinh_cosh_overflow_bound = _Tp{int(_Tp{__exp_max + 2} * __ln2)};
  constexpr _Tp __ans_always_overflow_bound =
    _Tp{int(_Tp{(__exp_max + 1) + (__exp_max + __mant_nbits - 1)} * __ln2) + 1};

  // We split the above interval into two. Get the midpoint:
  constexpr _Tp __overflow_interval_midpoint = (__sinh_cosh_overflow_bound + __ans_always_overflow_bound) * _Tp{0.5};

  // Our values look like this for double/float:
  // double:
  // __sinh_cosh_overflow_bound: 710.000000
  // __ans_always_overflow_bound: 1455.000000
  // __overflow_interval_midpoint: 1082.500000
  // float:
  // __sinh_cosh_overflow_bound: 89.000000
  // __ans_always_overflow_bound: 193.000000
  // __overflow_interval_midpoint: 141.000000

  _Tp __in_scale  = _Tp{0};
  _Tp __out_scale = _Tp{1};

  if (__real_x_abs >= __sinh_cosh_overflow_bound)
  {
    // Here sinh and cosh will overflow, but the final result may not.
    // Get scale factors for the first half of the interval:
    __in_scale = __overflow_interval_midpoint - __sinh_cosh_overflow_bound;

    // Here what we really want is an accurate constexpr exp(__in_scale * 0.5).
    // Calling exp loses too much accuracy (as well as relying on it being evaluated at compile time).
    // Set the value manually for now:
    // __out_scale = ::cuda::std::exp(__in_scale * _Tp{0.5});
    __out_scale = is_same_v<_Tp, double> ? static_cast<_Tp>(7.715201168634011507e80) : static_cast<_Tp>(1.95729609e11f);

    // Other half of the interval:
    if (__real_x_abs > __overflow_interval_midpoint)
    {
      __in_scale = __ans_always_overflow_bound - __sinh_cosh_overflow_bound;

      // Same as above, set the value manually for the moment:
      // __out_scale = ::cuda::std::exp(__in_scale * _Tp{0.5});
      __out_scale =
        is_same_v<_Tp, double> ? static_cast<_Tp>(5.952432907249161686e161) : static_cast<_Tp>(3.831008e22f);
    }

    if (__real_x_abs > __ans_always_overflow_bound)
    {
      // Here the final answer will overflow, with one exception.
      // If (imag == 0.0), we would end up later multiplying inf * 0.0 to get NaN,
      // while the correct answer is 0.0.
      // This seems be be the best place to correct this for (as opposed to an early return).
      // We clamp the value to a value that still always overflows the final answer, while the intermediate
      // calculation with our scaling will not. (real == inf) has already been purged.
      __real_x_abs = __ans_always_overflow_bound;
    }
  }

  const _Tp __cosh_re_scaled = ::cuda::std::cosh(__real_x_abs - __in_scale);
  const _Tp __sinh_re_scaled = ::cuda::std::sinh(__real_x_abs - __in_scale);

  // Order of multiplication matters for under/overflow:
  const _Tp __ans_re = (__sinh_re_scaled * (__cos * __out_scale)) * __out_scale;
  const _Tp __ans_im = (__cosh_re_scaled * (__sin * __out_scale)) * __out_scale;

  const complex __ans_abs{__ans_re, __ans_im};
  return __x_neg ? -__ans_abs : __ans_abs;
}

// We have performance issues with extended floating point types
#if _LIBCUDACXX_HAS_NVFP16()
template <>
_CCCL_API inline complex<__half> sinh(const complex<__half>& __x) noexcept
{
  return complex<__half>{::cuda::std::sinh(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
template <>
_CCCL_API inline complex<__nv_bfloat16> sinh(const complex<__nv_bfloat16>& __x) noexcept
{
  return complex<__nv_bfloat16>{::cuda::std::sinh(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVBF16()

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
