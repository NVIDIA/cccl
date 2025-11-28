//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___COMPLEX_INVERSE_HYPERBOLIC_FUNCTIONS_H
#define _CUDA_STD___COMPLEX_INVERSE_HYPERBOLIC_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/copysign.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__cmath/trigonometric_functions.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/__complex/exponential_functions.h>
#include <cuda/std/__complex/logarithms.h>
#include <cuda/std/__complex/nvbf16.h>
#include <cuda/std/__complex/nvfp16.h>
#include <cuda/std/__complex/roots.h>
#include <cuda/std/limits>
#include <cuda/std/numbers>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Specialization of 1/sqrt(x), uses device-only rsqrtf/rsqrt when viable.
// Sometimes compilers can optimize 1/sqrt(x) so we don't suffer the slowdown of the divide.
// It can happen that the double 1/sqrt() on device gets optimized better than CUDA rsqrt(),
// but it depends on the calling function. We needs to optimize on an individual basis.
template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_FORCEINLINE _Tp __internal_rsqrt_inverse_hyperbloic(_Tp __x) noexcept
{
#if _CCCL_CUDA_COMPILATION()
  if constexpr (is_same_v<_Tp, float>)
  {
    NV_IF_TARGET(NV_IS_DEVICE, (return ::rsqrtf(__x);))
  }
  if constexpr (is_same_v<_Tp, double>)
  {
    NV_IF_TARGET(NV_IS_DEVICE, (return ::rsqrt(__x);))
  }
#endif // _CCCL_CUDA_COMPILATION()
  return _Tp{1} / ::cuda::std::sqrt(__x);
}

template <class _Tp>
struct _CCCL_ALIGNAS(2 * sizeof(_Tp)) __cccl_asinh_sqrt_return_hilo
{
  _Tp __hi;
  _Tp __lo;
};

// An unsafe sqrt(_Tp + _Tp) extended precision sqrt.
template <typename _Tp>
[[nodiscard]] _CCCL_API _CCCL_FORCEINLINE __cccl_asinh_sqrt_return_hilo<_Tp>
__internal_double_Tp_sqrt_unsafe(_Tp __hi, _Tp __lo) noexcept
{
  // rsqrt
  const _Tp __initial_guess = __internal_rsqrt_inverse_hyperbloic<_Tp>(__hi);

  // Newton-Raphson, assume we have a reasonable rsqrt above and
  // that we don't need to update the first term (__initial_guess).
  //     x_(n+1) = x_n - 0.5*x_n*((__hi + __lo)(x_n^2) - 1)

  // __initial_guess^2:
  const _Tp __init_sq_hi = __initial_guess * __initial_guess;
  const _Tp __init_sq_lo = ::cuda::std::fma(__initial_guess, __initial_guess, -__init_sq_hi);

  // Times (__hi + __lo).
  // We need to add the -1.0 here in an fma, or different compilers (eg host vs device)
  // optimize differently and give (sometimes very) different results.
  const _Tp _hi_hi_hi = ::cuda::std::fma(__hi, __init_sq_hi, _Tp{-1.0});
  // Low part not needed for fp32/fp64, but might be if this is extended to other types:
  // _Tp _hi_hi_lo = fma(__hi, __init_sq_hi, -_hi_hi_hi);

  // Add all terms
  const _Tp __full_term = _hi_hi_hi + (::cuda::std::fma(__lo, __init_sq_hi, __hi * __init_sq_lo) /*+ _hi_hi_lo*/);

  const _Tp __correction_term = _Tp{-0.5} * __initial_guess * __full_term;

  // rsqrt(hi + lo) is now estimated well by (__initial_guess + __correction_term)
  // Multiply everything by (hi + lo) to get sqrt(hi + lo)
  const _Tp __ans_hi_hi = __hi * __initial_guess;
  _Tp __ans_hi_lo       = ::cuda::std::fma(__hi, __initial_guess, -__ans_hi_hi);

  // All terms needed, allow the compiler to pick which way to
  // optimize this to fma, same accuracy.
  __ans_hi_lo += __initial_guess * __lo + __correction_term * __hi;

  return __cccl_asinh_sqrt_return_hilo<_Tp>{__ans_hi_hi, __ans_hi_lo};
}

// asinh

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> asinh(const complex<_Tp>& __x) noexcept
{
  // Uint of the same size as our fp type.
  using __uint_t = __fp_storage_of_t<_Tp>;

  constexpr int32_t __mant_nbits = __fp_mant_nbits_v<__fp_format_of_v<_Tp>>;
  constexpr int32_t __exp_max    = __fp_exp_max_v<__fp_format_of_v<_Tp>>;
  constexpr int32_t __exp_bias   = __fp_exp_bias_v<__fp_format_of_v<_Tp>>;

  constexpr _Tp __pi  = __numbers<_Tp>::__pi();
  constexpr _Tp __ln2 = __numbers<_Tp>::__ln2();

  _Tp __realx = ::cuda::std::fabs(__x.real());
  _Tp __imagx = ::cuda::std::fabs(__x.imag());

  // Special cases that do not pass through:
  if (!::cuda::std::isfinite(__realx) || !::cuda::std::isfinite(__imagx))
  {
    // If z is (x,+inf) (for any positive finite x), the result is (+inf, pi/2)
    if (::cuda::std::isfinite(__realx) && ::cuda::std::isinf(__imagx))
    {
      return complex<_Tp>(::cuda::std::copysign(numeric_limits<_Tp>::infinity(), __x.real()),
                          ::cuda::std::copysign(_Tp{0.5} * __pi, __x.imag()));
    }

    // If z is (+inf,y) (for any positive finite y), the result is (+inf,+0)
    if (::cuda::std::isinf(__realx) && ::cuda::std::isfinite(__imagx))
    {
      return complex<_Tp>(::cuda::std::copysign(numeric_limits<_Tp>::infinity(), __x.real()),
                          ::cuda::std::copysign(_Tp{0}, __x.imag()));
    }

    // If z is (+inf,+inf), the result is (+inf, pi/4)
    if (::cuda::std::isinf(__realx) && ::cuda::std::isinf(__imagx))
    {
      return complex<_Tp>(::cuda::std::copysign(numeric_limits<_Tp>::infinity(), __x.real()),
                          ::cuda::std::copysign(_Tp{0.25} * __pi, __x.imag()));
    }

    // If z is (+inf,NaN), the result is (+inf,NaN)
    if (::cuda::std::isinf(__realx) && ::cuda::std::isnan(__imagx))
    {
      return __x;
    }

    // If z is (NaN,+0), the result is (NaN,+0)
    if (::cuda::std::isnan(__realx) && (__imagx == _Tp{0}))
    {
      return __x;
    }

    // If z is (NaN,+inf), the result is (±INF,NaN) (the sign of the real part is unspecified)
    if (::cuda::std::isnan(__realx) && ::cuda::std::isinf(__imagx))
    {
      return complex<_Tp>(__x.imag(), numeric_limits<_Tp>::quiet_NaN());
    }
  }

  // Special case that for various reasons does not pass
  // easily through the algorithm below:
  if ((__realx == _Tp{0}) && (__imagx == _Tp{1}))
  {
    return complex<_Tp>(__x.real(), ::cuda::std::copysign(_Tp{0.5} * __pi, __x.imag()));
  }

  // It is a little involved to account for large inputs in an inlined-into-existing-code fashion.
  // The easiest place to account for large values appears to be here at the start.
  // Use asinh(x) ~ log(2x) for large x.
  // Get the largest exponent that passes through the algorithm without issue.
  // This is ~(max_exponent / 4), with a small bias to make sure edge cases get caught
  // ~254 for double, ~30 for float
  constexpr int32_t __max_allowed_exponent = (__exp_max / 4) - 2;
  constexpr __uint_t __max_allowed_val_as_uint =
    (__uint_t(__max_allowed_exponent + __exp_bias) << __mant_nbits) | __fp_explicit_bit_mask_of_v<_Tp>;

  //  Check if the largest component of __x is > 2^__max_allowed_exponent:
  _Tp __x_big_factor = _Tp{0};
  const _Tp __max    = ::cuda::std::fmax(__realx, __imagx);
  const bool __x_big = ::cuda::std::__fp_get_storage(__max) > __max_allowed_val_as_uint;

  if (__x_big)
  {
    // We need __max to be <= ~(2^__max_allowed_exponent),
    // but not small enough that the asinh(x) ~ log(2x) estimate does
    // not break down. We are not able to reduce this with a single simple reduction,
    // so we do a fast/inlined frexp/ldexp:
    const int32_t __exp_biased = static_cast<int32_t>(::cuda::std::__fp_get_storage(__max) >> __mant_nbits);

    // Get a factor such that (__max * __exp_mul_factor) <= __max_allowed_exponent
    const __uint_t __exp_reduce_factor =
      (__uint_t((2 * __exp_max) + __max_allowed_exponent - __exp_biased) << __mant_nbits)
      | __fp_explicit_bit_mask_of_v<_Tp>;
    const _Tp __exp_mul_factor = ::cuda::std::__fp_from_storage<_Tp>(__exp_reduce_factor);

    // Scale down to a working range.
    __realx *= __exp_mul_factor;
    __imagx *= __exp_mul_factor;

    __x_big_factor = static_cast<_Tp>((__exp_biased - __exp_max) - __max_allowed_exponent) * __ln2;
  }

  // let compiler pick which way to fma this, accuracy stays the same.
  const _Tp __diffx_m1 = __realx * __realx - (__imagx - _Tp{1}) * (__imagx + _Tp{1});

  // Get the real and imag parts of |sqrt(z^2 + 1)|^2
  // This equates to calculating:
  //     sqrt((re*re + (im + 1.0)*(im + 1.0))*(re*re + (im - 1.0)*(im - 1.0)));
  // Where we need both the term inside the sqrt in extended precision, as well
  // as evaluation the sqrt itself in extended precision.

  // Get re^2 + im^2 + 1 in extended precision.
  // The low part of re^2 doesn't seem to matter.
  const _Tp __imagx_sq_hi = __imagx * __imagx;
  const _Tp __imagx_sq_lo = ::cuda::std::fma(__imagx, __imagx, -__imagx_sq_hi);

  const _Tp __x_abs_sq_hi = __imagx_sq_hi;
  const _Tp __x_abs_sq_lo = ::cuda::std::fma(__realx, __realx, __imagx_sq_lo);

  // Add one:
  const _Tp __x_abs_sq_p1_hi = (__x_abs_sq_hi + _Tp{1});
  const _Tp __x_abs_sq_p1_lo = __x_abs_sq_lo - ((__x_abs_sq_p1_hi - _Tp{1}) - __x_abs_sq_hi);

  // square:
  const _Tp __x_abs_sq_p1_sq_hi = __x_abs_sq_p1_hi * __x_abs_sq_p1_hi;
  _Tp __x_abs_sq_p1_sq_lo       = ::cuda::std::fma(__x_abs_sq_p1_hi, __x_abs_sq_p1_hi, -__x_abs_sq_p1_sq_hi);

  // Add in the lower square terms, all needed
  __x_abs_sq_p1_sq_lo =
    ::cuda::std::fma(__x_abs_sq_p1_lo, (_Tp{2} * __x_abs_sq_p1_hi + __x_abs_sq_p1_lo), __x_abs_sq_p1_sq_lo);

  // Get __x_abs_sq_p1_sq_hi/lo - 4.0*__imagx_sq_hi/lo:
  // Subtract high parts:
  _Tp __inner_most_term_hi = __x_abs_sq_p1_sq_hi - _Tp{4} * __imagx_sq_hi;
  _Tp __inner_most_term_lo = ((__x_abs_sq_p1_sq_hi - __inner_most_term_hi) - _Tp{4} * __imagx_sq_hi);
  // lo parts, all needed:
  __inner_most_term_lo += __x_abs_sq_p1_sq_lo - _Tp{4} * __imagx_sq_lo;

  // We can have some slightly bad cases here due to catastrohip cancellation that can't be fixed easily.
  // We still need to to the extended-sqrt on these values, so we fix them now.
  // It occurs around "real ~= small" and "imag ~= (1 - small)", and imag < 1.
  // Worked out through targeted testing on fp64 and fp32.
  _Tp __realx_small_bound = _Tp{1.0e-13};
  _Tp __imagx_close_bound = _Tp{0.98};

  if constexpr (is_same_v<_Tp, float>)
  {
    __realx_small_bound = _Tp{1.0e-5f};
    __imagx_close_bound = _Tp{0.9f};
  }

  if ((__realx < __realx_small_bound) && (__imagx_close_bound < __imagx) && (__imagx <= _Tp{1}))
  {
    // Get (real^2 + (1 - imag)^2) * (real^2 + (1 + imag)^2) in double-double:
    // term1 = (real^2 + (1 - imag)^2)
    const _Tp __term1_hi = (_Tp{1} - __imagx) * (_Tp{1} - __imagx);
    const _Tp __term1_lo = ::cuda::std::fma(_Tp{1} - __imagx, _Tp{1} - __imagx, -__term1_hi) + __realx * __realx;

    // Need (1.0 + __imagx)^2 with nearly full accuracy.
    const _Tp __term2_sum_hi = (_Tp{1} + __imagx);
    const _Tp __term2_sum_lo = ((_Tp{1} - __term2_sum_hi) + __imagx);

    const _Tp __term2_sq_hi = __term2_sum_hi * __term2_sum_hi;
    _Tp __term2_sq_lo       = ::cuda::std::fma(__term2_sum_hi, __term2_sum_hi, -__term2_sq_hi);
    __term2_sq_lo += _Tp{2} * __term2_sum_hi * __term2_sum_lo;

    // Multiple __term1_hi/lo and __term2_sq_hi/lo:
    __inner_most_term_hi = __term1_hi * __term2_sq_hi;
    __inner_most_term_lo = ::cuda::std::fma(__term1_hi, __term2_sq_hi, -__inner_most_term_hi);
    // All needed:
    __inner_most_term_lo += __term1_hi * __term2_sq_lo + __term1_lo * __term2_sq_hi;
  }

  // Normalize the above (assumed in the extended-sqrt function):
  const _Tp __norm_hi  = __inner_most_term_hi + __inner_most_term_lo;
  const _Tp __norm_lo  = -((__norm_hi - __inner_most_term_hi) - __inner_most_term_lo);
  __inner_most_term_hi = __norm_hi;
  __inner_most_term_lo = __norm_lo;

  // Extended sqrt function:
  // (__extended_sqrt_hi + __extended_sqrt_lo) = sqrt(__inner_most_term_hi + __inner_most_term_lo)
  const __cccl_asinh_sqrt_return_hilo<_Tp> __extended_sqrt_hilo =
    ::cuda::std::__internal_double_Tp_sqrt_unsafe<_Tp>(__inner_most_term_hi, __inner_most_term_lo);

  _Tp __extended_sqrt_hi = __extended_sqrt_hilo.__hi;
  _Tp __extended_sqrt_lo = __extended_sqrt_hilo.__lo;

  // 0.0, and some very particular values, do not survive this unsafe sqrt function.
  // This case occurs when (1 + x^2) is zero or denormal. (and rsqrt(x)*rsqrt(x) become inf).
  constexpr __uint_t __min_normal_bits = __uint_t{0x1} << __mant_nbits;
  const _Tp __min_normal               = ::cuda::std::__fp_from_storage<_Tp>(__min_normal_bits);

  if (__inner_most_term_hi <= _Tp{2} * __min_normal)
  {
    __extended_sqrt_hi = _Tp{2} * __realx;
    __extended_sqrt_lo = _Tp{0};
  }

  // Get sqrt(0.5*(__extended_sqrt_hi + __diffx_m1))
  // This can result in catastrophic cancellation if __diffx_m1 < 0, in this case
  // We instead use the equivalent
  //     (__realx*__imagx) / sqrt(0.5*(__extended_sqrt_hi - __diffx_m1))

  const _Tp __inside_sqrt_term = _Tp{0.5} * (::cuda::std::fabs(__diffx_m1) + __extended_sqrt_hi);

  // Allow for rsqrt optimization:
  // We can have two slightly different paths depending on whether rsqrt is available
  // or not, aka are we on device or host.

  const _Tp __recip_sqrt = ::cuda::std::__internal_rsqrt_inverse_hyperbloic<_Tp>(__inside_sqrt_term);
  _Tp __pos_evaluation_real;

  // This reuses the sqrt calculated on CPU already in __recip_sqrt,
  // And gets sqrt quickly on device using the rsqrt already calculated.
#if _CCCL_CUDA_COMPILATION()
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (__pos_evaluation_real = (__recip_sqrt * __inside_sqrt_term);),
                    (__pos_evaluation_real = ::cuda::std::sqrt(__inside_sqrt_term);))
#else
  __pos_evaluation_real = ::cuda::std::sqrt(__inside_sqrt_term);
#endif // _CCCL_CUDA_COMPILATION()

  // Here, in a happy coincidence(?), we happen to intermediately calculate an accurate
  // return value for the real part of the answer in the case that __realx is small,
  // as you would obtain from the Taylor expansion of asinh. (~ real/sqrt(1 - imag^2)).
  // The following parts of the calculation result in bad catastrophic cancellation for
  // this case, so we save this intermediate value:
  const _Tp __small_x_real_return_val = __realx * __recip_sqrt;
  const _Tp __pos_evaluation_imag     = __imagx * __small_x_real_return_val;

  const _Tp __sqrt_real_part = (__diffx_m1 > _Tp{0}) ? __pos_evaluation_real : __pos_evaluation_imag;
  const _Tp __sqrt_imag_part = (__diffx_m1 > _Tp{0}) ? __pos_evaluation_imag : __pos_evaluation_real;

  // For an accurate log, we calculate |(__sqrt_real_part + i*__sqrt_imag_part)| - 1 and use log1p.
  // This can normally have bad catastrophic cancellation, however
  // we have a lot of retained enough accuracy to subtract fairly simply:
  const _Tp __m1  = __extended_sqrt_hi - _Tp{1};
  const _Tp __rem = -((__m1 + _Tp{1}) - __extended_sqrt_hi);

  __extended_sqrt_hi = __m1;
  __extended_sqrt_lo += __rem;

  // Final sum before sending it to log1p, all terms needed.
  // Add our sum via three terms, adding equally sized components.
  const _Tp __sum1 = (__x_abs_sq_hi + __extended_sqrt_hi);
  const _Tp __sum2 = _Tp{2} * (__realx * __sqrt_real_part + __imagx * __sqrt_imag_part);
  const _Tp __sum3 = (__extended_sqrt_lo + __x_abs_sq_lo);

  const _Tp __abs_sqrt_part_sq = __sum1 + (__sum2 + __sum3);

  const _Tp __atan2_input1 = __imagx + __sqrt_imag_part;
  const _Tp __atan2_input2 = __realx + __sqrt_real_part;

  _Tp __ans_real       = _Tp{0.5} * ::cuda::std::log1p(__abs_sqrt_part_sq);
  const _Tp __ans_imag = ::cuda::std::atan2(__atan2_input1, __atan2_input2);

  // The small |real| case, as mentioned above.
  // Bounds found by testing.
  _Tp __realx_small_bound_override = _Tp{2.220446e-16};

  if constexpr (is_same_v<_Tp, float>)
  {
    __realx_small_bound_override = _Tp{6.0e-08f};
  }

  if (__realx < __realx_small_bound_override && __imagx < _Tp{1})
  {
    __ans_real = __small_x_real_return_val;
  }

  __ans_real += __x_big_factor;

  // Copy signs back in
  return complex<_Tp>(::cuda::std::copysign(__ans_real, __x.real()), ::cuda::std::copysign(__ans_imag, __x.imag()));
}

// We have performance issues with some trigonometric functions with extended floating point types
#if _LIBCUDACXX_HAS_NVBF16()
template <>
_CCCL_API inline complex<__nv_bfloat16> asinh(const complex<__nv_bfloat16>& __x) noexcept
{
  return complex<__nv_bfloat16>{::cuda::std::asinh(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _LIBCUDACXX_HAS_NVFP16()
template <>
_CCCL_API inline complex<__half> asinh(const complex<__half>& __x) noexcept
{
  return complex<__half>{::cuda::std::asinh(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVFP16()

// acosh

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> acosh(const complex<_Tp>& __x)
{
  constexpr _Tp __pi = __numbers<_Tp>::__pi();
  if (::cuda::std::isinf(__x.real()))
  {
    if (::cuda::std::isnan(__x.imag()))
    {
      return complex<_Tp>(::cuda::std::abs(__x.real()), __x.imag());
    }
    if (::cuda::std::isinf(__x.imag()))
    {
      if (__x.real() > _Tp(0))
      {
        return complex<_Tp>(__x.real(), ::cuda::std::copysign(__pi * _Tp(0.25), __x.imag()));
      }
      else
      {
        return complex<_Tp>(-__x.real(), ::cuda::std::copysign(__pi * _Tp(0.75), __x.imag()));
      }
    }
    if (__x.real() < _Tp(0))
    {
      return complex<_Tp>(-__x.real(), ::cuda::std::copysign(__pi, __x.imag()));
    }
    return complex<_Tp>(__x.real(), ::cuda::std::copysign(_Tp(0), __x.imag()));
  }
  if (::cuda::std::isnan(__x.real()))
  {
    if (::cuda::std::isinf(__x.imag()))
    {
      return complex<_Tp>(::cuda::std::abs(__x.imag()), __x.real());
    }
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (::cuda::std::isinf(__x.imag()))
  {
    return complex<_Tp>(::cuda::std::abs(__x.imag()), ::cuda::std::copysign(__pi / _Tp(2), __x.imag()));
  }
  complex<_Tp> __z = ::cuda::std::log(__x + ::cuda::std::sqrt(::cuda::std::__sqr(__x) - _Tp(1)));
  return complex<_Tp>(::cuda::std::copysign(__z.real(), _Tp(0)), ::cuda::std::copysign(__z.imag(), __x.imag()));
}

// We have performance issues with some trigonometric functions with extended floating point types
#if _LIBCUDACXX_HAS_NVBF16()
template <>
_CCCL_API inline complex<__nv_bfloat16> acosh(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{::cuda::std::acosh(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _LIBCUDACXX_HAS_NVFP16()
template <>
_CCCL_API inline complex<__half> acosh(const complex<__half>& __x)
{
  return complex<__half>{::cuda::std::acosh(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVFP16()

// atanh

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> atanh(const complex<_Tp>& __x)
{
  constexpr _Tp __pi = __numbers<_Tp>::__pi();
  if (::cuda::std::isinf(__x.imag()))
  {
    return complex<_Tp>(::cuda::std::copysign(_Tp(0), __x.real()), ::cuda::std::copysign(__pi / _Tp(2), __x.imag()));
  }
  if (::cuda::std::isnan(__x.imag()))
  {
    if (::cuda::std::isinf(__x.real()) || __x.real() == _Tp(0))
    {
      return complex<_Tp>(::cuda::std::copysign(_Tp(0), __x.real()), __x.imag());
    }
    return complex<_Tp>(__x.imag(), __x.imag());
  }
  if (::cuda::std::isnan(__x.real()))
  {
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (::cuda::std::isinf(__x.real()))
  {
    return complex<_Tp>(::cuda::std::copysign(_Tp(0), __x.real()), ::cuda::std::copysign(__pi / _Tp(2), __x.imag()));
  }
  if (::cuda::std::abs(__x.real()) == _Tp(1) && __x.imag() == _Tp(0))
  {
    return complex<_Tp>(::cuda::std::copysign(numeric_limits<_Tp>::infinity(), __x.real()),
                        ::cuda::std::copysign(_Tp(0), __x.imag()));
  }
  complex<_Tp> __z = ::cuda::std::log((_Tp(1) + __x) / (_Tp(1) - __x)) / _Tp(2);
  return complex<_Tp>(::cuda::std::copysign(__z.real(), __x.real()), ::cuda::std::copysign(__z.imag(), __x.imag()));
}

// We have performance issues with some trigonometric functions with extended floating point types
#if _LIBCUDACXX_HAS_NVBF16()
template <>
_CCCL_API inline complex<__nv_bfloat16> atanh(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{::cuda::std::atanh(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _LIBCUDACXX_HAS_NVFP16()
template <>
_CCCL_API inline complex<__half> atanh(const complex<__half>& __x)
{
  return complex<__half>{::cuda::std::atanh(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVFP16()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___COMPLEX_INVERSE_HYPERBOLIC_FUNCTIONS_H
