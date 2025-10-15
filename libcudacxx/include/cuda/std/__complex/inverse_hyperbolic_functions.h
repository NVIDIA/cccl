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

// asinh

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> asinh(const complex<_Tp>& __x)
{
  constexpr _Tp __pi = __numbers<_Tp>::__pi();
  if (::cuda::std::isinf(__x.real()))
  {
    if (::cuda::std::isnan(__x.imag()))
    {
      return __x;
    }
    if (::cuda::std::isinf(__x.imag()))
    {
      return complex<_Tp>(__x.real(), ::cuda::std::copysign(__pi * _Tp(0.25), __x.imag()));
    }
    return complex<_Tp>(__x.real(), ::cuda::std::copysign(_Tp(0), __x.imag()));
  }
  if (::cuda::std::isnan(__x.real()))
  {
    if (::cuda::std::isinf(__x.imag()))
    {
      return complex<_Tp>(__x.imag(), __x.real());
    }
    if (__x.imag() == _Tp(0))
    {
      return __x;
    }
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (::cuda::std::isinf(__x.imag()))
  {
    return complex<_Tp>(::cuda::std::copysign(__x.imag(), __x.real()),
                        ::cuda::std::copysign(__pi / _Tp(2), __x.imag()));
  }
  complex<_Tp> __z = ::cuda::std::log(__x + ::cuda::std::sqrt(::cuda::std::__sqr(__x) + _Tp(1)));
  return complex<_Tp>(::cuda::std::copysign(__z.real(), __x.real()), ::cuda::std::copysign(__z.imag(), __x.imag()));
}












// double-double approx sqrt for asinh/acosh/atanh
// Assumes __hi & __lo are normalized/close to normalized.
static void __device__ __host__ __forceinline__ __internal_double_double_sqrt_unsafe(double __hi, double __lo, double* __out_hi, double* __out_lo){

  // Calculate using rsqrt and N-R.
#ifdef __CUDA_ARCH__
  // Faster/as accurate on device:
  double __initial_guess = rsqrt(__hi);
#else
  double __initial_guess = 1.0/_CUDA_VSTD::sqrt(__hi);
#endif
  // Newton-Raphson, assume we have a reasonable rsqrt above and
  // we won't need to update the first term (__initial_guess).
  //     x_(n+1) = x_n - 0.5*x_n*((__hi + __lo)(x_n^2) - 1)

  // __initial_guess^2:
  double __init_sq_hi = __initial_guess*__initial_guess;
  double __init_sq_lo = _CUDA_VSTD::fma(__initial_guess, __initial_guess, -__init_sq_hi);

  // Times __hi + __lo - 1.0
  // Need to do the -1.0 here in an fma, or different compilers (eg host vs device)
  // optimize differently and give very different results.
  double _hi_hi_hi = fma(__hi, __init_sq_hi, -1.0);
  // Not needed it seems:
  // double _hi_hi_lo = fma(__hi, __init_sq_hi, -_hi_hi_hi);

  // Add these terms (and subtract 1.0, but this is done in _hi_hi_hi above):
  double __full_term = (_hi_hi_hi) + (fma(__lo, __init_sq_hi, __hi * __init_sq_lo) /*+ _hi_hi_lo*/);

  double __correction_term = -0.5*__initial_guess*__full_term;

  // rsqrt(hi + lo) is now estimated well by (__initial_guess + __correction_term)
  // Multiply everything by (hi + lo) to get sqrt(hi + lo)
  double __ans_hi_hi = __hi * __initial_guess;
  double __ans_hi_lo = _CUDA_VSTD::fma(__hi, __initial_guess, -__ans_hi_hi);

    // All terms needed
  __ans_hi_lo += __initial_guess*__lo + __correction_term*__hi;

  *__out_hi = __ans_hi_hi;
  *__out_lo = __ans_hi_lo;

  return;
}

template<typename T>
static void __device__ __host__ __forceinline__ __internal__dadd(T* __hi1, T* __lo1){
  T __max = (fabsf(*__hi1) > fabsf(*__lo1)) ? *__hi1 : *__lo1;
  T __min = (fabsf(*__hi1) > fabsf(*__lo1)) ? *__lo1 : *__hi1;

  T __temp = __max + __min;
  *__lo1 = (__max - __temp) + __min;
  *__hi1 = __temp;
}

// Modifies inputs
template<typename T>
static void __device__ __host__ __forceinline__ __internal__ddadd(T* __hi1, T* __lo1, T* __hi2, T* __lo2){
  // From paper:
  __internal__dadd<T>(__hi1, __hi2);
  __internal__dadd<T>(__lo1, __lo2);
  __internal__dadd<T>(__hi1, __lo1);
  *__hi2 += *__lo2;
  *__hi2 += *__lo1;
  __internal__dadd<T>(__hi1, __hi2);
}

// double-double approx sqrt for asinh/acosh/atanh
// Assumes __hi & __lo are normalized/close to normalized.
static void __device__ __host__ __forceinline__ __internal_float_float_sqrt_unsafe(float __hi, float __lo, float* __out_hi, float* __out_lo){

  // Calculate using rsqrt and N-R.
#ifdef __CUDA_ARCH__
  // Faster/as accurate on device:
  float __initial_guess = rsqrtf(__hi);
#else
  float __initial_guess = 1.0f/_CUDA_VSTD::sqrtf(__hi);
#endif
  // Newton-Raphson, assume we have a reasonable rsqrt above and
  // we won't need to update the first term (__initial_guess).
  //     x_(n+1) = x_n - 0.5*x_n*((__hi + __lo)(x_n^2) - 1)

  // __initial_guess^2:
  float __init_sq_hi = __initial_guess*__initial_guess;
  float __init_sq_lo = _CUDA_VSTD::fmaf(__initial_guess, __initial_guess, -__init_sq_hi);

  // Times __hi + __lo - 1.0
  // Need to do the -1.0 here in an fmaf, or different compilers (eg host vs device)
  // optimize differently and give very different results.
  float _hi_hi_hi = fmaf(__hi, __init_sq_hi, -1.0);
  // Not needed it seems:
  // float _hi_hi_lo = fmaf(__hi, __init_sq_hi, -_hi_hi_hi);

  // Add these terms (and subtract 1.0, but this is done in _hi_hi_hi above):
  float __full_term = (_hi_hi_hi) + (fmaf(__lo, __init_sq_hi, __hi * __init_sq_lo) /*+ _hi_hi_lo*/);

  float __correction_term = -0.5*__initial_guess*__full_term;

  // rsqrt(hi + lo) is now estimated well by (__initial_guess + __correction_term)
  // Multiply everything by (hi + lo) to get sqrt(hi + lo)
  float __ans_hi_hi = __hi * __initial_guess;
  float __ans_hi_lo = _CUDA_VSTD::fmaf(__hi, __initial_guess, -__ans_hi_hi);

    // All terms needed
  __ans_hi_lo += __initial_guess*__lo + __correction_term*__hi;

  *__out_hi = __ans_hi_hi;
  *__out_lo = __ans_hi_lo;

  return;
}






// asinh double specialization

// ---------------------------------------------- Overall worst errors ----------------------------------------------
// Max relative real error (5.615e-16,-3.268e-16) @ (-0.03396100371,0.004313937501)        (0xbfa1635630852d4e,0x3f71ab7dc7c61061)
//         Ours = (-0.03395479441,0.004311465223)    Ref = (-0.03395479441,0.004311465223)
//         Ours = (0xbfa16285d7098dad,0x3f71a8e6221a6a7a)               Ref = (0xbfa16285d7098db0,0x3f71a8e6221a6a78)

// Max relative imag error (0,-0.001637) @ (69.55096466,-1.049889497e-319) (0x405163430145de58,0x8000000000005302)
//         Ours = (4.935258646,-1.511840876e-321)    Ref = (4.935258646,-1.50690022e-321)
//         Ours = (0x4013bdb4714a5275,0x8000000000000132)               Ref = (0x4013bdb4714a5275,0x8000000000000131)

// Max ulp real error (4.607,0.5635) @ (0.03071642028,1.964854902e-12)     (0x3f9f7420123a50b7,0x3d814875bc2b8904)
//         Ours = (0.03071159218,1.963928639e-12)    Ref = (0.03071159218,1.963928639e-12)
//         Ours = (0x3f9f72dc101b623f,0x3d81465fc7d1636e)               Ref = (0x3f9f72dc101b6244,0x3d81465fc7d1636d)

// Max ulp imag error (0.6259,4.54) @ (-0.04998852748,0.007643015659)      (0xbfa99818a566ddf0,0x3f7f4e4864bd84b3)
//         Ours = (-0.04996918652,0.007633557714)    Ref = (-0.04996918652,0.007633557714)
//         Ours = (0xbfa9958fabdc4abd,0x3f7f445d8bbb7185)               Ref = (0xbfa9958fabdc4abc,0x3f7f445d8bbb718a)
template <>
_LIBCUDACXX_HIDE_FROM_ABI complex<double> asinh(const complex<double>& __x)
{
  double __realx = _CUDA_VSTD::fabs(__x.real());
  double __imagx = _CUDA_VSTD::fabs(__x.imag());

  // Special cases that do not pass through:
  if(!_CUDA_VSTD::isfinite(__realx) || !_CUDA_VSTD::isfinite(__imagx)){
    //If z is (x,+inf) (for any positive finite x), the result is (+inf, pi/2)
    if(_CUDA_VSTD::isfinite(__realx) && _CUDA_VSTD::isinf(__imagx)){
      return complex<double>(::cuda::std::copysign(numeric_limits<double>::infinity(), __x.real()), ::cuda::std::copysign(1.570796326794896619231, __x.imag()));
    }

    //If z is (+inf,y) (for any positive finite y), the result is (+inf,+0)
    if(_CUDA_VSTD::isinf(__realx) && _CUDA_VSTD::isfinite(__imagx)){
      return complex<double>(::cuda::std::copysign(numeric_limits<double>::infinity(), __x.real()), ::cuda::std::copysign(0.0, __x.imag()));
    }

    //If z is (+inf,+inf), the result is (+inf, pi/4)
    if(_CUDA_VSTD::isinf(__realx) && _CUDA_VSTD::isinf(__imagx)){
      return complex<double>(::cuda::std::copysign(numeric_limits<double>::infinity(), __x.real()), ::cuda::std::copysign(0.5*1.570796326794896619231, __x.imag()));
    }

    //If z is (+inf,NaN), the result is (+inf,NaN)
    if(_CUDA_VSTD::isinf(__realx) && _CUDA_VSTD::isnan(__imagx)){
      return __x;
    }

    //If z is (NaN,+0), the result is (NaN,+0)
    if(_CUDA_VSTD::isnan(__realx) && (__imagx == 0.0)){
      return __x;
    }

    // If z is (NaN,+inf), the result is (±INF,NaN) (the sign of the real part is unspecified)
    if(_CUDA_VSTD::isnan(__realx) && _CUDA_VSTD::isinf(__imagx)){
      return complex<double>(__x.imag(), NAN);
    }
  }

  // A case that for various reasons does not pass
  // easily through the algorithm below:
  if((__realx == 0.0) && (__imagx == 1.0)){
    // pi/2
    return complex<double>(__x.real(), ::cuda::std::copysign(1.57079632679489661923132, __x.imag()));
  }

  // It is a little involved to account for large inputs in an
  // inlined-into-existing-code fashion. The easiest place to
  // account for large values appears to be here at the start.
  // Uses asinh(x) ~ log(2x) for large x.
  // Largest exponent that passes through the algorithm without issue:
  const int __max_allowed_exponent = 254;

  //  Check if the largest component of __x is > 2^__max_allowed_exponent:
  double __max = (__realx > __imagx) ? __realx : __imagx;
  bool __x_big = reinterpret_cast<uint64_t&>(__max) > (uint64_t(__max_allowed_exponent + 1023) << 52);
  double __x_big_factor = 0.0;

  if(__x_big){
    // We need __max to be <= ~(2^__max_allowed_exponent),
    // but not small enough that the asinh(x) ~ log(2x) estimate does
    // not break down. We are not able to reduce with a single simple reduction
    // for the remaining double range, so we use a fast/inlined frexp/ldexp:
    int __exp_biased = int(reinterpret_cast<uint64_t&>(__max) >> 52);

    // Get a factor such that (__max * __exp_mul_factor) <= 2^254
    uint64_t __exp_reduce_factor = uint64_t(2046 + __max_allowed_exponent - __exp_biased) << 52;
    double __exp_mul_factor = reinterpret_cast<double&>(__exp_reduce_factor);

    __realx *= __exp_mul_factor;
    __imagx *= __exp_mul_factor;

    __x_big_factor = double((__exp_biased - 1023) - __max_allowed_exponent) * 0.693147180559945309417; // ln(2)
  }

  double __diffx_m1 = __realx*__realx - (__imagx - 1.0)*(__imagx + 1.0);

  // Get the real and imag parts of |sqrt(z^2 + 1)|^2 via pure algebra.
  // This equates to calculating:
  //     sqrt((re*re + (im + 1.0)*(im + 1.0))*(re*re + (im - 1.0)*(im - 1.0)));
  // Where we need both the term inside the sqrt in extended precision, as well
  // as evaluation the sqrt itself in extended precision.

  // Get re^2 + im^2 + 1 in extended precision.
  // The low part of re^2 doesn't seem to matter.
  double __imagx_sq_hi = __imagx*__imagx;
  double __imagx_sq_lo = _CUDA_VSTD::fma(__imagx, __imagx, -__imagx_sq_hi);

  double __x_abs_sq_hi = __imagx_sq_hi;
  double __x_abs_sq_lo =  _CUDA_VSTD::fma(__realx,__realx, __imagx_sq_lo);

  // Add one:
  double __x_abs_sq_p1_hi = (__x_abs_sq_hi + 1.0);
  double __x_abs_sq_p1_lo = __x_abs_sq_lo - ((__x_abs_sq_p1_hi - 1.0) - __x_abs_sq_hi);

  // square:
  double __x_abs_sq_p1_sq_hi = __x_abs_sq_p1_hi*__x_abs_sq_p1_hi;
  double __x_abs_sq_p1_sq_lo = _CUDA_VSTD::fma(__x_abs_sq_p1_hi,__x_abs_sq_p1_hi, -__x_abs_sq_p1_sq_hi);

  // Add in the lower square terms, all needed
  //  __x_abs_sq_p1_sq_lo +=      2.0*__x_abs_sq_p1_lo *__x_abs_sq_p1_hi    +     __x_abs_sq_p1_lo *__x_abs_sq_p1_lo
  __x_abs_sq_p1_sq_lo = _CUDA_VSTD::fma(__x_abs_sq_p1_lo, (2.0*__x_abs_sq_p1_hi + __x_abs_sq_p1_lo), __x_abs_sq_p1_sq_lo);

  // Get __x_abs_sq_p1_sq_hi/lo - 4.0*__imagx_sq_hi/lo:
  // Subtract high parts:
  double __inner_most_term_hi = __x_abs_sq_p1_sq_hi - 4.0*__imagx_sq_hi;
  double __inner_most_term_lo = ((__x_abs_sq_p1_sq_hi - __inner_most_term_hi) - 4.0*__imagx_sq_hi);
  // lo parts, all needed:
  __inner_most_term_lo += __x_abs_sq_p1_sq_lo - 4.0*__imagx_sq_lo;

  // We can have some slightly bad cases here still that can't be fixed easily later,
  // when real is ~ 1.0e-14 and imag = 1-esp (slightly less than 1).
  // We still need to to the extended sqrt later on these values, so we have to
  // evaluate this accurately in here before that. Keep the interval as tight as possible
  // to try avoid this branch. Can probably be made tighter with more analysis
  if((__realx < 1.0e-8) && (0.999 < __imagx) && (__imagx <= 1.0)){
    // Need to get (real^2 + (1 - imag)^2) * (real^2 + (1 + imag)^2) in double-double:
    // term1 = (real^2 + (1 - imag)^2)
    double __term1_hi = (1.0 - __imagx) * (1.0 - __imagx);
    double __term1_lo = fma(1.0 - __imagx, 1.0 - __imagx, -__term1_hi) + __realx* __realx;

    // Need (1.0 + __imagx)^2 with annoyingly accuracy.
    double __term2_sum_hi = (1.0 + __imagx);
    double __term2_sum_lo = ((1.0 -__term2_sum_hi) + __imagx);

    double __term2_sq_hi = __term2_sum_hi*__term2_sum_hi;
    double __term2_sq_lo = fma(__term2_sum_hi,__term2_sum_hi, -__term2_sq_hi);
    __term2_sq_lo += 2.0*__term2_sum_hi*__term2_sum_lo;

      // Multiple __term1_hi/lo and __term2_sq_hi/lo:
      __inner_most_term_hi = __term1_hi*__term2_sq_hi;
      __inner_most_term_lo = fma(__term1_hi, __term2_sq_hi, -__inner_most_term_hi);
      // All needed:
      __inner_most_term_lo += __term1_hi*__term2_sq_lo + __term1_lo*__term2_sq_hi;
  }

  // Normalize the above (needed for calling the sqrt function below):
  double __norm_hi = __inner_most_term_hi + __inner_most_term_lo;
  double __norm_lo = -((__norm_hi - __inner_most_term_hi) - __inner_most_term_lo);
  __inner_most_term_hi = __norm_hi;
  __inner_most_term_lo = __norm_lo;

  // Extended sqrt function:
  // (__extended_sqrt_hi + __extended_sqrt_lo) = sqrt(__inner_most_term_hi + __inner_most_term_lo)
  double __extended_sqrt_hi;
  double __extended_sqrt_lo;
  __internal_double_double_sqrt_unsafe(__inner_most_term_hi, __inner_most_term_lo, &__extended_sqrt_hi, &__extended_sqrt_lo);

  // 0.0, and some very particular values, do not survive this unsafe sqrt function.
  // This case occurs when real^2 == 0.0 and imag == 1.0, or when
  // 1 + x^2 is denormal. (and rsqrt(x)*rsqrt(x) become inf).
  // In both cases the fix is the same.
  // The mathematical value needed here is the smallest normal, but double it
  // to allow for some rsqrt/rounding errors:
  if(__inner_most_term_hi <= 2.0*2.22507385850720138309023271733E-308){
    __extended_sqrt_hi = 2.0*__realx;
    __extended_sqrt_lo = 0.0;
  }

  // Get sqrt(0.5*(__diffx_m1 + __extended_sqrt_hi)
  // This can result in catastrophic cancellation if __diffx_m1 < 0, in this case
  // We instead use the equivalent
  //     __realx*__imagx/sqrt(0.5*(-__diffx_m1 + __extended_sqrt_hi)

  double __inside_sqrt_term = 0.5*(_CUDA_VSTD::fabs(__diffx_m1) + __extended_sqrt_hi);

  // Allow for rsqrt optimization:
  // We can have two slightly different paths depending on whether rsqrt is available
  // or not, aka are we on device or host.
#ifdef __CUDA_ARCH__
  // Faster/more accurate on device:
  double __recip_sqrt = rsqrt(__inside_sqrt_term);
  double __pos_evaluation_real = (__recip_sqrt*__inside_sqrt_term);
#else
  double __recip_sqrt = 1.0/_CUDA_VSTD::sqrt(__inside_sqrt_term);
  double __pos_evaluation_real = _CUDA_VSTD::sqrt(__inside_sqrt_term);
#endif

  // Here, in a happy coincidence(?), we happen to intermediately calculate an accurate
  // return value for the real part of the answer in the case that __realx is small,
  // as you would obtain from the Taylor expansion of asinh. (~ real/sqrt(1 - imag^2)).
  // The following parts of the calculation result in bad catastrophic cancellation for
  // this case, so we save this intermediate value:
  double __small_x_real_return_val = __realx * __recip_sqrt;
  double __pos_evaluation_imag = __imagx * __small_x_real_return_val;

  double __sqrt_real_part = (__diffx_m1 > 0.0) ? __pos_evaluation_real: __pos_evaluation_imag;
  double __sqrt_imag_part = (__diffx_m1 > 0.0) ? __pos_evaluation_imag: __pos_evaluation_real;

  // for an accurate log, we calculate |(__sqrt_real_part + i*__sqrt_imag_part)| - 1 and use log1p.
  // We need evaluate this without any cancellations.

  // This can normally have bad catastrophic cancellation, however
  // we have a lot of retained enough accuracy to subtract fairly simply:
  double __m1 = __extended_sqrt_hi - 1.0;
  double __rem = -((__m1 + 1.0) - __extended_sqrt_hi);

  __extended_sqrt_hi = __m1;
  __extended_sqrt_lo += __rem;

  // Final sum before sending it to log1p, all terms needed.
  // Add our sum via three terms, there might be a better order.
  double __sum1 = (__x_abs_sq_hi + __extended_sqrt_hi);
  double __sum2 = 2.0*(__realx*__sqrt_real_part + __imagx*__sqrt_imag_part);
  double __sum3 = (__extended_sqrt_lo + __x_abs_sq_lo);

  double __abs_sqrt_part_sq = __sum1 + (__sum2  + __sum3);

  double __atan2_input1 = __imagx + __sqrt_imag_part;
  double __atan2_input2 = __realx + __sqrt_real_part;

  double __ans_real = 0.5*_CUDA_VSTD::log1p(__abs_sqrt_part_sq);
  double __ans_imag = _CUDA_VSTD::atan2(__atan2_input1, __atan2_input2);

  // The small |real| case, as mentioned above.
  // Bound found by testing, can't be wiggled much.
  // Some extrememly/intensive targetted testing does
  if(__realx < 1.0e-14 && __imagx < 1.0){
    __ans_real = __small_x_real_return_val;
  }

  __ans_real += __x_big_factor;

  // Copy signs back in
  return complex<double>(::cuda::std::copysign(__ans_real, __x.real()), ::cuda::std::copysign(__ans_imag, __x.imag()));
}













































#include<iostream>








template <>
_LIBCUDACXX_HIDE_FROM_ABI complex<float> asinh(const complex<float>& __x)
{
  float __realx = _CUDA_VSTD::fabsf(__x.real());
  float __imagx = _CUDA_VSTD::fabsf(__x.imag());

  int debug = (__x.real() == float(9.665625385e-07) && __x.imag() == float(0.9999999404)); 
  // debug = 0;
  // {
  //   printf("\n\n__x = (%e, %e)\t ans = (%e, %e)\n\n", float(__x.real()), float(__x.imag()), float(__ans_real), float(__ans_imag));
  // }

  // Special cases that do not pass through:
  if(!_CUDA_VSTD::isfinite(__realx) || !_CUDA_VSTD::isfinite(__imagx)){
    //If z is (x,+inf) (for any positive finite x), the result is (+inf, pi/2)
    if(_CUDA_VSTD::isfinite(__realx) && _CUDA_VSTD::isinf(__imagx)){
      return complex<float>(::cuda::std::copysign(numeric_limits<float>::infinity(), __x.real()), ::cuda::std::copysign(1.570796326794896619231f, __x.imag()));
    }

    //If z is (+inf,y) (for any positive finite y), the result is (+inf,+0)
    if(_CUDA_VSTD::isinf(__realx) && _CUDA_VSTD::isfinite(__imagx)){
      return complex<float>(::cuda::std::copysign(numeric_limits<float>::infinity(), __x.real()), ::cuda::std::copysign(0.0f, __x.imag()));
    }

    //If z is (+inf,+inf), the result is (+inf, pi/4)
    if(_CUDA_VSTD::isinf(__realx) && _CUDA_VSTD::isinf(__imagx)){
      return complex<float>(::cuda::std::copysign(numeric_limits<float>::infinity(), __x.real()), ::cuda::std::copysign(0.5f*1.570796326794896619231f, __x.imag()));
    }

    //If z is (+inf,NaN), the result is (+inf,NaN)
    if(_CUDA_VSTD::isinf(__realx) && _CUDA_VSTD::isnan(__imagx)){
      return __x;
    }

    //If z is (NaN,+0), the result is (NaN,+0)
    if(_CUDA_VSTD::isnan(__realx) && (__imagx == 0.0f)){
      return __x;
    }

    // If z is (NaN,+inf), the result is (±INF,NaN) (the sign of the real part is unspecified)
    if(_CUDA_VSTD::isnan(__realx) && _CUDA_VSTD::isinf(__imagx)){
      return complex<float>(__x.imag(), NAN);
    }
  }

  // A case that for various reasons does not pass
  // easily through the algorithm below:
  if((__realx == 0.0f) && (__imagx == 1.0f)){
    // pi/2
    return complex<float>(__x.real(), ::cuda::std::copysign(1.57079632679489661923132f, __x.imag()));
  }

  // It is a little involved to account for large inputs in an
  // inlined-into-existing-code fashion. The easiest place to
  // account for large values appears to be here at the start.
  // Uses asinh(x) ~ log(2x) for large x.
  // Largest exponent that passes through the algorithm without issue:
  const int __max_allowed_exponent = 30;

  //  Check if the largest component of __x is > 2^__max_allowed_exponent:
  float __max = (__realx > __imagx) ? __realx : __imagx;
  bool __x_big = reinterpret_cast<uint32_t&>(__max) > (uint32_t(__max_allowed_exponent + 127) << 23);
  float __x_big_factor = 0.0f;

  if(__x_big){
    // We need __max to be <= ~(2^__max_allowed_exponent),
    // but not small enough that the asinh(x) ~ log(2x) estimate does
    // not break down. We are not able to reduce with a single simple reduction
    // for the remaining double range, so we use a fast/inlined frexp/ldexp:
    int __exp_biased = int(reinterpret_cast<uint32_t&>(__max) >> 23);

    // Get a factor such that (__max * __exp_mul_factor) <= 2^254
    uint32_t __exp_reduce_factor = uint32_t(254 + __max_allowed_exponent - __exp_biased) << 23;
    float __exp_mul_factor = reinterpret_cast<float&>(__exp_reduce_factor);

    __realx *= __exp_mul_factor;
    __imagx *= __exp_mul_factor;

    __x_big_factor = float((__exp_biased - 127) - __max_allowed_exponent) * 0.693147180559945309417f; // ln(2)
  }

  float __diffx_m1 = __realx*__realx - (__imagx - 1.0f)*(__imagx + 1.0f);

  // Get the real and imag parts of |sqrt(z^2 + 1)|^2 via pure algebra.
  // This equates to calculating:
  //     sqrt((re*re + (im + 1.0)*(im + 1.0))*(re*re + (im - 1.0)*(im - 1.0)));
  // Where we need both the term inside the sqrt in extended precision, as well
  // as evaluation the sqrt itself in extended precision.

  // Get re^2 + im^2 + 1 in extended precision.
  // The low part of re^2 doesn't seem to matter.
  float __imagx_sq_hi = __imagx*__imagx;
  float __imagx_sq_lo = _CUDA_VSTD::fmaf(__imagx, __imagx, -__imagx_sq_hi);

  float __x_abs_sq_hi = __imagx_sq_hi;
  float __x_abs_sq_lo =  _CUDA_VSTD::fmaf(__realx,__realx, __imagx_sq_lo);

  // Add one:
  float __x_abs_sq_p1_hi = (__x_abs_sq_hi + 1.0f);
  float __x_abs_sq_p1_lo = __x_abs_sq_lo - ((__x_abs_sq_p1_hi - 1.0f) - __x_abs_sq_hi);

  // square:
  float __x_abs_sq_p1_sq_hi = __x_abs_sq_p1_hi*__x_abs_sq_p1_hi;
  float __x_abs_sq_p1_sq_lo = _CUDA_VSTD::fmaf(__x_abs_sq_p1_hi,__x_abs_sq_p1_hi, -__x_abs_sq_p1_sq_hi);

  // Add in the lower square terms, all needed
  //  __x_abs_sq_p1_sq_lo +=      2.0*__x_abs_sq_p1_lo *__x_abs_sq_p1_hi    +     __x_abs_sq_p1_lo *__x_abs_sq_p1_lo
  __x_abs_sq_p1_sq_lo = _CUDA_VSTD::fmaf(__x_abs_sq_p1_lo, (2.0*__x_abs_sq_p1_hi + __x_abs_sq_p1_lo), __x_abs_sq_p1_sq_lo);

  // Get __x_abs_sq_p1_sq_hi/lo - 4.0*__imagx_sq_hi/lo:
  // Subtract high parts:
  float __inner_most_term_hi = __x_abs_sq_p1_sq_hi - 4.0f*__imagx_sq_hi;
  float __inner_most_term_lo = ((__x_abs_sq_p1_sq_hi - __inner_most_term_hi) - 4.0f*__imagx_sq_hi);
  // lo parts, all needed:
  __inner_most_term_lo += __x_abs_sq_p1_sq_lo - 4.0f*__imagx_sq_lo;

     if(debug){
      printf("\n\n__inner_most_term_hi = %.20e, __inner_most_term_lo = %.20e\n\n", __inner_most_term_hi, __inner_most_term_lo);
     }

  // We can have some slightly bad cases here still that can't be fixed easily later,
  // when real is ~ 1.0e-14 and imag = 1-esp (slightly less than 1).
  // We still need to to the extended sqrt later on these values, so we have to
  // evaluate this accurately in here before that. Keep the interval as tight as possible
  // to try avoid this branch. Can probably be made tighter with more analysis
  // if(0){
  // if((__realx < 1.0e-8f) && (0.999f < __imagx) && (__imagx <= 1.0f)){
  //   // Need to get (real^2 + (1 - imag)^2) * (real^2 + (1 + imag)^2) in double-double:
  //   // (real^2 + (1 - imag)^2) * (real^2 + (1 + imag)^2) = (real^2 + (1 - imag)^2) * (1 + 2*imag + img^2 + real^2)
  //   //                                                   = (real^2 + (1 - imag)^2) * (1 + 2*imag + img^2 + real^2)
  //   //                                                   = (real^2 + (1 - imag)^2) * (2*imag + img^2 + real^2) + (real^2 + (1 - imag)^2)
  //   //                                                   = (real^2 + (1 - imag)^2) * (2*imag + (-1 + 1 - img)^2 + real^2) + (real^2 + (1 - imag)^2)
  //   //                                                   = (real^2 + (1 - imag)^2) * (2*imag + (1 - 2*(1 - img) + (1 - img)^2) + real^2) + (real^2 + (1 - imag)^2)
  //   //                                                   = (real^2 + (1 - imag)^2) * (2*imag + (1 - (2 - 2*img) + (1 - img)^2) + real^2) + (real^2 + (1 - imag)^2)
  //   //                                                   = (real^2 + (1 - imag)^2) * (4*imag - 1 + (1 - imag)^2 + real^2) + (real^2 + (1 - imag)^2)
  //   // with inter = (real^2 + (1 - imag)^2)              = inter * (4*imag - 1 + inter) + inter
  //   // with inter = (real^2 + (1 - imag)^2)              = inter * (4*imag + inter)
  //   // with inter = (real^2 + (1 - imag)^2)              = inter^2 + 4*imag*inter


  //   // term1 = (real^2 + (1 - imag)^2)
  //   float __term1_hi = (1.0f - __imagx) * (1.0f - __imagx);
  //   float __term1_lo =  fmaf(__realx, __realx, fmaf(1.0f - __imagx, 1.0f - __imagx, -__term1_hi));

  //   // float __realx_sq_hi = __realx* __realx;
  //   // float __realx_sq_lo = fmaf(__realx, __realx, -__realx_sq_hi);
  // //   //Add up these terms. Needs to be quite accurate.
  // // __internal__ddadd(&__term1_hi, &__term1_lo, &__realx_sq_hi, &__realx_sq_lo);
  // // __term1_lo = __realx_sq_hi;

  //   __float128 re_128 = __realx;
  //   __float128 im_128 = __imagx;
  //   __float128 t1_hi_128 = (__float128(1.0) - im_128) * (__float128(1.0) - im_128) + re_128* re_128;


  //     float hi = t1_hi_128;
  //     float lo = float(t1_hi_128 - __float128(hi));


  //  if(debug){
  //     printf("\n __term1_hi = %.20e (0x%x), __term1_lo = %.20e (0x%x)", __term1_hi, *(uint32_t*)&__term1_hi, __term1_lo, *(uint32_t*)&__term1_lo);
  //     printf("\n hi = %.20e (0x%x), lo = %.20e (0x%x)", hi, *(uint32_t*)&hi, lo, *(uint32_t*)&lo);
  //  }

  //   __float128 t2 = (__float128(1.0) + im_128)*(__float128(1.0) + im_128);
  //   t2 = (re_128*re_128 + t2);

  //   // Need (1.0 + __imagx)^2 with annoyingly accuracy.
  //   float __term2_sum_hi = (1.0f + __imagx);
  //   float __term2_sum_lo = ((1.0f -__term2_sum_hi) + __imagx);

  //   float __term2_sq_hi = __term2_sum_hi*__term2_sum_hi;
  //   float __term2_sq_lo = fmaf(__term2_sum_hi,__term2_sum_hi, -__term2_sq_hi);
  //         __term2_sq_lo += 2.0f*__term2_sum_hi*__term2_sum_lo;
  //   // float __term2_sq_lo = fmaf(2.0f*__term2_sum_hi, __term2_sum_lo, fmaf(__term2_sum_hi,__term2_sum_hi, -__term2_sq_hi));

  //     // Multiple __term1_hi/lo and __term2_sq_hi/lo:
  //     __inner_most_term_hi = __term1_hi*__term2_sq_hi;
  //     // __inner_most_term_lo = fmaf(__term1_hi, __term2_sq_hi, -__inner_most_term_hi);
  //     // // All needed:
  //     // __inner_most_term_lo += __term1_hi*__term2_sq_lo + __term1_lo*__term2_sq_hi;

  //     __inner_most_term_lo = fmaf(__term1_lo, __term2_sq_hi, fmaf( __term1_hi, __term2_sq_lo, fmaf(__term1_hi, __term2_sq_hi, -__inner_most_term_hi)));
  //     // All needed:
  //    // __inner_most_term_lo += __term1_hi*__term2_sq_lo + __term1_lo*__term2_sq_hi;

  //   __float128 prod = t2 * t1_hi_128;

  //   float ref = float(prod);
  //   float ref_lo = float(prod - __float128(ref));

  //   // __inner_most_term_hi = ref;
  //   // __inner_most_term_lo = ref_lo;

  //    if(debug){
  //     printf("\n\n__inner_most_term_hi = %.20e (0x%x), __inner_most_term_lo = %.20e (0x%x)", __inner_most_term_hi, *(uint32_t*)&__inner_most_term_hi, __inner_most_term_lo, *(uint32_t*)&__inner_most_term_lo);
  //     printf("\nref = %.20e (0x%x), ref_lo = %.20e (0x%x)\n", ref, *(uint32_t*)&ref, ref_lo, *(uint32_t*)&ref_lo);
  //    }
  // }






if((__realx < 1.0e-5f) && (0.99f < __imagx) && (__imagx <= 1.0f)){
    // Need to get (real^2 + (1 - imag)^2) * (real^2 + (1 + imag)^2) in double-double:
    // (real^2 + (1 - imag)^2) * (real^2 + (1 + imag)^2) = (real^2 + (1 - imag)^2) * (1 + 2*imag + img^2 + real^2)
    //                                                   = (real^2 + (1 - imag)^2) * (1 + 2*imag + img^2 + real^2)
    //                                                   = (real^2 + (1 - imag)^2) * (2*imag + img^2 + real^2) + (real^2 + (1 - imag)^2)
    //                                                   = (real^2 + (1 - imag)^2) * (2*imag + (-1 + 1 - img)^2 + real^2) + (real^2 + (1 - imag)^2)
    //                                                   = (real^2 + (1 - imag)^2) * (2*imag + (1 - 2*(1 - img) + (1 - img)^2) + real^2) + (real^2 + (1 - imag)^2)
    //                                                   = (real^2 + (1 - imag)^2) * (2*imag + (1 - (2 - 2*img) + (1 - img)^2) + real^2) + (real^2 + (1 - imag)^2)
    //                                                   = (real^2 + (1 - imag)^2) * (4*imag - 1 + (1 - imag)^2 + real^2) + (real^2 + (1 - imag)^2)
    // with inter = (real^2 + (1 - imag)^2)              = inter * (4*imag - 1 + inter) + inter
    // with inter = (real^2 + (1 - imag)^2)              = inter * (4*imag + inter)
    // with inter = (real^2 + (1 - imag)^2)              = inter^2 + 4*imag*inter
    // with inter = (real^2 + (1 - imag)^2)              = inter * (inter + 4*imag)


    // term1 = (real^2 + (1 - imag)^2)
    // not normalized
    float __term1_hi = (1.0f - __imagx) * (1.0f - __imagx);
    float __term1_lo = fmaf(1.0f - __imagx, 1.0f - __imagx, -__term1_hi);

    float __term2_hi = __realx * __realx;
    float __term2_lo = 0; // fmaf(__realx, __realx, -__term2_hi);
    
    // D-D Add:
    __internal__ddadd(&__term1_hi, &__term1_lo, &__term2_hi, &__term2_lo);
    // Hi-Lo is in __term1_hi, __term2_hi

    // Get 4*imag + (hi-lo) in D-D:
    // ddadd modifies the inputs so need a copy.
    // 4*imag is much greater than hi-lo is just the seperate parts
    float __term3_hi = (4.0f*__imagx) ;//+ __term1_hi;
    float __term3_lo = __term1_hi;
    // float __term3_lo = ((4.0f*__imagx) - __term3_hi) + __term1_hi;
         // __term3_lo += __term2_hi;

   
    // Now multiply: (__term1_hi + __term2_hi) * (__term3_hi + __term3_lo)
    float final_mul1_hi = __term1_hi * __term3_hi;
    float final_mul1_lo = fma(__term1_hi, __term3_hi, -final_mul1_hi);

    float final_mul2_hi = __term2_hi * __term3_hi;
    float final_mul2_lo = 0; // fma(__term2_hi, __term3_hi, -final_mul2_hi);

    // Add together:
    __internal__ddadd(&final_mul1_hi, &final_mul1_lo, &final_mul2_hi, &final_mul2_lo);
    final_mul2_hi += (__term3_lo * __term1_hi);

    // is normalized.
    __inner_most_term_hi = final_mul1_hi;
    __inner_most_term_lo = final_mul2_hi;

    // __inner_most_term_lo = fmaf(__term3_lo, __term1_hi, __inner_most_term_lo);
  }









  // Normalize the above (needed for calling the sqrt function below):
  float __norm_hi = __inner_most_term_hi + __inner_most_term_lo;
  float __norm_lo = -((__norm_hi - __inner_most_term_hi) - __inner_most_term_lo);

     if(debug){
      printf("\n__norm_hi = %.20e (0x%x), __norm_lo = %.20e (0x%x)\n\n", __norm_hi, *(uint32_t*)&__norm_hi, __norm_lo, *(uint32_t*)&__norm_lo);
     }

  __inner_most_term_hi = __norm_hi;
  __inner_most_term_lo = __norm_lo;

  // Extended sqrt function:
  // (__extended_sqrt_hi + __extended_sqrt_lo) = sqrt(__inner_most_term_hi + __inner_most_term_lo)
  float __extended_sqrt_hi;
  float __extended_sqrt_lo;
  __internal_float_float_sqrt_unsafe(__inner_most_term_hi, __inner_most_term_lo, &__extended_sqrt_hi, &__extended_sqrt_lo);


    if(debug){
    printf("\n5)__x __inner_most_term_hi = %.10e, __inner_most_term_lo = %.10e, __extended_sqrt_hi = %.10e, __extended_sqrt_lo = %.10e\n", __inner_most_term_hi, __inner_most_term_lo, __extended_sqrt_hi, __extended_sqrt_lo);
  }


  // 0.0, and some very particular values, do not survive this unsafe sqrt function.
  // This case occurs when real^2 == 0.0 and imag == 1.0, or when
  // 1 + x^2 is denormal. (and rsqrt(x)*rsqrt(x) become inf).
  // In both cases the fix is the same.
  // The mathematical value needed here is the smallest normal, but double it
  // to allow for some rsqrt/rounding errors:
  if(__inner_most_term_hi <= 2.0f*1.17549435082228750796873653722E-38f){ // Smallest normal.

    __extended_sqrt_hi = 2.0f*__realx;
    __extended_sqrt_lo = 0.0f;
  }

  // Get sqrt(0.5*(__diffx_m1 + __extended_sqrt_hi)
  // This can result in catastrophic cancellation if __diffx_m1 < 0, in this case
  // We instead use the equivalent
  //     __realx*__imagx/sqrt(0.5*(-__diffx_m1 + __extended_sqrt_hi)

  float __inside_sqrt_term = 0.5f*(_CUDA_VSTD::fabsf(__diffx_m1) + __extended_sqrt_hi);

  // Allow for rsqrt optimization:
  // We can have two slightly different paths depending on whether rsqrt is available
  // or not, aka are we on device or host.
#ifdef __CUDA_ARCH__
  // Faster/more accurate on device:
  float __recip_sqrt = rsqrtf(__inside_sqrt_term);
  float __pos_evaluation_real = (__recip_sqrt*__inside_sqrt_term);
#else
  float __recip_sqrt = 1.0f/_CUDA_VSTD::sqrtf(__inside_sqrt_term);
  float __pos_evaluation_real = _CUDA_VSTD::sqrtf(__inside_sqrt_term);
#endif

  // Here, in a happy coincidence(?), we happen to intermediately calculate an accurate
  // return value for the real part of the answer in the case that __realx is small,
  // as you would obtain from the Taylor expansion of asinh. (~ real/sqrt(1 - imag^2)).
  // The following parts of the calculation result in bad catastrophic cancellation for
  // this case, so we save this intermediate value:
  float __small_x_real_return_val = __realx * __recip_sqrt;
  float __pos_evaluation_imag = __imagx * __small_x_real_return_val;

  float __sqrt_real_part = (__diffx_m1 > 0.0f) ? __pos_evaluation_real: __pos_evaluation_imag;
  float __sqrt_imag_part = (__diffx_m1 > 0.0f) ? __pos_evaluation_imag: __pos_evaluation_real;

  // for an accurate log, we calculate |(__sqrt_real_part + i*__sqrt_imag_part)| - 1 and use log1p.
  // We need evaluate this without any cancellations.

  // This can normally have bad catastrophic cancellation, however
  // we have a lot of retained enough accuracy to subtract fairly simply:
  float __m1 = __extended_sqrt_hi - 1.0f;
  float __rem = -((__m1 + 1.0f) - __extended_sqrt_hi);

  __extended_sqrt_hi = __m1;
  __extended_sqrt_lo += __rem;

  // Final sum before sending it to log1p, all terms needed.
  // Add our sum via three terms, there might be a better order.
  float __sum1 = (__x_abs_sq_hi + __extended_sqrt_hi);
  float __sum2 = 2.0f*(__realx*__sqrt_real_part + __imagx*__sqrt_imag_part);
  float __sum3 = (__extended_sqrt_lo + __x_abs_sq_lo);

  if(debug){
    printf("\n6)__x __extended_sqrt_lo = %e, __x_abs_sq_lo = %e, __rem = %e\n", __extended_sqrt_lo, __x_abs_sq_lo, __rem);
  }


  float __abs_sqrt_part_sq = __sum1 + (__sum2  + __sum3);

  float __atan2_input1 = __imagx + __sqrt_imag_part;
  float __atan2_input2 = __realx + __sqrt_real_part;

  if(debug){
    printf("\n7)__x __sum1 = %e, sum2 = %e, sum3 = %e, __abs_sqrt_part_sq = %e\n", __sum1, __sum2, __sum3, __abs_sqrt_part_sq);
  }

  float __ans_real = 0.5f*_CUDA_VSTD::log1pf(__abs_sqrt_part_sq);
  float __ans_imag = _CUDA_VSTD::atan2f(__atan2_input1, __atan2_input2);

  // The small |real| case, as mentioned above.
  // Bound found by testing, can't be wiggled much.
  // Some extrememly/intensive targetted testing does
  if(__realx < 1.0e-6f && __imagx < 1.0f){
    __ans_real = __small_x_real_return_val;
  }

  if(debug){
    printf("\n8)__x __ans_real = %e, __x_big_factor = %e\n", __ans_real, __x_big_factor);
  }
  __ans_real += __x_big_factor;

  if(debug){
    printf("\n\n__x = (%e, %e)\t ans = (%e, %e)\n\n", float(__x.real()), float(__x.imag()), float(__ans_real), float(__ans_imag));
  }

  // Copy signs back in
  return complex<float>(::cuda::std::copysign(__ans_real, __x.real()), ::cuda::std::copysign(__ans_imag, __x.imag()));
}















// asinh(double) CPU:
// ---------------------------------------------- Overall worst errors ----------------------------------------------
// Max relative real error (0.2929,-1.534e-17) @ (4.940656458e-324,0.7071067812)   (0x1,0x3fe6a09e667f3bcd)
//         Ours = (4.940656458e-324,0.7853981634)    Ref = (4.940656458e-324,0.7853981634)
//         Ours = (0x1,0x3fe921fb54442d19)               Ref = (0x1,0x3fe921fb54442d19)

// Max relative imag error (3.48e-17,-0.9999) @ (1.481634398e+221,3.660271137e-103)        (0x6dda3b971368f000,0x2aaa3bdc86a8f000)
//         Ours = (509.9575985,4.940656458e-324)    Ref = (509.9575985,4.940656458e-324)
//         Ours = (0x407fdf5252d7139c,0x1)               Ref = (0x407fdf5252d7139c,0x1)

// Max ulp real error (4.121,0) @ (-0.03124057908,-0)      (0xbf9ffd87c5de3800,0x8000000000000000)
//         Ours = (-0.03123549965,-0)    Ref = (-0.03123549965,-0)
//         Ours = (0xbf9ffc32e5dbf236,0x8000000000000000)               Ref = (0xbf9ffc32e5dbf232,0x8000000000000000)

// Max ulp imag error (0.5094,3.487) @ (0.8869326854,-1.12001698e-254)     (0x3fec61c0a7b18800,0x8b3505783ad41800)
//         Ours = (0.7991224705,-8.379245502e-255)    Ref = (0.7991224705,-8.379245502e-255)
//         Ours = (0x3fe99269498d3a37,0x8b2f742347588681)               Ref = (0x3fe99269498d3a36,0x8b2f742347588684)


// asinh(float) CPU:
// ---------------------------------------------- Overall worst errors ----------------------------------------------
// Max relative real error (0.3333,-2.97e-08) @ (-1.401298464e-45,0.745307982)     (0x80000001,0x3f3ecc81)
//         Ours = (-1.401298464e-45,0.8409966826)    Ref = (-1.401298464e-45,0.8409966826)
//         Ours = (0x80000001,0x3f574b8f)               Ref = (0x80000001,0x3f574b8f)

// Max relative imag error (4.016e-08,-1.125) @ (-1.874807358,-1.401298464e-45)    (0xbfeff9b0,0x80000001)
//         Ours = (-1.386203647,-1.401298464e-45)    Ref = (-1.386203647,-0)
//         Ours = (0xbfb16f1f,0x80000001)               Ref = (0xbfb16f1f,0x80000000)

// Max ulp real error (4.827,0.5125) @ (0.001131535857,0.9893865585)       (0x3a94500b,0x3f7d4870)
//         Ours = (0.007776218932,1.424767017)    Ref = (0.007776216604,1.424766898)
//         Ours = (0x3bfecfa7,0x3fb65ec4)               Ref = (0x3bfecfa2,0x3fb65ec3)

// Max ulp imag error (1.158,4.565) @ (5.023037459e+24,-4.881198872e+21)   (0x6884f56d,0xe3844e1f)
//         Ours = (57.56922913,-0.000971762347)    Ref = (57.56922531,-0.000971762056)
//         Ours = (0x426646e4,0xba7ebdde)               Ref = (0x426646e3,0xba7ebdd9)

// asinh(__half) CPU:
// ---------------------------------------------- Overall worst errors ----------------------------------------------
// Max relative real error (0.3331,-0.0001306) @ (-5.960464478e-08,0.7451171875)   (0x8001,0x39f6)
//         Ours = (-5.960464478e-08,0.8408203125)    Ref = (-5.960464478e-08,0.8408203125)
//         Ours = (0x8001,0x3aba)               Ref = (0x8001,0x3aba)

// Max relative imag error (0.000189,-0.9995) @ (1.731445312,-5.960464478e-08)     (0x3eed,0x8001)
//         Ours = (1.31640625,-5.960464478e-08)    Ref = (1.31640625,-5.960464478e-08)
//         Ours = (0x3d44,0x8001)               Ref = (0x3d44,0x8001)

// Max ulp real error (0.5004,0.3714) @ (0.1186523438,-0.005825042725)     (0x2f98,0x9df7)
//         Ours = (0.1184082031,-0.005783081055)    Ref = (0.118347168,-0.005783081055)
//         Ours = (0x2f94,0x9dec)               Ref = (0x2f93,0x9dec)

// Max ulp imag error (0.4087,0.5002) @ (1.924804688,0.1062011719) (0x3fb3,0x2ecc)
//         Ours = (1.41015625,0.04895019531)    Ref = (1.41015625,0.04891967773)
//         Ours = (0x3da4,0x2a44)               Ref = (0x3da4,0x2a43)


// asinh(__nv_bfloat16) CPU:
// ---------------------------------------------- Overall worst errors ----------------------------------------------
// Max relative real error (-0.3317,-0.001868) @ (-9.183549616e-41,0.74609375)     (0x8001,0x3f3f)
//         Ours = (-1.836709923e-40,0.84375)    Ref = (-1.836709923e-40,0.84375)
//         Ours = (0x8002,0x3f58)               Ref = (0x8002,0x3f58)

// Max relative imag error (0.003029,-0.9968) @ (5.90625,-2.755064885e-40) (0x40bd,0x8003)
//         Ours = (2.46875,-9.183549616e-41)    Ref = (2.46875,-9.183549616e-41)
//         Ours = (0x401e,0x8001)               Ref = (0x401e,0x8001)

// Max ulp real error (0.5,0) @ (3.723491524e+26,-0)       (0x6b9a,0x8000)
//         Ours = (62,-0)    Ref = (62,-0)
//         Ours = (0x4278,0x8000)               Ref = (0x4278,0x8000)

// Max ulp imag error (0.2041,0.5) @ (472,-6.501953128e-38)        (0x43ec,0x81b1)
//         Ours = (6.84375,-1.836709923e-40)    Ref = (6.84375,-9.183549616e-41)
//         Ours = (0x40db,0x8002)               Ref = (0x40db,0x8001)




// We have performance issues with some trigonometric functions with extended floating point types
#if _LIBCUDACXX_HAS_NVBF16()
template <>
_CCCL_API inline complex<__nv_bfloat16> asinh(const complex<__nv_bfloat16>& __x)
{
  const complex<__nv_bfloat16> ans = complex<__nv_bfloat16>{::cuda::std::asinh(complex<float>{__x})};

  return ans;
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _LIBCUDACXX_HAS_NVFP16()
template <>
_CCCL_API inline complex<__half> asinh(const complex<__half>& __x)
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
