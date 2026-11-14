//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_CLASSIFY_H
#define _CUDA___FP_FPMP_MATH_IMPL_CLASSIFY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_classify.h - fpmp2 classification and comparison (isfinite/isinf/isnan/signbit, fmax/fmin/max/min,
   fdim)
    ==================================================================================================
*/

#include <cuda/__fp/fpmp_math_impl.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * ====================================================================
 * fmax(x, y) - larger of two values
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Maximum fmax(x, y) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * fmax: max(x, y).  Lexicographic comparison on (hi, lo) -- valid because
 * normalized fpmp2 inputs satisfy |lo| < ulp(hi)/2, so `x > y` iff
 * `x_hi > y_hi || (x_hi == y_hi && x_lo > y_lo)`.  NaN handling follows
 * C99/IEEE-754-2008: if exactly one operand is NaN, return the other.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_fmax(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  const bool __x_is_nan = ::cuda::__fpmp_internal_isnan(__x_hi);
  const bool __y_is_nan = ::cuda::__fpmp_internal_isnan(__y_hi);
  if (__x_is_nan && !__y_is_nan)
  {
    *__res_hi = __y_hi;
    *__res_lo = __y_lo;
    return;
  }
  if (__y_is_nan && !__x_is_nan)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
    return;
  }
  const bool __x_greater = (__x_hi > __y_hi) || (__x_hi == __y_hi && __x_lo > __y_lo);
  if (__x_greater)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
  }
  else
  {
    *__res_hi = __y_hi;
    *__res_lo = __y_lo;
  }
}

/*
 * ====================================================================
 * fmin(x, y) - smaller of two values
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Minimum fmin(x, y) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * fmin: min(x, y).  Mirror image of fmax.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_fmin(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  const bool __x_is_nan = ::cuda::__fpmp_internal_isnan(__x_hi);
  const bool __y_is_nan = ::cuda::__fpmp_internal_isnan(__y_hi);
  if (__x_is_nan && !__y_is_nan)
  {
    *__res_hi = __y_hi;
    *__res_lo = __y_lo;
    return;
  }
  if (__y_is_nan && !__x_is_nan)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
    return;
  }
  const bool __x_less = (__x_hi < __y_hi) || (__x_hi == __y_hi && __x_lo < __y_lo);
  if (__x_less)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
  }
  else
  {
    *__res_hi = __y_hi;
    *__res_lo = __y_lo;
  }
}

/*
 * ====================================================================
 * max(x, y) - std::max-like selection
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Selection max(x, y) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * max: std::max-like selection for fpmp2 values.  Uses the same
 * lexicographic ordering as fmax, but keeps std::max semantics:
 * return y only when x < y; otherwise return x (ties/unordered -> x).
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_max(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  const bool __x_less = (__x_hi < __y_hi) || (__x_hi == __y_hi && __x_lo < __y_lo);
  if (__x_less)
  {
    *__res_hi = __y_hi;
    *__res_lo = __y_lo;
  }
  else
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
  }
}

/*
 * ====================================================================
 * min(x, y) - std::min-like selection
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Selection min(x, y) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * min: std::min-like selection for fpmp2 values.  Uses the same
 * lexicographic ordering as fmin, but keeps std::min semantics:
 * return y only when y < x; otherwise return x (ties/unordered -> x).
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_min(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  const bool __y_less = (__y_hi < __x_hi) || (__y_hi == __x_hi && __y_lo < __x_lo);
  if (__y_less)
  {
    *__res_hi = __y_hi;
    *__res_lo = __y_lo;
  }
  else
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
  }
}

/*
 * ====================================================================
 * fdim(x, y) - positive difference
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Positive difference fdim(x, y) (fp32mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * fdim(x, y) = x - y when x > y, +0 otherwise, NaN when either is NaN.
 * The comparison is the lexicographic one on the renormalized pairs
 * (exact), the difference the accurate pair subtraction; a difference
 * that overflows (x = +inf or y = -inf) is +inf with a zero low limb,
 * which the subtraction would turn into (inf, NaN).  No FP64.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_fdim(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  if ((__x_hi != __x_hi) || (__y_hi != __y_hi))
  {
    *__res_hi = ::cuda::std::__fp_nan<float>();
    *__res_lo = ::cuda::std::__fp_nan<float>();
    return;
  }
  const bool __gt = (__x_hi > __y_hi) || ((__x_hi == __y_hi) && (__x_lo > __y_lo));
  if (!__gt)
  {
    *__res_hi = 0.0f;
    *__res_lo = 0.0f;
    return;
  }
  if ((__x_hi == ::cuda::std::__fp_inf<float>()) || (__y_hi == -::cuda::std::__fp_inf<float>()))
  {
    *__res_hi = ::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }
  ::cuda::__fpmp2_high_sub<float>(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
  /* A finite difference beyond FLT_MAX (fdim(FLT_MAX, -FLT_MAX)) overflows the subtraction
   * into (inf, NaN) or (NaN, NaN): it is +inf. */
  const bool __finite = *__res_hi < ::cuda::std::__fp_inf<float>();
  *__res_hi           = __finite ? *__res_hi : ::cuda::std::__fp_inf<float>();
  *__res_lo           = __finite ? *__res_lo : 0.0f;
}

/*
 * --------------------------------------------------------------------
 * Positive difference fdim(x, y) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_fdim(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  ::cuda::__fpmp2_from_double(
    ::fdim(::cuda::__fpmp2_to_double(__x_hi, __x_lo), ::cuda::__fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_2A(fdim)

/*
 * ====================================================================
 * isfinite(x), isinf(x), isnan(x), signbit(x) - classification predicates
 * ====================================================================
 * The value of a pair is hi + lo, and a loose pair out of a low-accuracy
 * operation can carry an infinity or a NaN in lo alone, or be (+inf, -inf),
 * which is a NaN.  So every predicate classifies both limbs and combines
 * the two answers by logic alone -- no arithmetic, not even the sum:
 *   isnan     : a NaN limb, or two infinities of opposite sign.
 *   isinf     : no NaN limb, and one infinite limb or two of the same sign.
 *   isfinite  : both limbs finite (the sum would overflow in float for a
 *               finite pair such as (FLT_MAX, 2^103)).
 *   signbit   : hi's sign when the limbs agree or lo is zero (so (-0, 0)
 *               keeps reporting a negative zero); otherwise the sign of the
 *               limb with the larger magnitude, and +0 when they cancel.
 */

/*
 * --------------------------------------------------------------------
 * Finite test isfinite(x) (fp32mp2) - both limbs
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_isfinite(const float __x_hi, const float __x_lo) noexcept
{
  return ::cuda::std::isfinite(__x_hi) && ::cuda::std::isfinite(__x_lo);
}

/*
 * --------------------------------------------------------------------
 * Finite test isfinite(x) (fp64mp2) - both limbs
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_isfinite(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::std::isfinite(__x_hi) && ::cuda::std::isfinite(__x_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETINT(isfinite)

/*
 * --------------------------------------------------------------------
 * Infinity test isinf(x) (fp32mp2) - both limbs
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_isinf(const float __x_hi, const float __x_lo) noexcept
{
  if (::cuda::std::isnan(__x_hi) || ::cuda::std::isnan(__x_lo))
  {
    return 0;
  }
  const bool __ih = ::cuda::std::isinf(__x_hi);
  const bool __il = ::cuda::std::isinf(__x_lo);
  return (__ih != __il) || (__ih && (::cuda::std::signbit(__x_hi) == ::cuda::std::signbit(__x_lo)));
}

/*
 * --------------------------------------------------------------------
 * Infinity test isinf(x) (fp64mp2) - both limbs
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_isinf(const double __x_hi, const double __x_lo) noexcept
{
  if (::cuda::std::isnan(__x_hi) || ::cuda::std::isnan(__x_lo))
  {
    return 0;
  }
  const bool __ih = ::cuda::std::isinf(__x_hi);
  const bool __il = ::cuda::std::isinf(__x_lo);
  return (__ih != __il) || (__ih && (::cuda::std::signbit(__x_hi) == ::cuda::std::signbit(__x_lo)));
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETINT(isinf)

/*
 * --------------------------------------------------------------------
 * NaN test isnan(x) (fp32mp2) - both limbs
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_isnan(const float __x_hi, const float __x_lo) noexcept
{
  return ::cuda::std::isnan(__x_hi) || ::cuda::std::isnan(__x_lo)
      || (::cuda::std::isinf(__x_hi) && ::cuda::std::isinf(__x_lo)
          && (::cuda::std::signbit(__x_hi) != ::cuda::std::signbit(__x_lo)));
}

/*
 * --------------------------------------------------------------------
 * NaN test isnan(x) (fp64mp2) - both limbs
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_isnan(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::std::isnan(__x_hi) || ::cuda::std::isnan(__x_lo)
      || (::cuda::std::isinf(__x_hi) && ::cuda::std::isinf(__x_lo)
          && (::cuda::std::signbit(__x_hi) != ::cuda::std::signbit(__x_lo)));
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETINT(isnan)

/*
 * --------------------------------------------------------------------
 * Sign-bit test signbit(x) (fp32mp2) - both limbs
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_signbit(const float __x_hi, const float __x_lo) noexcept
{
  const bool __sh = ::cuda::std::signbit(__x_hi);
  const bool __sl = ::cuda::std::signbit(__x_lo);
  if (__sh == __sl || __x_lo == float(0) || __x_hi != __x_hi || __x_lo != __x_lo)
  {
    return __sh;
  }
  /* Opposite signs and a nonzero lo (a loose pair): the larger magnitude decides,
   * equal magnitudes cancel to +0. */
  const float __ah = ::cuda::__fpmp_internal_fabs(__x_hi);
  const float __al = ::cuda::__fpmp_internal_fabs(__x_lo);
  return (__al > __ah) ? __sl : ((__al < __ah) ? __sh : false);
}

/*
 * --------------------------------------------------------------------
 * Sign-bit test signbit(x) (fp64mp2) - both limbs
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_signbit(const double __x_hi, const double __x_lo) noexcept
{
  const bool __sh = ::cuda::std::signbit(__x_hi);
  const bool __sl = ::cuda::std::signbit(__x_lo);
  if (__sh == __sl || __x_lo == double(0) || __x_hi != __x_hi || __x_lo != __x_lo)
  {
    return __sh;
  }
  /* Opposite signs and a nonzero lo (a loose pair): the larger magnitude decides,
   * equal magnitudes cancel to +0. */
  const double __ah = ::cuda::__fpmp_internal_fabs(__x_hi);
  const double __al = ::cuda::__fpmp_internal_fabs(__x_lo);
  return (__al > __ah) ? __sl : ((__al < __ah) ? __sh : false);
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETINT(signbit)

#endif // _CCCL_FPMP_USE_LIB
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_CLASSIFY_H
