//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
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

namespace cuda::experimental
{
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
  const bool __x_is_nan = __fpmp_internal_isnan(__x_hi);
  const bool __y_is_nan = __fpmp_internal_isnan(__y_hi);
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
  const bool __x_is_nan = __fpmp_internal_isnan(__x_hi);
  const bool __y_is_nan = __fpmp_internal_isnan(__y_hi);
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
 * Positive difference fdim(x, y) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_MATH_FALLBACK_2A(fdim)

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
  __fpmp2_from_double(::fdim(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_2A(fdim)

/*
 * ====================================================================
 * isfinite(x), isinf(x), isnan(x), signbit(x) - classification predicates
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Finite test isfinite(x) (fp32mp2) - hi-limb test
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_isfinite(const float __x_hi, [[maybe_unused]] const float __x_lo) noexcept
{
  return ::cuda::std::isfinite(static_cast<double>(__x_hi));
}

/*
 * --------------------------------------------------------------------
 * Finite test isfinite(x) (fp64mp2) - hi-limb test
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_isfinite(const double __x_hi, [[maybe_unused]] const double __x_lo) noexcept
{
  return ::cuda::std::isfinite(__x_hi);
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETINT(isfinite)

/*
 * --------------------------------------------------------------------
 * Infinity test isinf(x) (fp32mp2) - hi-limb test
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_isinf(const float __x_hi, [[maybe_unused]] const float __x_lo) noexcept
{
  return ::cuda::std::isinf(static_cast<double>(__x_hi));
}

/*
 * --------------------------------------------------------------------
 * Infinity test isinf(x) (fp64mp2) - hi-limb test
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_isinf(const double __x_hi, [[maybe_unused]] const double __x_lo) noexcept
{
  return ::cuda::std::isinf(__x_hi);
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETINT(isinf)

/*
 * --------------------------------------------------------------------
 * NaN test isnan(x) (fp32mp2) - hi-limb test
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_isnan(const float __x_hi, [[maybe_unused]] const float __x_lo) noexcept
{
  return ::cuda::std::isnan(static_cast<double>(__x_hi));
}

/*
 * --------------------------------------------------------------------
 * NaN test isnan(x) (fp64mp2) - hi-limb test
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_isnan(const double __x_hi, [[maybe_unused]] const double __x_lo) noexcept
{
  return ::cuda::std::isnan(__x_hi);
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETINT(isnan)

/*
 * --------------------------------------------------------------------
 * Sign-bit test signbit(x) (fp32mp2) - hi-limb test
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_signbit(const float __x_hi, [[maybe_unused]] const float __x_lo) noexcept
{
  return ::cuda::std::signbit(static_cast<double>(__x_hi));
}

/*
 * --------------------------------------------------------------------
 * Sign-bit test signbit(x) (fp64mp2) - hi-limb test
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_signbit(const double __x_hi, [[maybe_unused]] const double __x_lo) noexcept
{
  return ::cuda::std::signbit(__x_hi);
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETINT(signbit)

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_CLASSIFY_H
