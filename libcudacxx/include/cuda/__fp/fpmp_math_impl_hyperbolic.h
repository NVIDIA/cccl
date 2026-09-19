//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_HYPERBOLIC_H
#define _CUDA___FP_FPMP_MATH_IMPL_HYPERBOLIC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_hyperbolic.h - fpmp2 hyperbolic functions (sinh, cosh, tanh, asinh, acosh, atanh)
    ==================================================================================================
*/

#include <cuda/__fp/fpmp_math_impl.h>
// Sibling families whose kernels this family calls (exp/log/log1p are used by the hyperbolics).
#include <cuda/__fp/fpmp_math_impl_exp.h>
#include <cuda/std/__floating_point/constants.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * ====================================================================
 * tanh(x) - hyperbolic tangent
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Hyperbolic tangent tanh(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *   1. Saturation branch: |x| >= TANH_SAT  -> result = sign(x).
 *      Threshold chosen so that 1 - tanh(|x|) < 0.5 ulp at fp32mp2
 *      precision: with ulp_fp32mp2(1) ~ 2^-48 and
 *        1 - tanh(x) ~ 2*exp(-2x) for large x,
 *      we need 2*exp(-2x) <= 2^-49 ->  x >= 25*ln(2) ~ 17.33.
 *      We use 17.5 for a small safety margin.
 *
 *   2. Large-|x| branch (|x| >= 0.6554117):
 *        tanh(|x|) = 1 - 2/(exp(2|x|) + 1)
 *      Reuses the existing dedicated __fpmp2_exp<float>; the
 *      branch point is the optimal crossover (chosen so
 *      that the polynomial side stays within its 1.5-ulp envelope
 *      while keeping the exp-side argument bounded away from cancel-
 *      lation in exp(2x) - 1).
 *
 *   3. Small-|x| branch (|x| < 0.6554117): degree-22 minimax
 *      polynomial in x^2:
 *        tanh(x) = x + x * x^2 * Q(x^2)
 *      with Q(x^2) = d1 + d2*x^2 + ... + d11*x^20.
 *
 *   4. Apply sign(x) at the end. Both branches are computed on |x|
 *      using fp32mp2_low and the result is negated when x < 0,
 *      mirroring the erf code path.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_tanh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* optimal crossover between polynomial and exp paths. */
  constexpr float __branch_point = 0.6554117f;
  /* tanh(|x|) >= 1 - 0.5 ulp_fp32mp2 for |x| >= 17.33; use 17.5. */
  constexpr float __tanh_sat = 17.5f;

  const bool __is_neg  = __x_hi < 0.f;
  const float __abs_hi = __is_neg ? -__x_hi : __x_hi;

  /* ---- (1) saturation ------------------------------------------- */
  if (!(__abs_hi < __tanh_sat)) /* also catches NaN -> falls through to poly */
  {
    if (__abs_hi >= __tanh_sat)
    {
      *__res_hi = __is_neg ? -1.f : 1.f;
      *__res_lo = 0.f;
      return;
    }
    /* NaN: propagate */
    *__res_hi = __x_hi + __x_lo;
    *__res_lo = *__res_hi;
    return;
  }

  __ffloat __x(__x_hi, __x_lo);
  __ffloat __abs_a = __is_neg ? -__x : __x;

  if (__abs_hi >= __branch_point)
  {
    /* ---- (2) large-|x| branch: 1 - 2/(exp(2|x|)+1) ------------ */
    __ffloat __two_abs = __abs_a + __abs_a; /* exactly 2|x|: addition of equals */
    float __u_hi;
    float __u_lo;
    __fpmp2_exp<float>(__two_abs.hi(), __two_abs.lo(), &__u_hi, &__u_lo);
    __ffloat __denom  = __ffloat(__u_hi, __u_lo) + __ffloat(1.f);
    __ffloat __r      = __ffloat(2.f) / __denom;
    __ffloat __result = __ffloat(1.f) - __r;
    if (__is_neg)
    {
      __result = -__result;
    }
    *__res_hi = __result.hi();
    *__res_lo = __result.lo();
    return;
  }

  /* ---- (3) small-|x| branch: degree-22 polynomial in x^2 -------
   *
   * Q(x^2) = d1 + d2*x^2 + ... + d11*x^20, packed in ascending
   * degree.  Layout for the mixed-precision Horner dispatcher:
   *   - bottom 9 entries (d1..d9) are ff (full double precision):
   *     |d_n * x^{2n+1} * 2^-24| stays above the fp32mp2 ulp at
   *     these degrees, so float-rounding the coefficient would
   *     leak ~5e-9..3e-15 of absolute error into the result.
   *   - top 2 entries (d10, d11) are plain float literals (.lo == 0):
   *     |d_n * x^{2n+1} * 2^-24| <= 5e-16 (well below 0.5 ulp),
   *     so the high-degree Horner steps run in float for free.
   *
   */
  constexpr __ffloat __tanh_c[11] = {
    /* 9 low-degree ff entries (full double precision) */
    __ffloat(-0.33333333333333304), /* [0] = d1  = -1/3 */
    __ffloat(0.13333333333317149), /* [1] = d2 */
    __ffloat(-5.3968253953220913e-2), /* [2] = d3 */
    __ffloat(2.1869487987893173e-2), /* [3] = d4 */
    __ffloat(-8.863225224458907e-3), /* [4] = d5 */
    __ffloat(3.5920144108182715e-3), /* [5] = d6 */
    __ffloat(-1.4550475435045451e-3), /* [6] = d7 */
    __ffloat(5.8648819462048805e-4), /* [7] = d8 */
    __ffloat(-2.2870121144856145e-4), /* [8] = d9 (last ff term) */
    /* high-order M = 2 entries: .lo() == 0 by construction */
    __ffloat(7.709298e-5f), /* [9] = d10 */
    __ffloat(-1.596018e-5f), /* [10] = d11 (leading) */
  };

  __ffloat __a2 = __x * __x; /* x^2 (sign of x cancels) */
  __ffloat __q  = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 2>(__a2, __tanh_c);

  /* tanh(x) = x + x * x^2 * Q(x^2). Sign-preserving in x; no
   * separate sign fixup needed for the polynomial branch. */
  __ffloat __result = renormalize(__x + __x * (__a2 * __q));
  *__res_hi         = __result.hi();
  *__res_lo         = __result.lo();
} // __internal_fpmp2_tanh

/*
 * --------------------------------------------------------------------
 * Hyperbolic tangent tanh(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_tanh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(tanh, _CCCL_FPMP_TANHQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(tanh)

/*
 * ====================================================================
 * sinh(x) - hyperbolic sine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Hyperbolic sine sinh(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Reuses __fpmp2_exp<float> once and recombines its result; all arithmetic
 * runs in fp32mp2_low, with one renormalize per output.  Past the exp
 * overflow boundary (~88.7) exp returns +inf and the recombination gives
 * the IEEE-correct infinity; a NaN input propagates on its own.
 *
 * Polynomial branch covers |x| <= 0.6554 (matches the tanh crossover);
 * exp branch covers everything above.  At the crossover point we have
 * sinh(0.6554)/cosh(0.6554) = tanh(0.6554) ~= 0.575, so the exp branch
 * loses < 1 bit of precision to cancellation -- well within fp32mp2 ulp.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sinh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  constexpr float __branch_point = 0.6554117f;

  const bool __is_neg  = __x_hi < 0.f;
  const float __abs_hi = __is_neg ? -__x_hi : __x_hi;

  /* NaN propagation: any NaN component pollutes the result. */
  if (__x_hi != __x_hi || __x_lo != __x_lo)
  {
    const float __nan_val = __x_hi + __x_lo;
    *__res_hi             = __nan_val;
    *__res_lo             = __nan_val;
    return;
  }

  __ffloat __x(__x_hi, __x_lo);
  __ffloat __abs_a = __is_neg ? -__x : __x;

  if (__abs_hi >= __branch_point)
  {
    /* ---- large-|x| branch:  sinh(|x|) = (e - 1/e) / 2 ---------- */
    float __u_hi;
    float __u_lo;
    __fpmp2_exp<float>(__abs_a.hi(), __abs_a.lo(), &__u_hi, &__u_lo);
    __ffloat __e(__u_hi, __u_lo);
    __ffloat __half_e     = __e * __ffloat(0.5f);
    __ffloat __half_inv_e = __ffloat(0.5f) / __e;
    __ffloat __result     = renormalize(__half_e - __half_inv_e);
    if (__is_neg)
    {
      __result = -__result;
    }
    *__res_hi = __result.hi();
    *__res_lo = __result.lo();
    return;
  }

  /* ---- small-|x| branch: degree-23 polynomial in x^2 -------------
   *
   *   sinh(x) = x + x * x^2 * P(x^2),   P(y) = Sum_{k>=0} y^k / (2k+3)!
   *
   * Layout for the mixed-precision Horner dispatcher:
   *   - bottom 8 entries (1/3! .. 1/17!) are ff (full double precision):
   *     these contribute terms |x|^{2k+3} / (2k+3)! that, even at
   *     |x| = 0.6554, sit above ~3*10^-19 -- float-rounding the
   *     coefficient would leak ~10^-10 ... 10^-24 of absolute error,
   *     marginally near the fp32mp2 ulp at the bottom of the range.
   *   - top 3 entries (1/19!, 1/21!, 1/23!) are plain float (.lo == 0):
   *     contributions stay below 5*10^-24, so float-rounded constants
   *     are below 0.5 ulp -- same trade-off the tanh kernel makes.
   *
   * The exact rational Taylor coefficients have zero truncation
   * noise; the only source of error is fp32mp2 arithmetic.
   */
  constexpr __ffloat __sinh_c[11] = {
    /* 8 low-degree ff entries (full double precision) */
    __ffloat(1.6666666666666666e-1), /* [0] = 1/3!  = 1/6 */
    __ffloat(8.3333333333333333e-3), /* [1] = 1/5!  */
    __ffloat(1.9841269841269841e-4), /* [2] = 1/7!  */
    __ffloat(2.7557319223985891e-6), /* [3] = 1/9!  */
    __ffloat(2.5052108385441718e-8), /* [4] = 1/11! */
    __ffloat(1.6059043836821614e-10), /* [5] = 1/13! */
    __ffloat(7.6471637318198164e-13), /* [6] = 1/15! */
    __ffloat(2.8114572543455207e-15), /* [7] = 1/17! (last ff term) */
    /* high-order M = 3 entries: .lo() == 0 by construction */
    __ffloat(8.220635246624329e-18f), /* [8] = 1/19! */
    __ffloat(1.957294106339126e-20f), /* [9] = 1/21! */
    __ffloat(3.866968596927381e-23f), /* [10] = 1/23! (leading) */
  };

  __ffloat __a2 = __x * __x; /* x^2 (sign of x cancels) */
  __ffloat __q  = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 3>(__a2, __sinh_c);

  /* sinh(x) = x + x * x^2 * P(x^2).  Sign-preserving in x; no
   * separate sign fixup needed for the polynomial branch. */
  __ffloat __result = renormalize(__x + __x * (__a2 * __q));
  *__res_hi         = __result.hi();
  *__res_lo         = __result.lo();
} // __internal_fpmp2_sinh

/*
 * --------------------------------------------------------------------
 * Hyperbolic sine sinh(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sinh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(sinh, _CCCL_FPMP_SINHQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(sinh)

/*
 * ====================================================================
 * cosh(x) - hyperbolic cosine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Hyperbolic cosine cosh(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Reuses __fpmp2_exp<float> once and recombines its result; all arithmetic
 * runs in fp32mp2_low, with one renormalize per output.  Past the exp
 * overflow boundary (~88.7) exp returns +inf and the recombination gives
 * the IEEE-correct infinity; a NaN input propagates on its own.
 *
 * Branchless: cosh is even (cosh(-x) = cosh(x)), and the formula
 * `(e + 1/e) / 2` is well-conditioned everywhere -- both terms are
 * positive, so addition never cancels.  At |x| = 0 the lo parts of
 * e and 1/e carry the x^2/2 correction exactly, so no separate
 * polynomial branch is needed for small |x|.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cosh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* NaN propagation. */
  if (__x_hi != __x_hi || __x_lo != __x_lo)
  {
    const float __nan_val = __x_hi + __x_lo;
    *__res_hi             = __nan_val;
    *__res_lo             = __nan_val;
    return;
  }

  const bool __is_neg = __x_hi < 0.f;
  __ffloat __x(__x_hi, __x_lo);
  __ffloat __abs_a = __is_neg ? -__x : __x;

  float __u_hi;
  float __u_lo;
  __fpmp2_exp<float>(__abs_a.hi(), __abs_a.lo(), &__u_hi, &__u_lo);
  __ffloat __e(__u_hi, __u_lo);

  /* cosh(|x|) = 0.5*e + 0.5/e  (both terms positive; no cancellation). */
  __ffloat __half_e     = __e * __ffloat(0.5f);
  __ffloat __half_inv_e = __ffloat(0.5f) / __e;
  __ffloat __result     = renormalize(__half_e + __half_inv_e);

  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
} // __internal_fpmp2_cosh

/*
 * --------------------------------------------------------------------
 * Hyperbolic cosine cosh(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cosh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(cosh, _CCCL_FPMP_COSHQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(cosh)

/*
 * ====================================================================
 * asinh(x) - inverse hyperbolic sine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Inverse hyperbolic sine asinh(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *  Inverse hyperbolic functions on fp32mp2.
 *
 * All three implementations reduce to a single fp32mp2 log1p call,
 * with arithmetic forms chosen to avoid catastrophic cancellation
 * across the entire input domain:
 *
 *   asinh(x) = sign(x) * log1p(|x| + x^2 / (sqrt(x^2+1) + 1))
 *   acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1)))   for x >= 1
 *   atanh(x) = 0.5 * sign(x) * log1p(2|x| / (1-|x|))  for |x| >= 0.25
 *   atanh(x) = x * (1 + y*P(y)),  y = x^2,             for |x| <  0.25
 *
 * Compared to the textbook formulas log(x + sqrt(x^2+1)) and
 * log((1+x)/(1-x)), the log1p forms preserve full fp32mp2 precision
 * around x = 0 (asinh, atanh) and x = 1 (acosh) by replacing the
 * cancellation-prone subtraction "(x + sqrt(...)) - 1" with an
 * algebraically equivalent expression whose terms have the same sign.
 *
 * For asinh, the rationalized form
 *      |x| + sqrt(x^2+1) - 1 = |x| + x^2 / (sqrt(x^2+1) + 1)
 * sidesteps the subtraction entirely (both summands are >= 0).
 *
 * For acosh, x^2-1 is computed as (x-1)*(x+1) -- both factors are
 * well-conditioned (x-1 >= 0, x+1 >= 2), so the product carries the
 * full fp32mp2 precision of the difference even at x ~= 1,
 *
 * For atanh, the log1p form
 *      0.5 * log1p(2|x|/(1-|x|))
 * relies on a fast-method divide that costs ~1 ulp in the log1p
 * argument; that error becomes ~eps/|x| relative on the final result
 * and dominates as |x| -> 0.  A degree-23 Taylor polynomial in y=x^2
 * (i.e., 12 coefficients 1/3, 1/5, ..., 1/25) covers |x| < 0.25 with
 * truncation noise below fp32mp2 ulp, bypassing the divide and
 * delivering full precision near zero.  The branch range is
 * deliberately narrow to keep most threads on the log1p path and
 * limit warp divergence.  Above |x| = 1 the divisor -> 0+ and
 * log1p(+inf) saturates to +inf, so the IEEE-754 limits propagate
 * naturally without explicit guards.
 *
 * Large-argument paths (asinh, acosh) switch to log(2|x|) once |x|
 * exceeds 2^25 ~= 3.4e7.  The dropped 1/(4x^2) correction is below
 * fp32mp2 ulp from there on, and the early switch sidesteps the
 * lo-limb ulp accumulation that the `(x-1)(x+1)` / `x^2+1` chains
 * would suffer at very large |x|.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_asinh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* NaN propagation. */
  if (__x_hi != __x_hi || __x_lo != __x_lo)
  {
    const float __nan_val = __x_hi + __x_lo;
    *__res_hi             = __nan_val;
    *__res_lo             = __nan_val;
    return;
  }

  /* asinh is odd: handle +-inf via the sign branch.
   * asinh(+-inf) = +-inf. */
  const bool __is_neg  = __x_hi < 0.0f;
  const float __abs_hi = __is_neg ? -__x_hi : __x_hi;

  if (__abs_hi == ::cuda::std::__fp_inf<float>())
  {
    *__res_hi = __is_neg ? -::cuda::std::__fp_inf<float>() : ::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }

  /* asinh(+-0) = +-0. */
  if (__abs_hi == 0.0f && __x_lo == 0.0f)
  {
    *__res_hi = __x_hi; /* preserves signed zero */
    *__res_lo = 0.0f;
    return;
  }

  __ffloat __x(__x_hi, __x_lo);
  __ffloat __abs_a = __is_neg ? -__x : __x;

  /* Crossover threshold: above 2^25 we switch to the asymptotic
   * form
   *      asinh(x) = log(2|x|) + 1/(4x^2) - 3/(32x^4) + ...
   * and drop everything past the leading term.  At |x| = 2^25 the
   * dropped 1/(4x^2) is below 1.2*10^-17 relative -- comfortably
   * under fp32mp2 ulp.  Switching this early (rather than waiting
   * for x^2 to overflow at |x| ~= 1.8e19) avoids the precision loss
   * that the `|x| + x^2/(sqrt(x^2+1)+1)` chain accumulates at very
   * large |x|: each step (mul, sqrt, fast div, sum) bleeds 1-2
   * ulps into the lo limb and the absolute error survives the
   * subsequent log() because the chain produces a value of order
   * |x|^2 before the log compresses it back to log(2|x|), so the
   * lo errors don't shrink with the result.  Empirically, the
   * else branch loses ~10 bits at |x| ~ 2^60; the asymptotic form
   * is exact to fp32mp2 ulp throughout [2^25, FLT_MAX]. */
  constexpr float __large_asinh = 0x1.0p+25f;

  __ffloat __result;
  if (__abs_hi > __large_asinh)
  {
    constexpr __ffloat __ln2(0x1.62e42fefa39efp-1);
    float __l_hi;
    float __l_lo;
    __fpmp2_log<float>(__abs_a.hi(), __abs_a.lo(), &__l_hi, &__l_lo);
    __result = renormalize(__ffloat(__l_hi, __l_lo) + __ln2);
  }
  else
  {
    /* t = |x| + x^2 / (sqrt(x^2+1) + 1) ; result = log1p(t).
     * Both summands of t are non-negative -- no cancellation.
     * For |x| -> 0:  t ~= |x| + x^2/2,
     *   log1p(t) = |x| - |x|^3/6 + ... -> asinh series.
     * For |x| -> inf (within LARGE):  t ~= 2|x|,
     *   log1p(2|x|) = log(1 + 2|x|) ~= log(2|x|).
     *
     * Use accurate add for x^2+1: when |x| is small the +1
     * dominates and we want the lo to carry x^2 to full
     * fp32mp2 precision -- the same reasoning as in log1p. */
    __ffloat __a2   = __abs_a * __abs_a;
    __ffloat __a2p1 = add<fpmp2_accuracy::high>(__a2, 1.0f);
    float __s_hi;
    float __s_lo;
    __fpmp2_sqrt<float>(__a2p1.hi(), __a2p1.lo(), &__s_hi, &__s_lo);
    __ffloat __s     = __ffloat(__s_hi, __s_lo);
    __ffloat __denom = add<fpmp2_accuracy::high>(__s, 1.0f);
    __ffloat __t     = renormalize(__abs_a + __a2 / __denom);

    float __r_hi;
    float __r_lo;
    __fpmp2_log1p<float>(__t.hi(), __t.lo(), &__r_hi, &__r_lo);
    __result = __ffloat(__r_hi, __r_lo);
  }

  if (__is_neg)
  {
    __result = -__result;
  }
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
} // __internal_fpmp2_asinh

/*
 * --------------------------------------------------------------------
 * Inverse hyperbolic sine asinh(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_asinh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(asinh, _CCCL_FPMP_ASINHQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(asinh)

/*
 * ====================================================================
 * acosh(x) - inverse hyperbolic cosine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Inverse hyperbolic cosine acosh(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_acosh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* NaN propagation. */
  if (__x_hi != __x_hi || __x_lo != __x_lo)
  {
    const float __nan_val = __x_hi + __x_lo;
    *__res_hi             = __nan_val;
    *__res_lo             = __nan_val;
    return;
  }

  /* Domain: x >= 1.  Anything strictly below produces NaN.
   * Use lexicographic compare on (hi, lo) to capture x = 1 with
   * a negative lo (i.e., x < 1 by a sub-ulp amount). */
  if (__x_hi < 1.0f || (__x_hi == 1.0f && __x_lo < 0.0f))
  {
    *__res_hi = ::cuda::std::__fp_nan<float>();
    *__res_lo = ::cuda::std::__fp_nan<float>();
    return;
  }

  /* acosh(+inf) = +inf. */
  if (__x_hi == ::cuda::std::__fp_inf<float>())
  {
    *__res_hi = ::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }

  /* acosh(1) = 0 exactly. */
  if (__x_hi == 1.0f && __x_lo == 0.0f)
  {
    *__res_hi = 0.0f;
    *__res_lo = 0.0f;
    return;
  }

  __ffloat __x(__x_hi, __x_lo);

  /* Crossover threshold: above 2^25 we switch to the asymptotic
   * form
   *      acosh(x) = log(2x) - 1/(4x^2) - 3/(32x^4) - ...
   * dropping everything past the leading log(2x).  At |x| = 2^25
   * the omitted 1/(4x^2) sits at 2.2*10^-16 absolute (~=1.2*10^-17
   * relative against log(2x) ~= 18) -- under fp32mp2 ulp.  The
   * (x-1)*(x+1) chain that the else branch uses bleeds ~1-2 ulps
   * per step into the lo limb at large |x|, and those absolute
   * errors don't shrink through the subsequent log compression,
   * so accuracy degrades to ~36 bits near |x| ~ 2^60.  Switching
   * this early restores fp32mp2 ulp throughout [2^25, FLT_MAX]. */
  constexpr float __large_acosh = 0x1.0p+25f;

  __ffloat __result;
  if (__x_hi > __large_acosh)
  {
    /* Asymptotic form: acosh(x) ~= log(2x) = log(x) + ln(2).
     * O(1/x^2) correction is below fp32mp2 ulp at crossover. */
    constexpr __ffloat __ln2(0x1.62e42fefa39efp-1);
    float __l_hi;
    float __l_lo;
    __fpmp2_log<float>(__x.hi(), __x.lo(), &__l_hi, &__l_lo);
    __result = renormalize(__ffloat(__l_hi, __l_lo) + __ln2);
  }
  else
  {
    /* t = (x - 1) + sqrt(x^2 - 1) ; result = log1p(t).
     * For x -> 1+:  x^2-1 ~= 2(x-1) -> 0,  sqrt(x^2-1) -> sqrt(2(x-1));
     *   t ~= (x-1) + sqrt(2(x-1)),  no cancellation.
     * For x large within LARGE:  t ~= 2x-1,  log1p(2x-1) = log(2x).
     *
     * Compute x^2-1 as (x-1)*(x+1) to sidestep the catastrophic
     * cancellation in `x*x - 1` when x is close to 1.  Both
     * factors are well-conditioned (x-1 >= 0, x+1 >= 2), so the
     * product carries the full fp32mp2 precision of the
     * difference -- which is exactly what sqrt() needs to deliver
     * the bits that drive the log1p argument near the branch
     * point
     */
    __ffloat __xm1  = sub<fpmp2_accuracy::high>(__x, 1.0f);
    __ffloat __xp1  = add<fpmp2_accuracy::high>(__x, 1.0f);
    __ffloat __x2m1 = __xm1 * __xp1;
    float __s_hi;
    float __s_lo;
    __fpmp2_sqrt<float>(__x2m1.hi(), __x2m1.lo(), &__s_hi, &__s_lo);
    __ffloat __s = __ffloat(__s_hi, __s_lo);
    __ffloat __t = renormalize(__xm1 + __s);

    float __r_hi;
    float __r_lo;
    __fpmp2_log1p<float>(__t.hi(), __t.lo(), &__r_hi, &__r_lo);
    __result = __ffloat(__r_hi, __r_lo);
  }

  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
} // __internal_fpmp2_acosh

/*
 * --------------------------------------------------------------------
 * Inverse hyperbolic cosine acosh(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_acosh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(acosh, _CCCL_FPMP_ACOSHQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(acosh)

/*
 * ====================================================================
 * atanh(x) - inverse hyperbolic tangent
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Inverse hyperbolic tangent atanh(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_atanh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* NaN propagation. */
  if (__x_hi != __x_hi || __x_lo != __x_lo)
  {
    const float __nan_val = __x_hi + __x_lo;
    *__res_hi             = __nan_val;
    *__res_lo             = __nan_val;
    return;
  }

  const bool __is_neg  = __x_hi < 0.0f;
  const float __abs_hi = __is_neg ? -__x_hi : __x_hi;

  /* atanh(+-0) = +-0. */
  if (__abs_hi == 0.0f && __x_lo == 0.0f)
  {
    *__res_hi = __x_hi; /* preserves signed zero */
    *__res_lo = 0.0f;
    return;
  }

  /* atanh(+-1) = +-inf.  Strict |x| > 1 -> NaN. */
  if (__abs_hi >= 1.0f)
  {
    const float __abs_lo = __is_neg ? -__x_lo : __x_lo;
    if (__abs_hi == 1.0f && __abs_lo == 0.0f)
    {
      *__res_hi = __is_neg ? -::cuda::std::__fp_inf<float>() : ::cuda::std::__fp_inf<float>();
      *__res_lo = 0.0f;
      return;
    }
    /* |x| > 1 (including +inf): outside domain. */
    *__res_hi = ::cuda::std::__fp_nan<float>();
    *__res_lo = ::cuda::std::__fp_nan<float>();
    return;
  }

  __ffloat __x(__x_hi, __x_lo);
  __ffloat __abs_a = __is_neg ? -__x : __x;

  /* Small-|x| polynomial branch.
   *
   * Why split: the log1p form
   *     0.5 * log1p(2|x|/(1-|x|))
   * relies on a fast-method divide that introduces ~1 ulp of
   * relative error into the log1p argument; that error becomes
   * ~eps / |x| relative on the final result and dominates as
   * |x| -> 0.  The Taylor series
   *     atanh(x) = x * (1 + y*P(y)),  y = x^2,
   *     P(y) = 1/3 + y/5 + y^2/7 + ... + y^k/(2k+3) + ...
   * is sign-preserving in x (the leading factor carries the sign)
   * and avoids the divide entirely, so it delivers full fp32mp2
   * precision for small |x| with no setup error.
   *
   * Branch point 0.25 keeps the polynomial narrow (covers ~25 %
   * of the typical work range) so most threads stay on the log1p
   * path, limiting warp divergence; at |x| = 0.25 the y^11 term is
   * 0.04 * 0.0625^11 ~= 5*10^-16, below fp32mp2 ulp at atanh(0.25). */
  constexpr float __atanh_branch_point = 0.25f;
  if (__abs_hi < __atanh_branch_point)
  {
    /* P(y) = sum_{k>=0} y^k / (2k+3), packed in ascending degree.
     *   atanh_poly_c[0] = 1/3 (constant of P),
     *   atanh_poly_c[k] = 1/(2k+3),
     *   atanh_poly_c[11] = 1/25 (leading).
     * Layout for poly_eval<horner_mixed, M=4>: bottom 8 entries
     * are full ff (their contributions stay above fp32mp2 ulp at
     * the branch point), top 4 entries are plain float (.lo == 0
     * by construction; their contributions sit below 0.5 ulp). */
    constexpr __ffloat __atanh_poly_c[12] = {
      __ffloat(3.3333333333333333e-1), /* [ 0] 1/3  */
      __ffloat(2.0e-1), /* [ 1] 1/5  */
      __ffloat(1.4285714285714286e-1), /* [ 2] 1/7  */
      __ffloat(1.1111111111111111e-1), /* [ 3] 1/9  */
      __ffloat(9.0909090909090909e-2), /* [ 4] 1/11 */
      __ffloat(7.6923076923076923e-2), /* [ 5] 1/13 */
      __ffloat(6.6666666666666667e-2), /* [ 6] 1/15 */
      __ffloat(5.8823529411764706e-2), /* [ 7] 1/17  (last ff term) */
      /* high-order M = 4 entries: .lo() == 0 by construction */
      __ffloat(5.263158e-2f), /* [ 8] 1/19 */
      __ffloat(4.761905e-2f), /* [ 9] 1/21 */
      __ffloat(4.347826e-2f), /* [10] 1/23 */
      __ffloat(4.0e-2f), /* [11] 1/25  (leading) */
    };

    __ffloat __y      = __x * __x; /* x^2 (sign of x cancels) */
    __ffloat __q      = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 4>(__y, __atanh_poly_c);
    __ffloat __result = renormalize(__x + __x * (__y * __q));
    *__res_hi         = __result.hi();
    *__res_lo         = __result.lo();
    return;
  }

  /* t = 2|x| / (1 - |x|) ; result = 0.5 * log1p(t).
   *
   * For |x| -> 1-:  1 - |x| -> 0+, t -> +inf, log1p(+inf) = +inf.
   * For |x| ~= 0.25 (lower edge of this branch):  t ~= 0.667,
   *   log1p(0.667) ~= 0.511, * 0.5 = 0.2554 ~= atanh(0.25).
   *
   * Use accurate sub for 1 - |x| to capture full precision when
   * |x| is close to 1. */
  __ffloat __one_minus = sub<fpmp2_accuracy::high>(__ffloat(1.0f), __abs_a);
  __ffloat __two_abs   = __abs_a + __abs_a;
  __ffloat __t         = __two_abs / __one_minus;

  float __l_hi;
  float __l_lo;
  __fpmp2_log1p<float>(__t.hi(), __t.lo(), &__l_hi, &__l_lo);

  __ffloat __result = __ffloat(__l_hi, __l_lo) * __ffloat(0.5f);
  if (__is_neg)
  {
    __result = -__result;
  }

  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
} // __internal_fpmp2_atanh

/*
 * --------------------------------------------------------------------
 * Inverse hyperbolic tangent atanh(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_atanh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(atanh, _CCCL_FPMP_ATANHQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(atanh)

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_HYPERBOLIC_H
