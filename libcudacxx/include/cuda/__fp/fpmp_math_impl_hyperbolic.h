//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_HYPERBOLIC_H
#define _CUDA___FP_FPMP_MATH_IMPL_HYPERBOLIC_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
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
// Sibling family whose kernels this family calls (reductions and tables of exp and expm1; log and log1p).
#include <cuda/__fp/fpmp_math_impl_exp.h>
#include <cuda/std/__floating_point/constants.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * ====================================================================
 * sinh(x), cosh(x), tanh(x) (fp32mp2)
 * ====================================================================
 * All three reduce |x| as exp and expm1 do, to a multiple k of ln2/128 plus a pair phi,
 * |phi| <= ln2/256, and read the tables of those functions; none evaluates a polynomial in x.
 *
 * sinh takes e^(k ln2/128) = 2^n (1 + A) and e^(-k ln2/128) = 2^n' (1 + B) from the table of
 * expm1, which holds 2^(j/128) - 1 to full relative accuracy, and combines them with
 * sinh(phi) and cosh(phi) without a division (see the kernel). For |x| < 0.34 both powers of
 * two are 1 and the leading term is A - B, so that sinh keeps its relative accuracy down to
 * x = 0 on the same path as everywhere else.
 *
 * cosh has no cancellation: e^|x| = 2^n y comes from the tail of exp, and
 * cosh(x) = 2^(n-1) (y + 4^-n / y) costs one reciprocal.
 *
 * tanh is ((1 - z) + s) / ((1 + z) + s) with e^(2|x|) = 2^n (1 + s), z = 2^-n, from the
 * reduction of expm1 for |x| < 3/4, where s carries the relative accuracy, and
 * 1 - 2 z / (y + z) with e^(2|x|) = 2^n y from the tail of exp above. Below 2^-12 it is
 * x - x^3/3 + 2 x^5/15.
 *
 * sinh and cosh scale by 2^(n-1) at the end, so they stay finite up to their own overflow
 * threshold ln(2^129) = 89.416, past the threshold 88.72 of exp. Against MPFR the results are
 * within 6.2u^2 (sinh), 3.5u^2 (cosh) and 7.4u^2 (tanh) over the whole range (u = 2^-24, means
 * 0.3u^2, 0.3u^2 and 0.1 to 0.5u^2).
 *
 * Special values follow C99 Annex F and are answered on entry: sinh(+-0) = +-0,
 * sinh(+-inf) = +-inf, cosh(+-inf) = +inf, tanh(+-0) = +-0, tanh(+-inf) = +-1, and NaN gives
 * NaN in both limbs. sinh and tanh return x itself below 2^-25 in magnitude (the next term is
 * x^3/6 or x^3/3), which also keeps subnormal arguments exact. An overflowing result is
 * infinite with a zero low limb.
 *
 * These are the kernels of the default and low accuracy levels. sinh and tanh have a second
 * kernel for fpmp2_accuracy::high (__internal_fpmp2_high_sinh, __internal_fpmp2_high_tanh,
 * below): a Taylor or minimax polynomial in x next to zero and a refined division by e^|x| or
 * e^(2|x|) + 1 beyond, within 3.8u^2 and 2.3u^2, and slower than these past the polynomial's
 * range. They have the same special values and the same overflow threshold. cosh has one
 * kernel for every level.
 * --------------------------------------------------------------------
 */

// e^x = 2^n y for 0 <= x_hi <= 89.5: returns n, with y in [1, 2) as a pair.
[[nodiscard]] _CCCL_FPMP_CORE_API int
__internal_fpmp2_exp_mantissa(const float __x_hi, const float __x_lo, float* __y_hi, float* __y_lo) noexcept
{
  float __phi_hi, __phi_lo;
  int __n;
  const int __k = ::cuda::__internal_fpmp2_exp_reduce(
    __x_hi, __x_lo, __fpmp2_inv_ln2_128, __fpmp2_ln2_128_0, __fpmp2_ln2_128_1, __fpmp2_ln2_128_2, &__phi_hi, &__phi_lo);
  ::cuda::__internal_fpmp2_exp_tail(__phi_hi, __phi_lo, __k, __y_hi, __y_lo, &__n);
  return __n;
}

/*
 * 2^(1 - n) sinh(a) for a pair a, 2^-25 <= a_hi <= 89.5; returns n >= 0.
 *
 * With a = k ln2/128 + phi, e^(k ln2/128) = 2^n (1 + A) and e^(-k ln2/128) = 2^n' (1 + B), and
 * s = 2^(n' - n) <= 1:
 *
 *     2^(1 - n) sinh(a) = D cosh(phi) + M sinh(phi),
 *     D = (1 - s) + (A - s B),   M = (1 + s) + (A + s B).
 *
 * D carries the result (M sinh(phi) is at most half of it), so its terms are summed without
 * error; A and -s B have the same sign for n = 0. cosh(phi) - 1 = phi^2/2 + phi^4/24 and
 * sinh(phi) = phi + phi^3 (1/6 + phi^2/120) are good to 2^-50 of phi.
 */
[[nodiscard]] _CCCL_FPMP_CORE_API int
__internal_fpmp2_sinh_kernel(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept
{
  constexpr float __g_hi = 0x1.555556p-3f; // 1/6 = __g_hi + __g_lo
  constexpr float __g_lo = -0x1.555556p-28f;

  float __phi_hi, __phi_lo;
  const int __k = ::cuda::__internal_fpmp2_exp_reduce(
    __a_hi, __a_lo, __fpmp2_inv_ln2_128, __fpmp2_ln2_128_0, __fpmp2_ln2_128_1, __fpmp2_ln2_128_2, &__phi_hi, &__phi_lo);

  // sh = sinh(phi) and ch = cosh(phi) - 1 as pairs; phi^3 (1/6 + phi^2/120) = phi^2 (phi (g_hi + w))
  float __sq_lo;
  const float __sq_hi = ::cuda::__fpmp_two_mult_fma(__phi_hi, __phi_hi, &__sq_lo);
  __sq_lo             = ::cuda::__fpmp_fma_rn(2.0f * __phi_hi, __phi_lo, __sq_lo);
  const float __w     = ::cuda::__fpmp_fma_rn(__sq_hi, 0x1.111112p-7f, __g_lo);
  float __u_hi, __u_lo, __c_hi, __c_lo, __e;
  ::cuda::__internal_fpmp2_mul_unnormalized(__phi_hi, __phi_lo, __g_hi, __w, &__u_hi, &__u_lo);
  ::cuda::__internal_fpmp2_mul_unnormalized(__sq_hi, __sq_lo, __u_hi, __u_lo, &__c_hi, &__c_lo);
  const float __sh_hi = ::cuda::__fpmp_fast_two_sum(__phi_hi, __c_hi, &__e);
  const float __sh_lo = ::cuda::__fpmp_add_rn(__e, ::cuda::__fpmp_add_rn(__phi_lo, __c_lo));
  const float __ch_hi = 0.5f * __sq_hi;
  const float __ch_lo =
    ::cuda::__fpmp_fma_rn(::cuda::__fpmp_mul_rn_nocontract(__sq_hi, __sq_hi), 0x1.555556p-5f, 0.5f * __sq_lo);

  const int __kb                     = __k + 64;
  const int __kbm                    = 64 - __k;
  const int __n                      = __kb >> 7; // floor divisions
  const int __ds                     = (__kbm >> 7) - __n; // n' - n <= 0
  const __fpmp2_exp_table_entry __ta = ::cuda::__fpmp2_expm1_table(__kb & 127);
  const __fpmp2_exp_table_entry __tb = ::cuda::__fpmp2_expm1_table(__kbm & 127);
  const float __s                    = ::cuda::__internal_fpmp2_exp2i(__ds < -64 ? -64 : __ds);
  const float __sb_hi                = __s * __tb.__hi;
  const float __sb_lo                = __s * __tb.__lo;
  // 1 -+ s is exact for s >= 2^-23; a smaller s goes into the low limbs.
  const bool __s_small = __ds < -23;
  const float __s_top  = __s_small ? 0.0f : __s;
  const float __s_bot  = __s_small ? __s : 0.0f;
  // |A| >= |s B|, so the fast two-sums are exact.
  float __ue, __ve, __de, __me;
  const float __u    = ::cuda::__fpmp_fast_two_sum(__ta.__hi, -__sb_hi, &__ue);
  const float __v    = ::cuda::__fpmp_fast_two_sum(__ta.__hi, __sb_hi, &__ve);
  const float __d_hi = ::cuda::__fpmp_two_sum(1.0f - __s_top, __u, &__de);
  const float __m_hi = ::cuda::__fpmp_fast_two_sum(1.0f + __s_top, __v, &__me);
  const float __d_lo = ::cuda::__fpmp_add_rn(
    __de, ::cuda::__fpmp_add_rn(__ue, ::cuda::__fpmp_sub_rn(::cuda::__fpmp_sub_rn(__ta.__lo, __sb_lo), __s_bot)));
  const float __m_lo = ::cuda::__fpmp_add_rn(
    __me, ::cuda::__fpmp_add_rn(__ve, ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__ta.__lo, __sb_lo), __s_bot)));

  // D + D ch + M sh: D is zero or at least twice M sh, and D ch is below both.
  float __p_lo, __q_lo, __e1, __e2;
  const float __p_hi = ::cuda::__fpmp_two_mult_fma(__d_hi, __ch_hi, &__p_lo);
  __p_lo             = ::cuda::__fpmp_fma_rn(__d_hi, __ch_lo, ::cuda::__fpmp_fma_rn(__d_lo, __ch_hi, __p_lo));
  const float __q_hi = ::cuda::__fpmp_two_mult_fma(__m_hi, __sh_hi, &__q_lo);
  __q_lo             = ::cuda::__fpmp_fma_rn(__m_hi, __sh_lo, ::cuda::__fpmp_fma_rn(__m_lo, __sh_hi, __q_lo));
  float __sum        = ::cuda::__fpmp_fast_two_sum(__d_hi, __q_hi, &__e1);
  __sum              = ::cuda::__fpmp_fast_two_sum(__sum, __p_hi, &__e2);
  const float __lo   = ::cuda::__fpmp_add_rn(
    ::cuda::__fpmp_add_rn(__e1, __e2), ::cuda::__fpmp_add_rn(__d_lo, ::cuda::__fpmp_add_rn(__p_lo, __q_lo)));
  *__res_hi = ::cuda::__fpmp_fast_two_sum(__sum, __lo, __res_lo);
  return __n;
}

/*
 * --------------------------------------------------------------------
 * Hyperbolic sine sinh(x) (fp32mp2)
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sinh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  const float __a_hi = ::cuda::__fpmp_internal_fabs(__x_hi);
  // sinh(|x|) >= 2^128 above 89.416
  if (!(__a_hi >= 0x1p-25f && __a_hi <= 89.5f))
  {
    if (__a_hi < 0x1p-25f) // sinh(x) = x to 2^-52 |x|
    {
      *__res_hi = __x_hi;
      *__res_lo = __x_lo;
    }
    else if (__a_hi != __a_hi)
    {
      *__res_hi = __x_hi;
      *__res_lo = __x_hi;
    }
    else
    {
      *__res_hi = __x_hi < 0.0f ? -::cuda::std::__fp_inf<float>() : ::cuda::std::__fp_inf<float>();
      *__res_lo = 0.0f;
    }
    return;
  }
  const bool __neg = __x_hi < 0.0f;
  float __r_hi, __r_lo;
  const int __n = ::cuda::__internal_fpmp2_sinh_kernel(__a_hi, __neg ? -__x_lo : __x_lo, &__r_hi, &__r_lo);
  ::cuda::__internal_fpmp2_exp_scale(__r_hi, __r_lo, __n - 1, &__r_hi, &__r_lo);
  *__res_hi = __neg ? -__r_hi : __r_hi;
  *__res_lo = __neg ? -__r_lo : __r_lo;
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
 * --------------------------------------------------------------------
 * Hyperbolic sine sinh(x) (fp32mp2), fpmp2_accuracy::high
 * --------------------------------------------------------------------
 * Polynomial branch covers |x| < 1.25; the exponential branch covers
 * everything above, as (e - 1/e) / 2 with e = e^|x| = 2^n y taken as a
 * mantissa y in [1, 2) and an exponent:
 *
 *     sinh(|x|) = 2^(n-1) (y - 4^-n / y)
 *
 * The power of two is applied at the end, so the result stays finite up
 * to sinh's own overflow threshold ln(2^129) = 89.416, past the
 * threshold 88.72 of exp, and an overflowing result is +-inf with a zero
 * low limb.  Past 4^-n = 2^-64 the second term is below 2^-65 of the
 * first, and the scale stops there.
 *
 * The exp branch's error is exp's scaled by coth(x) plus the
 * reciprocal's scaled by 1/(e^2x - 1): at 1.25 that is 1.18 exp +
 * 0.09 div, with the refined division so that the second share is
 * 5 u^2 worst instead of 26.  The polynomial cannot take over much
 * further out: its Horner roundings scale with the x^3 term's share of
 * the result and the host model reads 3.1 u^2 up to |x| = 1.25, 4.2 at
 * 1.5 and 5.3 at 2 (the 11-term series itself is exact to 2^-60 at 2).
 * The crossover at 1.25 balances the two sides at ~3.5 u^2, and the
 * polynomial is also cheaper than exp plus a division.
 *
 * Special values follow C99 Annex F and are answered on entry:
 * sinh(+-inf) = +-inf, NaN gives NaN in both limbs, and below 2^-25 in
 * magnitude the result is x itself (the next term is x^3/6), which
 * keeps the sign of zero and subnormal arguments exact.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_sinh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  constexpr float __branch_point = 1.25f;

  const bool __is_neg  = __x_hi < 0.f;
  const float __abs_hi = ::cuda::__fpmp_internal_fabs(__x_hi);

  /* sinh(|x|) >= 2^128 above 89.416 */
  if (!(__abs_hi >= 0x1p-25f && __abs_hi <= 89.5f))
  {
    if (__abs_hi < 0x1p-25f) /* sinh(x) = x to 2^-52 |x| */
    {
      *__res_hi = __x_hi;
      *__res_lo = __x_lo;
    }
    else if (__abs_hi != __abs_hi)
    {
      *__res_hi = __x_hi;
      *__res_lo = __x_hi;
    }
    else
    {
      *__res_hi = __is_neg ? -::cuda::std::__fp_inf<float>() : ::cuda::std::__fp_inf<float>();
      *__res_lo = 0.0f;
    }
    return;
  }

  __ffloat __x(__x_hi, __x_lo);

  if (__abs_hi >= __branch_point)
  {
    /* ---- large-|x| branch:  2^(1-n) sinh(|x|) = y - 4^-n / y, n >= 1 ---- */
    float __y_hi, __y_lo, __i_hi, __i_lo, __r_hi, __r_lo;
    const int __n   = ::cuda::__internal_fpmp2_exp_mantissa(__abs_hi, __is_neg ? -__x_lo : __x_lo, &__y_hi, &__y_lo);
    const float __z = ::cuda::__internal_fpmp2_exp2i(__n > 32 ? -64 : -2 * __n);
    ::cuda::__fpmp_math_div_refined<false>(__z, 0.0f, __y_hi, __y_lo, &__i_hi, &__i_lo);
    const __ffloat __m = renormalize(__ffloat(__y_hi, __y_lo) - __ffloat(__i_hi, __i_lo));
    ::cuda::__internal_fpmp2_exp_scale(__m.hi(), __m.lo(), __n - 1, &__r_hi, &__r_lo);
    *__res_hi = __is_neg ? -__r_hi : __r_hi;
    *__res_lo = __is_neg ? -__r_lo : __r_lo;
    return;
  }

  /* ---- small-|x| branch: degree-23 polynomial in x^2 -------------
   *
   *   sinh(x) = x + x * x^2 * P(x^2),   P(y) = Sum_{k>=0} y^k / (2k+3)!
   *
   * Layout for the mixed-precision Horner dispatcher:
   *   - bottom 8 entries (1/3! .. 1/17!) are ff (full double precision):
   *     float-rounding these coefficients would leak above the fp32mp2
   *     ulp of the result somewhere in the range.
   *   - top 3 entries (1/19!, 1/21!, 1/23!) are plain float (.lo == 0):
   *     at |x| = 1.25 their float rounding contributes 2^-24 * 1.25^16 /
   *     19! = 2*10^-23 absolute, far below 0.5 ulp of sinh(1.25).
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
  /* Fast2Sum Horner: |v x^2| <= 0.08 |c_k| up to |x| = 1.25 (consecutive
   * 1/(2k+3)! ratios are below 1/20), bit-identical to the compensated form. */
  __ffloat __q = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 3>(__a2, __sinh_c);

  /* sinh(x) = x + x * x^2 * P(x^2).  Sign-preserving in x; no
   * separate sign fixup needed for the polynomial branch. */
  __ffloat __result = renormalize(__x + __x * (__a2 * __q));
  *__res_hi         = __result.hi();
  *__res_lo         = __result.lo();
} // __internal_fpmp2_high_sinh

/* fp64mp2 has one sinh for every level */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_sinh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_sinh(__x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(high_sinh)

/*
 * --------------------------------------------------------------------
 * Hyperbolic cosine cosh(x) (fp32mp2)
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cosh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  const float __a_hi = ::cuda::__fpmp_internal_fabs(__x_hi);
  // cosh(x) >= 2^128 above 89.416
  if (!(__a_hi <= 89.5f))
  {
    const bool __nan = __a_hi != __a_hi;
    *__res_hi        = __nan ? __x_hi : ::cuda::std::__fp_inf<float>();
    *__res_lo        = __nan ? __x_hi : 0.0f;
    return;
  }
  // e^|x| = 2^n y: 2^(1 - n) cosh(x) = y + 4^-n / y. Past 4^-n = 2^-64 the second term is below 2^-65 of the
  // first, and the scale stops there.
  float __y_hi, __y_lo, __q_hi, __q_lo, __e;
  const int __n   = ::cuda::__internal_fpmp2_exp_mantissa(__a_hi, __x_hi < 0.0f ? -__x_lo : __x_lo, &__y_hi, &__y_lo);
  const float __z = ::cuda::__internal_fpmp2_exp2i(__n > 32 ? -64 : -2 * __n);
  ::cuda::__fpmp2_low_div<float>(1.0f, 0.0f, __y_hi, __y_lo, &__q_hi, &__q_lo);
  const float __sum = ::cuda::__fpmp_fast_two_sum(__y_hi, __z * __q_hi, &__e);
  float __r_lo;
  const float __r_hi =
    ::cuda::__fpmp_fast_two_sum(__sum, ::cuda::__fpmp_add_rn(__e, ::cuda::__fpmp_fma_rn(__z, __q_lo, __y_lo)), &__r_lo);
  ::cuda::__internal_fpmp2_exp_scale(__r_hi, __r_lo, __n - 1, __res_hi, __res_lo);
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
 * --------------------------------------------------------------------
 * Hyperbolic tangent tanh(x) (fp32mp2)
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_tanh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  const float __a_hi = ::cuda::__fpmp_internal_fabs(__x_hi);
  if (!(__a_hi >= 0x1p-12f && __a_hi < 17.5f))
  {
    if (__a_hi < 0x1p-25f) // tanh(x) = x to 2^-51 |x|
    {
      *__res_hi = __x_hi;
      *__res_lo = __x_lo;
    }
    else if (__a_hi < 0x1p-12f)
    {
      // tanh(x) = x - x c, c = x^2 (1/3 - 2 x^2 / 15), to 2^-51 |x|: c is below 2^-25, so a float
      // holds it.
      const float __q = ::cuda::__fpmp_mul_rn(__a_hi, __a_hi);
      const float __c = ::cuda::__fpmp_mul_rn(__q, ::cuda::__fpmp_fma_rn(__q, -0x1.111112p-3f, 0x1.555556p-2f));
      *__res_hi       = ::cuda::__fpmp_fast_two_sum(__x_hi, ::cuda::__fpmp_fma_rn(-__x_hi, __c, __x_lo), __res_lo);
    }
    else if (__a_hi != __a_hi)
    {
      *__res_hi = __x_hi;
      *__res_lo = __x_hi;
    }
    else // 1 - tanh(|x|) = 2 e^(-2|x|) < 2^-49.5
    {
      *__res_hi = __x_hi < 0.0f ? -1.0f : 1.0f;
      *__res_lo = 0.0f;
    }
    return;
  }
  const bool __neg   = __x_hi < 0.0f;
  const float __b_hi = 2.0f * __a_hi; // b = 2 |x|
  const float __b_lo = __neg ? -2.0f * __x_lo : 2.0f * __x_lo;
  float __r_hi, __r_lo;
  if (__a_hi < 0.75f)
  {
    // e^b = 2^n (1 + s) with n <= 2: tanh = ((1 - z) + s) / ((1 + z) + s), z = 2^-n. The numerator is
    // s itself for n = 0.
    float __s_hi, __s_lo, __num_lo, __den_lo;
    const int __n        = ::cuda::__internal_fpmp2_expm1_reduced(__b_hi, __b_lo, &__s_hi, &__s_lo);
    const float __z      = ::cuda::__internal_fpmp2_exp2i(-__n);
    const float __num_hi = ::cuda::__fpmp_two_sum(1.0f - __z, __s_hi, &__num_lo);
    __num_lo             = ::cuda::__fpmp_add_rn(__num_lo, __s_lo);
    const float __den_hi = ::cuda::__fpmp_fast_two_sum(1.0f + __z, __s_hi, &__den_lo);
    __den_lo             = ::cuda::__fpmp_add_rn(__den_lo, __s_lo);
    ::cuda::__fpmp2_div<float>(__num_hi, __num_lo, __den_hi, __den_lo, &__r_hi, &__r_lo);
  }
  else
  {
    // e^b = 2^n y: tanh = 1 - 2 z / (y + z), z = 2^-n <= 1/4, a quotient of at most 0.37
    float __y_hi, __y_lo, __den_lo, __w_hi, __w_lo, __e;
    const int __n        = ::cuda::__internal_fpmp2_exp_mantissa(__b_hi, __b_lo, &__y_hi, &__y_lo);
    const float __z      = ::cuda::__internal_fpmp2_exp2i(-__n);
    const float __den_hi = ::cuda::__fpmp_fast_two_sum(__y_hi, __z, &__den_lo);
    __den_lo             = ::cuda::__fpmp_add_rn(__den_lo, __y_lo);
    ::cuda::__fpmp2_low_div<float>(2.0f * __z, 0.0f, __den_hi, __den_lo, &__w_hi, &__w_lo);
    const float __sum = ::cuda::__fpmp_fast_two_sum(1.0f, -__w_hi, &__e);
    __r_hi            = ::cuda::__fpmp_fast_two_sum(__sum, ::cuda::__fpmp_sub_rn(__e, __w_lo), &__r_lo);
  }
  *__res_hi = __neg ? -__r_hi : __r_hi;
  *__res_lo = __neg ? -__r_lo : __r_lo;
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
 * --------------------------------------------------------------------
 * Hyperbolic tangent tanh(x) (fp32mp2), fpmp2_accuracy::high
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
 *      lation in exp(2x) - 1).  Just above the crossover the
 *      quotient r = 2/(e + 1) is 0.42 and its error reaches the
 *      result scaled by r/(1 - r) = 0.74, so the division is the
 *      refined one with the exact remainder (3.5 u^2 against 26 u^2
 *      worst for the generic fast division); exp's own error enters
 *      scaled by 2e/(e^2 - 1) = 0.58.
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
__internal_fpmp2_high_tanh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* optimal crossover between polynomial and exp paths. */
  constexpr float __branch_point = 0.6554117f;
  /* tanh(|x|) >= 1 - 0.5 ulp_fp32mp2 for |x| >= 17.33; use 17.5. */
  constexpr float __tanh_sat = 17.5f;

  const bool __is_neg  = __x_hi < 0.f;
  const float __abs_hi = ::cuda::__fpmp_internal_fabs(__x_hi);

  /* ---- (1) saturation, tiny arguments and NaN --------------------- */
  if (!(__abs_hi >= 0x1p-25f && __abs_hi < __tanh_sat))
  {
    if (__abs_hi < 0x1p-25f) /* tanh(x) = x to 2^-51 |x|; keeps -0 and subnormals */
    {
      *__res_hi = __x_hi;
      *__res_lo = __x_lo;
    }
    else if (__abs_hi != __abs_hi)
    {
      *__res_hi = __x_hi;
      *__res_lo = __x_hi;
    }
    else
    {
      *__res_hi = __is_neg ? -1.f : 1.f;
      *__res_lo = 0.f;
    }
    return;
  }

  __ffloat __x(__x_hi, __x_lo);
  __ffloat __abs_a = __is_neg ? -__x : __x;

  if (__abs_hi >= __branch_point)
  {
    /* ---- (2) large-|x| branch: 1 - 2/(exp(2|x|)+1) ------------ */
    __ffloat __two_abs = __abs_a * 2.0f; /* exactly 2|x| */
    float __u_hi;
    float __u_lo;
    ::cuda::__fpmp2_exp<float>(__two_abs.hi(), __two_abs.lo(), &__u_hi, &__u_lo);
    __ffloat __denom = __ffloat(__u_hi, __u_lo) + 1.0f;
    float __r_hi, __r_lo;
    ::cuda::__fpmp_math_div_refined<true>(2.0f, 0.0f, __denom.hi(), __denom.lo(), &__r_hi, &__r_lo);
    __ffloat __result = renormalize(1.0f - __ffloat(__r_hi, __r_lo));
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
  /* Fast2Sum Horner: |v x^2| <= 0.17 |c_k| for these coefficients, so it
   * is bit-identical to the compensated form and cheaper. */
  __ffloat __q = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 2>(__a2, __tanh_c);

  /* tanh(x) = x + x * x^2 * Q(x^2). Sign-preserving in x; no
   * separate sign fixup needed for the polynomial branch. */
  __ffloat __result = renormalize(__x + __x * (__a2 * __q));
  *__res_hi         = __result.hi();
  *__res_lo         = __result.lo();
} // __internal_fpmp2_high_tanh

/* fp64mp2 has one tanh for every level */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_tanh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_tanh(__x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(high_tanh)

/*
 * ====================================================================
 * asinh(x), acosh(x), atanh(x) (fp32mp2)
 * ====================================================================
 * Each forms the argument of one logarithm from the limbs, with every sum that can cancel
 * taken exactly:
 *
 *   asinh(|x|) = log1p(|x| + x^2 / (1 + r)),  r = sqrt(1 + x^2),   for |x| < 3/4
 *              = log(|x| + r)                                       up to 2^25
 *              = log(|x|) + ln2                                     beyond
 *   acosh(x)   = log1p(d + r),  d = x - 1,  r = sqrt(d (d + 2)),    for x < 5/4
 *              = log(x + r)                                         up to 2^25
 *              = log(x) + ln2                                       beyond
 *   atanh(|x|) = |x| + |x|^3 P(x^2)                                 for |x| < 1/4
 *              = log1p(2 |x| / (1 - |x|)) / 2                       up to 1
 *
 * The log1p forms keep the relative accuracy of the result where it is small (x near 0, or
 * near 1 for acosh). Where the result is above 0.69 the logarithm of the sum itself is as
 * accurate, and asinh leaves out its division there. Past 2^25 the term dropped from
 * log(2|x|) is below 2^-52, and for d < 2^-60 acosh is sqrt(2 d). Against MPFR the results
 * are within 5.5u^2 (asinh), 5u^2 (acosh) and 8.5u^2 (atanh), u = 2^-24, with means of 0.3 to
 * 0.5u^2.
 *
 * Special values follow C99 Annex F and are answered on entry: asinh(+-0) = +-0,
 * asinh(+-inf) = +-inf, acosh(1) = +0, acosh(+inf) = +inf, acosh(x < 1) = NaN,
 * atanh(+-0) = +-0, atanh(+-1) = +-inf, atanh(|x| > 1) = NaN, and NaN gives NaN in both limbs.
 * The domains of acosh and atanh are decided on both limbs: a high limb of 1 over a negative
 * low limb is below 1. asinh and atanh return x itself below 2^-25 in magnitude (the next
 * term is x^3/6 or x^3/3), which also keeps subnormal arguments exact.
 *
 * atanh has a second kernel for fpmp2_accuracy::high (__internal_fpmp2_high_atanh, below),
 * which splits the exact 2 |x| off the argument of log1p and refines the division of the
 * rest: within 3u^2. asinh and acosh have one kernel for every level.
 * --------------------------------------------------------------------
 */

// (hi, lo) + ln2 for a pair above 17: the logarithm of an argument past 2^25
_CCCL_FPMP_CORE_API void __internal_fpmp2_add_ln2(float* __hi, float* __lo) noexcept
{
  float __e;
  const float __s = ::cuda::__fpmp_fast_two_sum(*__hi, __fpmp2_ln2_hi, &__e);
  *__hi =
    ::cuda::__fpmp_fast_two_sum(__s, ::cuda::__fpmp_add_rn(__e, ::cuda::__fpmp_add_rn(*__lo, __fpmp2_ln2_lo)), __lo);
}

// 2 atanh(1 - d) = ln2 - d/2 - ln(d), to d^2/4 < 2^-52, for d = -__a_lo in (0, 2^-25]: on entry
// (*__r_hi, *__r_lo) is ln(d), below -17.
_CCCL_FPMP_CORE_API void __internal_fpmp2_atanh_pole(const float __a_lo, float* __r_hi, float* __r_lo) noexcept
{
  float __e;
  const float __s = ::cuda::__fpmp_fast_two_sum(-*__r_hi, __fpmp2_ln2_hi, &__e);
  *__r_hi         = ::cuda::__fpmp_fast_two_sum(
    __s,
    ::cuda::__fpmp_add_rn(__e, ::cuda::__fpmp_fma_rn(0.5f, __a_lo, ::cuda::__fpmp_sub_rn(__fpmp2_ln2_lo, *__r_lo))),
    __r_lo);
}

/*
 * --------------------------------------------------------------------
 * Inverse hyperbolic sine asinh(x) (fp32mp2)
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_asinh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  const float __a_hi = ::cuda::__fpmp_internal_fabs(__x_hi);
  if (!(__a_hi >= 0x1p-25f && __a_hi < ::cuda::std::__fp_inf<float>()))
  {
    if (__a_hi < 0x1p-25f) // asinh(x) = x to 2^-52 |x|
    {
      *__res_hi = __x_hi;
      *__res_lo = __x_lo;
    }
    else // +-inf, or NaN
    {
      *__res_hi = __x_hi;
      *__res_lo = (__a_hi != __a_hi) ? __x_hi : 0.0f;
    }
    return;
  }
  const bool __neg   = __x_hi < 0.0f;
  const float __a_lo = __neg ? -__x_lo : __x_lo;
  // Past 2^25 the result is log(|x|) + ln2, and from 3/4 on log(|x| + r): one logarithm, of v.
  const bool __large = __a_hi > 0x1p25f;
  float __v_hi       = __a_hi;
  float __v_lo       = __a_lo;
  float __r_hi, __r_lo;
  if (!__large)
  {
    // r = sqrt(1 + x^2)
    float __sq_lo, __w_lo, __s_hi, __s_lo, __e;
    const float __sq_hi = ::cuda::__fpmp_two_mult_fma(__a_hi, __a_hi, &__sq_lo);
    __sq_lo             = ::cuda::__fpmp_fma_rn(2.0f * __a_hi, __a_lo, __sq_lo);
    const float __w_hi  = ::cuda::__fpmp_two_sum(__sq_hi, 1.0f, &__w_lo);
    __w_lo              = ::cuda::__fpmp_add_rn(__w_lo, __sq_lo);
    ::cuda::__fpmp2_sqrt<float>(__w_hi, __w_lo, &__s_hi, &__s_lo);
    if (__a_hi < 0.75f)
    {
      // |x| + x^2 / (1 + r): the quotient is below |x|
      float __den_lo, __q_hi, __q_lo, __t_lo;
      const float __den_hi = ::cuda::__fpmp_fast_two_sum(__s_hi, 1.0f, &__den_lo);
      __den_lo             = ::cuda::__fpmp_add_rn(__den_lo, __s_lo);
      ::cuda::__fpmp2_low_div<float>(__sq_hi, __sq_lo, __den_hi, __den_lo, &__q_hi, &__q_lo);
      const float __t = ::cuda::__fpmp_fast_two_sum(__a_hi, __q_hi, &__e);
      const float __t_hi =
        ::cuda::__fpmp_fast_two_sum(__t, ::cuda::__fpmp_add_rn(__e, ::cuda::__fpmp_add_rn(__a_lo, __q_lo)), &__t_lo);
      ::cuda::__internal_fpmp2_high_log1p(__t_hi, __t_lo, &__r_hi, &__r_lo);
      *__res_hi = __neg ? -__r_hi : __r_hi;
      *__res_lo = __neg ? -__r_lo : __r_lo;
      return;
    }
    // |x| + r. r > |x|, but next to a power of two its high limb may round to the float below that of |x|.
    const float __t = ::cuda::__fpmp_two_sum(__s_hi, __a_hi, &__e);
    __v_hi =
      ::cuda::__fpmp_fast_two_sum(__t, ::cuda::__fpmp_add_rn(__e, ::cuda::__fpmp_add_rn(__s_lo, __a_lo)), &__v_lo);
  }
  ::cuda::__internal_fpmp2_log(__v_hi, __v_lo, &__r_hi, &__r_lo);
  if (__large)
  {
    ::cuda::__internal_fpmp2_add_ln2(&__r_hi, &__r_lo);
  }
  *__res_hi = __neg ? -__r_hi : __r_hi;
  *__res_lo = __neg ? -__r_lo : __r_lo;
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
 * --------------------------------------------------------------------
 * Inverse hyperbolic cosine acosh(x) (fp32mp2)
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_acosh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  if (!(__x_hi >= 1.0f && __x_hi < ::cuda::std::__fp_inf<float>()) || (__x_hi == 1.0f && __x_lo < 0.0f))
  {
    if (__x_hi == ::cuda::std::__fp_inf<float>())
    {
      *__res_hi = __x_hi;
      *__res_lo = 0.0f;
    }
    else // x < 1, or NaN
    {
      *__res_hi = (__x_hi != __x_hi) ? __x_hi : ::cuda::std::__fp_nan<float>();
      *__res_lo = *__res_hi;
    }
    return;
  }
  // Past 2^25 the result is log(x) + ln2, and from 5/4 on log(x + sqrt(d (d + 2))): one logarithm, of v.
  const bool __large = __x_hi > 0x1p25f;
  float __v_hi       = __x_hi;
  float __v_lo       = __x_lo;
  if (!__large)
  {
    // d = x - 1, exactly (x_hi - 1 is zero or at least an ulp of x_hi), and sqrt(d (d + 2))
    float __d_lo, __m_lo, __p_hi, __p_lo, __s_hi, __s_lo, __e;
    float __d_hi = ::cuda::__fpmp_two_sum(__x_hi, -1.0f, &__d_lo);
    __d_hi       = ::cuda::__fpmp_fast_two_sum(__d_hi, ::cuda::__fpmp_add_rn(__d_lo, __x_lo), &__d_lo);
    if (__d_hi < 0x1p-60f)
    {
      // acosh(1 + d) = sqrt(2 d) to 2^-62 of it. The root is taken of 2^65 d: the device's square
      // root flushes subnormal operands, which a low limb of x below 2^-127 gives.
      ::cuda::__fpmp2_sqrt<float>(0x1p65f * __d_hi, 0x1p65f * __d_lo, &__s_hi, &__s_lo);
      *__res_hi = 0x1p-32f * __s_hi;
      *__res_lo = 0x1p-32f * __s_lo;
      return;
    }
    const float __m_hi = ::cuda::__fpmp_two_sum(__d_hi, 2.0f, &__m_lo);
    __m_lo             = ::cuda::__fpmp_add_rn(__m_lo, __d_lo);
    ::cuda::__internal_fpmp2_mul_unnormalized(__d_hi, __d_lo, __m_hi, __m_lo, &__p_hi, &__p_lo);
    ::cuda::__fpmp2_sqrt<float>(__p_hi, __p_lo, &__s_hi, &__s_lo);
    if (__x_hi < 1.25f)
    {
      // d + sqrt(d (d + 2)): the root is at least d
      float __t_lo;
      const float __t = ::cuda::__fpmp_fast_two_sum(__s_hi, __d_hi, &__e);
      const float __t_hi =
        ::cuda::__fpmp_fast_two_sum(__t, ::cuda::__fpmp_add_rn(__e, ::cuda::__fpmp_add_rn(__s_lo, __d_lo)), &__t_lo);
      ::cuda::__internal_fpmp2_high_log1p(__t_hi, __t_lo, __res_hi, __res_lo);
      return;
    }
    // x + sqrt(d (d + 2)). The root is below x, but its high limb may round to the float above that of x.
    const float __t = ::cuda::__fpmp_two_sum(__x_hi, __s_hi, &__e);
    __v_hi =
      ::cuda::__fpmp_fast_two_sum(__t, ::cuda::__fpmp_add_rn(__e, ::cuda::__fpmp_add_rn(__s_lo, __x_lo)), &__v_lo);
  }
  ::cuda::__internal_fpmp2_log(__v_hi, __v_lo, __res_hi, __res_lo);
  if (__large)
  {
    ::cuda::__internal_fpmp2_add_ln2(__res_hi, __res_lo);
  }
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
 * --------------------------------------------------------------------
 * Inverse hyperbolic tangent atanh(x) (fp32mp2)
 * --------------------------------------------------------------------
 * Below 1/4 the odd Taylor series through x^25 is used: there the division of the log1p form
 * would cost more accuracy than the result has to spare (its error enters relative to the
 * result). For a high limb of +-1 over a low limb that puts |x| below 1, d = 1 - |x| is the
 * magnitude of that low limb, and atanh(|x|) = (ln(2 - d) - ln(d)) / 2 = (ln2 - d/2 - ln(d)) / 2 to d^2/8 < 2^-53;
 * the quotient 2 |x| / d would overflow for a subnormal d.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_atanh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  const float __a_hi = ::cuda::__fpmp_internal_fabs(__x_hi);
  if (__a_hi < 0.25f)
  {
    if (__a_hi < 0x1p-25f) // atanh(x) = x to 2^-51 |x|
    {
      *__res_hi = __x_hi;
      *__res_lo = __x_lo;
      return;
    }
    // atanh(x) = x + x^3 P(x^2), P(y) = 1/3 + y/5 + ... + y^11/25: the y^12 term is below
    // 2^-53 of the result. The low 8 coefficients are pairs; the contributions of the top 4
    // are below 2^-25 of the result, and those are floats.
    constexpr __ffloat __atanh_poly_c[12] = {
      __ffloat(3.3333333333333333e-1), // 1/3
      __ffloat(2.0e-1), // 1/5
      __ffloat(1.4285714285714286e-1), // 1/7
      __ffloat(1.1111111111111111e-1), // 1/9
      __ffloat(9.0909090909090909e-2), // 1/11
      __ffloat(7.6923076923076923e-2), // 1/13
      __ffloat(6.6666666666666667e-2), // 1/15
      __ffloat(5.8823529411764706e-2), // 1/17
      __ffloat(5.263158e-2f), // 1/19
      __ffloat(4.761905e-2f), // 1/21
      __ffloat(4.347826e-2f), // 1/23
      __ffloat(4.0e-2f), // 1/25
    };
    const __ffloat __x(__x_hi, __x_lo);
    const __ffloat __y = __x * __x;
    // Fast2Sum Horner: |v x^2| <= 0.07 |c_k| here, bit-identical to the compensated form.
    const __ffloat __q      = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 4>(__y, __atanh_poly_c);
    const __ffloat __result = renormalize(__x + __x * (__y * __q));
    *__res_hi               = __result.hi();
    *__res_lo               = __result.lo();
    return;
  }
  // t = 2 |x| / (1 - |x|), and the result is log1p(t) / 2.
  const bool __neg   = __x_hi < 0.0f;
  const float __a_lo = __neg ? -__x_lo : __x_lo;
  bool __pole        = false;
  float __t_hi, __t_lo;
  if (__a_hi < 1.0f)
  {
    // 1 - |x|, exactly, as a normalized pair: its limbs are comparable next to 1
    float __m_lo;
    float __m_hi = ::cuda::__fpmp_fast_two_sum(1.0f, -__a_hi, &__m_lo);
    __m_hi       = ::cuda::__fpmp_fast_two_sum(__m_hi, ::cuda::__fpmp_sub_rn(__m_lo, __a_lo), &__m_lo);
    ::cuda::__fpmp2_low_div<float>(2.0f * __a_hi, 2.0f * __a_lo, __m_hi, __m_lo, &__t_hi, &__t_lo);
  }
  else
  {
    if (!(__a_hi == 1.0f && __a_lo < 0.0f))
    {
      // +-inf at +-1; NaN beyond, and for NaN
      const bool __one = __a_hi == 1.0f && __a_lo == 0.0f;
      const float __r  = (__a_hi != __a_hi) ? __x_hi : ::cuda::std::__fp_nan<float>();
      *__res_hi        = __one ? (__neg ? -::cuda::std::__fp_inf<float>() : ::cuda::std::__fp_inf<float>()) : __r;
      *__res_lo        = __one ? 0.0f : __r;
      return;
    }
    // 1 - |x| = d is the negated low limb, and log1p of the pair (-1, d) is ln(d).
    __pole = true;
    __t_hi = -1.0f;
    __t_lo = -__a_lo;
  }
  float __r_hi, __r_lo;
  ::cuda::__internal_fpmp2_high_log1p(__t_hi, __t_lo, &__r_hi, &__r_lo);
  if (__pole)
  {
    ::cuda::__internal_fpmp2_atanh_pole(__a_lo, &__r_hi, &__r_lo);
  }
  *__res_hi = __neg ? -0.5f * __r_hi : 0.5f * __r_hi;
  *__res_lo = __neg ? -0.5f * __r_lo : 0.5f * __r_lo;
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

/*
 * --------------------------------------------------------------------
 * Inverse hyperbolic tangent atanh(x) (fp32mp2), fpmp2_accuracy::high
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_atanh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
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

  /* atanh(+-1) = +-inf.  Strict |x| > 1 -> NaN.  A high limb of +-1 with
   * a low limb pointing inwards is |x| < 1 by a sub-ulp amount and stays in
   * the domain. */
  if (__abs_hi >= 1.0f)
  {
    const float __abs_lo = __is_neg ? -__x_lo : __x_lo;
    if (__abs_hi > 1.0f || __abs_lo > 0.0f)
    {
      /* |x| > 1 (including +inf): outside domain. */
      *__res_hi = ::cuda::std::__fp_nan<float>();
      *__res_lo = ::cuda::std::__fp_nan<float>();
      return;
    }
    if (__abs_lo == 0.0f)
    {
      *__res_hi = __is_neg ? -::cuda::std::__fp_inf<float>() : ::cuda::std::__fp_inf<float>();
      *__res_lo = 0.0f;
      return;
    }
    /* |x| = 1 - d with d = -abs_lo <= 2^-25, down to a subnormal d, for which
     * the quotient of the log1p form below would overflow:
     *   atanh(|x|) = (ln(2 - d) - ln(d)) / 2 = (ln2 - d/2 - ln(d)) / 2
     * to d^2/8 < 2^-53, and log1p of the pair (-1, d) is ln(d), below -17.
     * A low-accuracy difference can carry a larger d under a high limb of 1
     * ((2^24, -0.25) - (2^24 - 1) is (1, -0.25)): that one takes the log1p
     * form, whose quotient is then in range. */
    if (__abs_lo >= -0x1p-25f)
    {
      float __r_hi, __r_lo;
      ::cuda::__fpmp2_high_log1p<float>(-1.0f, -__abs_lo, &__r_hi, &__r_lo);
      ::cuda::__internal_fpmp2_atanh_pole(__abs_lo, &__r_hi, &__r_lo);
      *__res_hi = __is_neg ? -0.5f * __r_hi : 0.5f * __r_hi;
      *__res_lo = __is_neg ? -0.5f * __r_lo : 0.5f * __r_lo;
      return;
    }
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

    __ffloat __y = __x * __x; /* x^2 (sign of x cancels) */
    /* Fast2Sum Horner: |v x^2| <= 0.07 |c_k| here, bit-identical to the
     * compensated form. */
    __ffloat __q      = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 4>(__y, __atanh_poly_c);
    __ffloat __result = renormalize(__x + __x * (__y * __q));
    *__res_hi         = __result.hi();
    *__res_lo         = __result.lo();
    return;
  }

  /* t = 2|x| / (1 - |x|) = 2|x| + 2x^2 / (1 - |x|) ; result = 0.5 * log1p(t).
   *
   * For |x| -> 1-:  1 - |x| -> 0+, t -> +inf, log1p(+inf) = +inf.
   * For |x| ~= 0.25 (lower edge of this branch):  t ~= 0.667,
   *   log1p(0.667) ~= 0.511, * 0.5 = 0.2554 ~= atanh(0.25).
   *
   * log1p passes t's relative error on scaled by t/((1+t) log1p(t)), 0.78
   * at the branch point, and the generic fast division (26 u^2 worst) made
   * this the family's worst cell (44 bits).  Splitting off the exact 2|x|
   * leaves the division only the share 2x^2/(1-|x|) : t = |x| of t
   * (0.25 at the branch point), and it is the refined one (5 u^2 worst; the
   * exact-remainder variant, 3.5 u^2, changed nothing visible here and cost
   * 10 % of the throughput).  Accurate sub for 1 - |x| to capture full
   * precision when |x| is close to 1. */
  __ffloat __one_minus = sub<fpmp2_accuracy::high>(1.0f, __abs_a);
  __ffloat __two_abs   = __abs_a * 2.0f; /* exact */
  __ffloat __two_a2    = __two_abs * __abs_a; /* 2 x^2 */
  float __q_hi, __q_lo;
  ::cuda::__fpmp_math_div_refined<false>(
    __two_a2.hi(), __two_a2.lo(), __one_minus.hi(), __one_minus.lo(), &__q_hi, &__q_lo);
  __ffloat __t = renormalize(__two_abs + __ffloat(__q_hi, __q_lo));

  float __l_hi;
  float __l_lo;
  ::cuda::__fpmp2_high_log1p<float>(__t.hi(), __t.lo(), &__l_hi, &__l_lo);

  __ffloat __result = __ffloat(__l_hi, __l_lo) * 0.5f;
  if (__is_neg)
  {
    __result = -__result;
  }

  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
} // __internal_fpmp2_high_atanh

/* fp64mp2 has one atanh for every level */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_atanh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_atanh(__x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(high_atanh)

#endif // _CCCL_FPMP_USE_LIB
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_HYPERBOLIC_H
