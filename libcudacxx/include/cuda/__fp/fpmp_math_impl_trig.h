//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_TRIG_H
#define _CUDA___FP_FPMP_MATH_IMPL_TRIG_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_trig.h - fpmp2 trigonometric functions (sin, cos, tan, asin/acos/atan, atan2, sincos, *pi)
    ==================================================================================================
*/

#include <cuda/__fp/fpmp_math_impl.h>
#include <cuda/std/numbers>
// Sibling families whose kernels this family calls (exp10 is used by trig).
#include <cuda/__fp/fpmp_math_impl_exp.h>
#include <cuda/std/__bit/countl.h> // countl_zero for the Payne-Hanek normalization

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * ====================================================================
 * Internal kernels for sin, cos and sincos (fp32mp2)
 * ====================================================================
 * Algorithm:
 *   1. Argument reduction: x = n*(pi/2) + r, |r| <= pi/4
 *      - Tiny (|x| < pi/4): no reduction
 *      - Fast (|x| < 2^20): Cody-Waite with exact error tracking
 *        via two_mult_fma + two_sum (3-piece pi/2, ~70 bits)
 *      - Large (|x| >= 2^20): Payne-Hanek using integer 2/pi table, combining
 *        x_hi and x_lo fractions in 64-bit fixed-point before converting to
 *        fp32mp2 (pure fp32 arithmetic, no fp64). Delivers ~46 bits in the
 *        reduced argument r; final precision can be lower for tan near
 *        singularities, where small input quantization is amplified by tan'.
 *   2. Evaluate sin(r) and cos(r) via Taylor polynomials in fp32mp2
 *      sin: 8 terms (x through x^15), cos: 9 terms (1 through x^16)
 *   3. Map to correct quadrant using n mod 4
 *      sincos computes both kernels; sin/cos call sincos internally
 * ====================================================================
 */

/*
 * Payne-Hanek stage 1: compute |a| * (2/pi) via integer arithmetic,
 * extract 2-bit quadrant and 62-bit unsigned fraction in [0, 1).
 * Does NOT apply the >0.5 adjustment -- caller handles that.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__internal_fpmp2_ph_frac(_FpType __a_hi, unsigned* __q_out, uint32_t* __frac_hi, uint32_t* __frac_lo) noexcept
{
  constexpr unsigned int __i2opi[] = {
    0x3c439041U,
    0xdb629599U,
    0xf534ddc0U,
    0xfc2757d1U,
    0x4e441529U,
    0xa2f9836eU,
  };

  uint32_t __ia = ::cuda::std::bit_cast<uint32_t>(__a_hi);
  uint32_t __result[7];
  uint32_t __hi;
  uint32_t __lo;
  int __iq;

  int __e = (int) ((__ia >> 23U) & 0xFFU) - 128;
  __ia    = (__ia << 8U) | 0x80000000U;
  __hi    = 0;

  for (__iq = 0; __iq < 6; __iq++)
  {
    uint64_t __p   = (uint64_t) __i2opi[__iq] * __ia + __hi;
    __result[__iq] = (uint32_t) __p;
    __hi           = (uint32_t) (__p >> 32);
  }
  __result[__iq] = __hi;

  /* Extract the window containing quadrant + fraction bits.
   * For e >= 0 (|a| >= 2): standard extraction with left shift.
   * For e < 0 (|a| < 2): extraction from idx=4 with right shift
   *   to handle small inputs without extending the table.
   */
  uint32_t __lo2;
  if (__e >= 0)
  {
    uint32_t __ue  = (uint32_t) __e;
    uint32_t __idx = 4U - (__ue >> 5U);
    __ue           = __ue & 31U;
    __hi           = __result[__idx + 2];
    __lo           = __result[__idx + 1];
    __lo2          = (__idx > 0) ? __result[__idx] : 0U;
    if (__ue != 0U)
    {
      uint32_t __q = 32U - __ue;
      __hi         = (__hi << __ue) + (__lo >> __q);
      __lo         = (__lo << __ue) + (__lo2 >> __q);
    }
  }
  else
  {
    int __r = -__e;
    __hi    = __result[6];
    __lo    = __result[5];
    __lo2   = __result[4];
    if (__r < 32)
    {
      uint32_t __q  = (uint32_t) (32 - __r);
      uint32_t __ur = (uint32_t) __r;
      __hi          = __hi >> __ur;
      __lo          = (__lo >> __ur) | (__result[6] << __q);
    }
    else
    {
      __hi = 0;
      __lo = __result[6];
    }
  }

  *__q_out   = __hi >> 30U;
  *__frac_hi = (__hi << 2U) + (__lo >> 30U);
  *__frac_lo = (__lo << 2U);
}

/*
 * Payne-Hanek stage 2: convert a 64-bit unsigned fraction in [0, 0.5)
 * (after the >0.5 adjustment) to an fp32mp2 angle by multiplying
 * by pi/2 using 64 bits of pi/4.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__internal_fpmp2_frac_to_angle(uint32_t __hi, uint32_t __lo, uint32_t __s, _FpType* __r_hi, _FpType* __r_lo) noexcept
{
  /* Normalize: shift so MSB of hi is 1.
   * countl_zero is well defined for 0 (it yields 32), unlike __builtin_clz, and lowers
   * to the clz instruction on device.
   */
  uint32_t __lz = (uint32_t) ::cuda::std::countl_zero(__hi);

  if (__lz >= 32U)
  {
    __lz += (__lo == 0U) ? 0U : (uint32_t) ::cuda::std::countl_zero(__lo);
    __hi             = __lo;
    __lo             = 0U;
    uint32_t __shift = __lz - 32U;
    if (__shift != 0U)
    {
      __hi <<= __shift;
    }
  }
  else if (__lz != 0U)
  {
    __hi = (__hi << __lz) | (__lo >> (32U - __lz));
    __lo = __lo << __lz;
  }

  /* Multiply by pi/2 using 64 bits of pi/4.
   * pi/4 = 0x0.C90FDAA2_2168C234...
   * The *2 (pi/4 -> pi/2) is in biased_exp = 127 - lz.
   */
  constexpr uint32_t __pio4_hi32 = 0xC90FDAA2U;
  constexpr uint32_t __pio4_lo32 = 0x2168C234U;

  uint64_t __p_hh = (uint64_t) __hi * __pio4_hi32;
  uint64_t __p_hl = (uint64_t) __hi * __pio4_lo32;
  uint64_t __p_lh = (uint64_t) __lo * __pio4_hi32;

  uint64_t __combined = __p_hh + (__p_hl >> 32) + (__p_lh >> 32);
  uint32_t __rhi      = (uint32_t) (__combined >> 32);
  uint32_t __rlo      = (uint32_t) __combined;

  if ((int32_t) __rhi > 0)
  {
    __rhi = (__rhi << 1) | (__rlo >> 31);
    __rlo = __rlo << 1;
    __lz++;
  }

  /* Convert to fp32mp2 */
  uint32_t __biased_exp = 127U - __lz;
  uint32_t __f1_bits    = __s | (__biased_exp << 23) | ((__rhi >> 8) & 0x7FFFFFU);

  uint32_t __rem       = (__rhi << 24) | (__rlo >> 8);
  uint32_t __rem_extra = __rlo << 24;

  if (__rem == 0U)
  {
    *__r_hi = ::cuda::std::bit_cast<_FpType>(__f1_bits);
    *__r_lo = _FpType(0);
    return;
  }

  uint32_t __rlz = (uint32_t) ::cuda::std::countl_zero(__rem);

  uint32_t __rem_norm = (__rlz > 0U) ? ((__rem << __rlz) | (__rem_extra >> (32U - __rlz))) : __rem;

  int __biased_exp2 = (int) __biased_exp - 24 - (int) __rlz;
  if (__biased_exp2 < 1)
  {
    *__r_hi = ::cuda::std::bit_cast<_FpType>(__f1_bits);
    *__r_lo = _FpType(0);
  }
  else
  {
    uint32_t __f2_bits = __s | ((uint32_t) __biased_exp2 << 23) | ((__rem_norm >> 8) & 0x7FFFFFU);
    *__r_hi            = ::cuda::std::bit_cast<_FpType>(__f1_bits);
    *__r_lo            = ::cuda::std::bit_cast<_FpType>(__f2_bits);
  }
}

/*
 * Trigonometric argument reduction for fp32mp2.
 * Returns quadrant (mod 4) and reduced argument r  in  [-pi/4, pi/4].
 *
 * Three paths:
 *   Tiny:  |x| < pi/4 -> no reduction
 *   Fast:  |x| < 2^20 -> Cody-Waite with exact error tracking
 *   Large: |x| >= 2^20 -> Payne-Hanek (integer 2/pi table)
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_trig_reduction(
  _FpType __x_hi, _FpType __x_lo, int* __quadrant, _FpType* __r_hi, _FpType* __r_lo) noexcept
{
  using __afloat = fp32mp2_high;

  _FpType __abs_hi    = (__x_hi < _FpType(0)) ? -__x_hi : __x_hi;
  uint32_t __abs_bits = ::cuda::std::bit_cast<uint32_t>(__abs_hi);

  /* No reduction for |x| < pi/4 */
  if (__abs_bits < 0x3F490FDBU)
  {
    *__quadrant = 0;
    *__r_hi     = __x_hi;
    *__r_lo     = __x_lo;
    return;
  }

  /* Inf / NaN -> return NaN, quadrant 0 */
  if (__abs_bits >= 0x7F800000U)
  {
    *__quadrant = 0;
    *__r_hi     = __x_hi - __x_hi;
    *__r_lo     = _FpType(0);
    return;
  }

  if (__abs_bits < 0x49800000U)
  {
    /* -- Fast path: Cody-Waite for |x_hi| < 2^20 --
     *
     * pi/2 split into 3 float pieces (~70 bits)
     * C1 has 2 trailing zero mantissa bits, making n*C1 exact
     * for |n| < 2^12 via two_mult_fma, and accurate for larger n.
     *
     * Error tracking: two_mult_fma gives exact n*C1 = ph + pl,
     * two_sum gives exact x_hi - ph = s + e.
     * Remaining corrections accumulated in fp32mp2_high
     * to preserve ~46 bits when s is near zero (catastrophic
     * cancellation near multiples of pi).
     */
    constexpr _FpType __c1 = _FpType(1.5707962512969971e+000);
    constexpr _FpType __c2 = _FpType(7.5497894158615964e-008);
    constexpr _FpType __c3 = _FpType(5.3903029534742384e-015);

    int __n       = __fpmp_fp2int_rn(__x_hi * _FpType(0x1.45f306p-1f));
    _FpType __n_f = __fpmp_int2fp_rn<_FpType>(__n);

    /* Exact product n*C1 = ph + pl */
    _FpType __pl;
    _FpType __ph = __fpmp_two_mult_fma(__n_f, __c1, &__pl);

    /* Exact subtraction x_hi - ph = s + e */
    _FpType __e;
    _FpType __s = __fpmp_two_sum(__x_hi, -__ph, &__e);

    /* Build result as fp32mp2_high from exact (s, e),
     * then accumulate corrections with full precision.
     */
    __afloat __result(__s, __e);
    __result = __result + __afloat(-__pl);
    __result = __result + __afloat(__x_lo);

    /* Exact product n*C2 = nC2_hi + nC2_lo via two_mult_fma */
    _FpType __n_c2_lo;
    _FpType __n_c2_hi = __fpmp_two_mult_fma(__n_f, __c2, &__n_c2_lo);
    __result          = __result - __afloat(__n_c2_hi, __n_c2_lo);

    /* n*C3 is tiny (~10^-11), single-precision product suffices */
    __result = __result + __afloat(__fpmp_mul_rn(__n_f, -__c3));

    *__quadrant = __n;
    *__r_hi     = __result.hi();
    *__r_lo     = __result.lo();
  }
  else
  {
    /* -- Slow path: |x_hi| >= 2^20 -- */

    /* Payne-Hanek: combine x_hi and x_lo 2/pi fractions in
     * 64-bit fixed-point BEFORE the pi/2 multiply to avoid
     * precision loss from floating-point cancellation.
     */
    uint32_t __fhi;
    uint32_t __flo;
    unsigned __q_hi;
    __internal_fpmp2_ph_frac(__x_hi, &__q_hi, &__fhi, &__flo);

    uint32_t __x_hi_sign = ::cuda::std::bit_cast<uint32_t>(__x_hi) & 0x80000000U;
    int __q              = (int) __q_hi;

    /* Add x_lo contribution in fixed-point.
     * |x_lo| <= |x_hi|*2^-24 can still span many quadrants,
     * and even small |x_lo| can dominate the fraction when
     * the result angle is near zero.  Handle ALL non-zero x_lo.
     */
    if (__x_lo != _FpType(0))
    {
      _FpType __abs_lo       = (__x_lo < _FpType(0)) ? -__x_lo : __x_lo;
      uint32_t __abs_lo_bits = ::cuda::std::bit_cast<uint32_t>(__abs_lo);
      bool __same_sign       = (__x_lo > _FpType(0)) == (__x_hi > _FpType(0));

      uint32_t __fhi2 = 0;
      uint32_t __flo2 = 0;
      unsigned __q_lo = 0;

      if (__abs_lo_bits >= 0x00800000U)
      {
        __internal_fpmp2_ph_frac(__abs_lo, &__q_lo, &__fhi2, &__flo2);
      }

      uint64_t __f1 = ((uint64_t) __fhi << 32) | __flo;
      uint64_t __f2 = ((uint64_t) __fhi2 << 32) | __flo2;

      if (__same_sign)
      {
        __q += (int) __q_lo;
        uint64_t __sum = __f1 + __f2;
        if (__sum < __f1)
        {
          __q++;
        }
        __fhi = (uint32_t) (__sum >> 32);
        __flo = (uint32_t) __sum;
      }
      else
      {
        __q -= (int) __q_lo;
        if (__f1 >= __f2)
        {
          __f1 -= __f2;
        }
        else
        {
          __f1 = 0ULL - (__f2 - __f1);
          __q--;
        }
        __fhi = (uint32_t) (__f1 >> 32);
        __flo = (uint32_t) __f1;
      }
    }

    uint32_t __top_bit = __fhi >> 31U;
    __q += __top_bit;
    if (__x_hi_sign != 0U)
    {
      __q = 0U - (unsigned) __q;
    }

    if (__top_bit != 0U)
    {
      __fhi = ~__fhi;
      __flo = ~__flo;
      __x_hi_sign ^= 0x80000000U;
    }

    if (__fhi == 0U && __flo == 0U)
    {
      *__quadrant = (int) __q;
      *__r_hi     = _FpType(0);
      *__r_lo     = _FpType(0);
      return;
    }

    *__quadrant = (int) __q;
    __internal_fpmp2_frac_to_angle(__fhi, __flo, __x_hi_sign, __r_hi, __r_lo);
  }
}

/*
 * Sin kernel: evaluate sin(x) for |x| <= pi/4 using fp32mp2 Taylor series.
 * sin(x) = x + x^3*Q(x^2), Q(u) = Sum Taylor coefficients from -1/3! to -1/15!.
 * Upper terms (s7..s4) in single precision, lower terms (s3..s1) in fp32mp2.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sin_kernel(_FpType __x_hi, _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  constexpr __ffloat __s1(-1.6666666666666666e-01);
  constexpr __ffloat __s2(8.3333333333333333e-03);
  constexpr __ffloat __s3(-1.9841269841269841e-04);
  constexpr float __s4(2.7557319223985893e-06f);
  constexpr float __s5(-2.5052108385441719e-08f);
  constexpr float __s6(1.6059043836821615e-10f);
  constexpr float __s7(-7.6471637318198165e-13f);

  __ffloat __x(__x_hi, __x_lo);
  __ffloat __x2 = __x * __x;
  float __x2f   = __x2.hi();

  float __qf = __s7;
  __qf       = __fpmp_fma_rn(__qf, __x2f, __s6);
  __qf       = __fpmp_fma_rn(__qf, __x2f, __s5);
  __qf       = __fpmp_fma_rn(__qf, __x2f, __s4);

  __ffloat __q = __qf * __x2 + __s3;
  __q          = __q * __x2 + __s2;
  __q          = __q * __x2 + __s1;

  __ffloat __result = renormalize(__q * __x2 * __x + __x);
  *__res_hi         = __result.hi();
  *__res_lo         = __result.lo();
}

/*
 * Cos kernel: evaluate cos(x) for |x| <= pi/4 using fp32mp2 Taylor series.
 * cos(x) = 1 + x^2*Q(x^2), Q(u) = Sum Taylor coefficients from -1/2! to 1/16!.
 * Upper terms (c8..c4) in single precision, lower terms (c3..c1) in fp32mp2.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cos_kernel(_FpType __x_hi, _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  constexpr __ffloat __c1(-5.0000000000000000e-01);
  constexpr __ffloat __c2(4.1666666666666667e-02);
  constexpr __ffloat __c3(-1.3888888888888889e-03);
  constexpr float __c4(2.4801587301587302e-05f);
  constexpr float __c5(-2.7557319223985893e-07f);
  constexpr float __c6(2.0876756987868099e-09f);
  constexpr float __c7(-1.1470745597729725e-11f);
  constexpr float __c8(4.7794773323873853e-14f);

  __ffloat __x(__x_hi, __x_lo);
  __ffloat __x2 = __x * __x;
  float __x2f   = __x2.hi();

  float __qf = __c8;
  __qf       = __fpmp_fma_rn(__qf, __x2f, __c7);
  __qf       = __fpmp_fma_rn(__qf, __x2f, __c6);
  __qf       = __fpmp_fma_rn(__qf, __x2f, __c5);
  __qf       = __fpmp_fma_rn(__qf, __x2f, __c4);

  __ffloat __q = __qf * __x2 + __c3;
  __q          = __q * __x2 + __c2;
  __q          = __q * __x2 + __c1;

  __ffloat __result = renormalize(__q * __x2 + __ffloat(_FpType(1)));
  *__res_hi         = __result.hi();
  *__res_lo         = __result.lo();
}

/*
 * ====================================================================
 * Internal kernels for asin, acos, atan and atan2 (fp32mp2)
 * ====================================================================
 *
 * All four functions are built on two shared polynomial kernels evaluated
 * in fp32mp2 arithmetic.  We evaluate them in
 * pure fp32mp2 Horner (M = 0): a coefficient as small as 2*10^-5
 * (atan c_18) still carries ~5*10^-13 of float-rounding noise when stored
 * as `float`, which is two decimals above the fp32mp2 ulp at the
 * |a| <= 1 boundary.  Mixed-precision Horner would require either a
 * finer reduction (|a| <= tan(pi/8) for atan, etc.) or refit coefficients;
 * pure-mp2 Horner buys us simplicity and full precision at the cost of
 * ~4* the float-only kernel ops.  These functions are not on the hottest
 * fp32mp2 paths, so the trade-off is favourable.
 *
 *   atan(x):   |x| > 1 -> atan(x) = sign(x)*(pi/2 - atan(1/|x|))
 *              |x| <= 1 -> polynomial Horner in x^2, 19 coefficients.
 *
 *   atan2(y,x):  octant analysis on (|y|,|x|), call atan_kernel on
 *              min/max ratio, reconstruct via pi and pi/2 anchors.
 *
 *   asin(x):   |x| < 0.575 -> polynomial in x^2, 13 coefficients;
 *                            asin(x) = x + x*(x^2*P(x^2))
 *              |x| >= 0.575 -> y = (1-|x|)/2;
 *                            asin(|x|) = pi/2 - 2*sqrty*(1 + y*P(y))
 *                            sign restored at the end.
 *
 *   acos(x):   |x| < 0.575 -> reuse asin polynomial,
 *                            acos(x) = pi/2 - asin(x)
 *              |x| >= 0.575 -> dedicated polynomial in y = 1 - |x|,
 *                            acos(|x|) = sqrt(2y)*(1 + y*P(y));
 *                            x < 0  -> acos(x) = pi - acos(|x|).
 *
 * Domain checks: NaN inputs propagate through arithmetic; |x| > 1
 * inputs to asin/acos return NaN via the sqrt of a negative y.
 * atan(+-inf) returns +-pi/2 via the 1/x reduction (1/+-inf -> +-0).
 * atan2 handles (0,0), (+-inf,+-inf) special cases explicitly.
 * ====================================================================
 */

/* ---- (kernel 1) atan on |a| <= 1, returns atan(|a|) in fp32mp2 ----
 * _FpType is unused -- the kernel is fp32mp2-only -- but keeping it a template
 * spares every translation unit that never calls it an unused-function warning. */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_atan_kernel(const fp32mp2_low& __a, fp32mp2_low* __result) noexcept
{
  using __ffloat = fp32mp2_low;

  /* 19-coefficient minimax; ascending degree.
   * Polynomial P(a^2) such that atan(a) = a*(1 + a^2*P(a^2)). */
  constexpr __ffloat __atan_c[19] = {
    __ffloat(-3.3333333333331860e-01), /* c0  */
    __ffloat(1.9999999999755019e-01), /* c1  */
    __ffloat(-1.4285714271334815e-01), /* c2  */
    __ffloat(1.1111110678749424e-01), /* c3  */
    __ffloat(-9.0909012354005225e-02), /* c4  */
    __ffloat(7.6922129305867837e-02), /* c5  */
    __ffloat(-6.6658603633512573e-02), /* c6  */
    __ffloat(5.8773077721790849e-02), /* c7  */
    __ffloat(-5.2392330054601317e-02), /* c8  */
    __ffloat(4.6739496199157994e-02), /* c9  */
    __ffloat(-4.0926382420509971e-02), /* c10 */
    __ffloat(3.4067811082715123e-02), /* c11 */
    __ffloat(-2.5826796814495994e-02), /* c12 */
    __ffloat(1.6978035834597331e-02), /* c13 */
    __ffloat(-9.1845592187165485e-03), /* c14 */
    __ffloat(3.8559749383629918e-03), /* c15 */
    __ffloat(-1.1640717779930576e-03), /* c16 */
    __ffloat(2.2302240345758510e-04), /* c17 */
    __ffloat(-2.0258553044438358e-05), /* c18 */
  };

  __ffloat __a2 = __a * __a;
  __ffloat __q  = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__a2, __atan_c);
  *__result     = renormalize(__a + __a * (__a2 * __q));
}

/* ---- (kernel 2) asin polynomial P(y); used by both asin & acos ---- */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_asin_poly(const fpmp2<_FpType>& __y, fpmp2<_FpType>* __result) noexcept
{
  using __ffloat = fp32mp2_low;

  __ffloat __y_fast(__y.hi(), __y.lo());

  /* 13-coefficient minimax; ascending degree.
   * Polynomial P(y) such that asin(z)/z - 1 ~= z^2*P(z^2) for small z,
   * and pi/2 - asin(|x|) = 2*sqrty*(1 + y*P(y)) for y = (1-|x|)/2. */
  constexpr __ffloat __asin_c[13] = {
    __ffloat(1.666666666667375e-01), /* c0  */
    __ffloat(7.499999998342270e-02), /* c1  */
    __ffloat(4.464285849810986e-02), /* c2  */
    __ffloat(3.038188875134962e-02), /* c3  */
    __ffloat(2.237350511593569e-02), /* c4  */
    __ffloat(1.733194598980628e-02), /* c5  */
    __ffloat(1.418108777515123e-02), /* c6  */
    __ffloat(1.000422754245580e-02), /* c7  */
    __ffloat(1.745227928732326e-02), /* c8  */
    __ffloat(-1.787828218369301e-02), /* c9  */
    __ffloat(6.686894879337643e-02), /* c10 */
    __ffloat(-7.620591484676952e-02), /* c11 */
    __ffloat(6.259798167646803e-02), /* c12 */
  };

  __ffloat __q = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__y_fast, __asin_c);
  fpmp2<_FpType> __res(__q.hi(), __q.lo());
  *__result = __res;
}

/* ---- (kernel 3) acos large-branch polynomial P(y); used by acos only ----
 *
 * Companion to `__internal_fpmp2_asin_poly` for the |x| >= 0.575 branch
 * of acos.  Evaluates the 13-coefficient minimax
 * P(y) such that, for y = 1 - |x|, acos(|x|) = sqrt(2y)*(1 + y*P(y)).
 * Same fp32mp2_low internal evaluation as the asin kernel -- no
 * per-op renormalisation, single conversion in/out around the call. */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_acos_poly(const fpmp2<_FpType>& __y, fpmp2<_FpType>* __result) noexcept
{
  using __ffloat = fp32mp2_low;

  __ffloat __y_fast(__y.hi(), __y.lo());

  constexpr __ffloat __acos_c[13] = {
    __ffloat(8.3333333333333329e-02), /* c0  */
    __ffloat(1.8749999999999475e-02), /* c1  */
    __ffloat(5.5803571429249681e-03), /* c2  */
    __ffloat(1.8988715243469585e-03), /* c3  */
    __ffloat(6.9913006155254860e-04), /* c4  */
    __ffloat(2.7113554445344455e-04), /* c5  */
    __ffloat(1.0911426300865435e-04), /* c6  */
    __ffloat(4.5031965455307141e-05), /* c7  */
    __ffloat(1.9480663162164715e-05), /* c8  */
    __ffloat(6.9283438595562408e-06), /* c9  */
    __ffloat(6.1185294127269731e-06), /* c10 */
    __ffloat(-1.5951212865388395e-06), /* c11 */
    __ffloat(2.7519189493111718e-06), /* c12 */
  };

  __ffloat __q = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__y_fast, __acos_c);
  fpmp2<_FpType> __res(__q.hi(), __q.lo());
  *__result = __res;
}

/*
 * ====================================================================
 * sincos(x, &s, &c) - sine and cosine of the same argument
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Sine and cosine sincos(x, &s, &c) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * sincos for fp32mp2: compute sin(x) and cos(x) simultaneously.
 * Shared argument reduction, separate sin/cos kernels on [-pi/4, pi/4],
 * quadrant-based swap and sign adjustment.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincos(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept
{
  int __quadrant;
  float __r_hi, __r_lo;
  __internal_fpmp2_trig_reduction(__x_hi, __x_lo, &__quadrant, &__r_hi, &__r_lo);

  float __s_hi, __s_lo, __c_hi, __c_lo;
  __internal_fpmp2_sin_kernel(__r_hi, __r_lo, &__s_hi, &__s_lo);
  __internal_fpmp2_cos_kernel(__r_hi, __r_lo, &__c_hi, &__c_lo);

  int __q = __quadrant & 3;
  if (__q < 0)
  {
    __q += 4;
  }

  if (__q & 1)
  {
    float __t;
    __t    = __s_hi;
    __s_hi = __c_hi;
    __c_hi = __t;
    __t    = __s_lo;
    __s_lo = __c_lo;
    __c_lo = __t;
  }
  if (__q == 1 || __q == 2)
  {
    __c_hi = -__c_hi;
    __c_lo = -__c_lo;
  }
  if (__q == 2 || __q == 3)
  {
    __s_hi = -__s_hi;
    __s_lo = -__s_lo;
  }

  *__sin_hi = __s_hi;
  *__sin_lo = __s_lo;
  *__cos_hi = __c_hi;
  *__cos_lo = __c_lo;
}

/*
 * --------------------------------------------------------------------
 * Sine and cosine sincos(x, &s, &c) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincos(
  const double __x_hi,
  const double __x_lo,
  double* __sin_hi,
  double* __sin_lo,
  double* __cos_hi,
  double* __cos_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(sin, _CCCL_FPMP_SINQ, __x_hi, __x_lo, __sin_hi, __sin_lo);
  _CCCL_FPMP_CALL_FP64MP2_MATH(cos, _CCCL_FPMP_COSQ, __x_hi, __x_lo, __cos_hi, __cos_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A_2OUT(sincos)

/*
 * ====================================================================
 * sin(x) - sine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Sine sin(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * sin for fp32mp2: calls sincos and returns only the sine.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sin(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  float __c_hi, __c_lo;
  __fpmp2_sincos(__x_hi, __x_lo, __res_hi, __res_lo, &__c_hi, &__c_lo);
}

/*
 * --------------------------------------------------------------------
 * Sine sin(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sin(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(sin, _CCCL_FPMP_SINQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(sin)

/*
 * ====================================================================
 * cos(x) - cosine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Cosine cos(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * cos for fp32mp2: calls sincos and returns only the cosine.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cos(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  float __s_hi, __s_lo;
  __fpmp2_sincos(__x_hi, __x_lo, &__s_hi, &__s_lo, __res_hi, __res_lo);
}

/*
 * --------------------------------------------------------------------
 * Cosine cos(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cos(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(cos, _CCCL_FPMP_COSQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(cos)

/*
 * ====================================================================
 * tan(x) - tangent
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Tangent tan(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Algorithm (no FP64 dependency on the hot path):
 *   1. Reduce x to r in [-pi/4, pi/4] via the shared
 *      __internal_fpmp2_trig_reduction; this also returns the quadrant
 *      index q (modulo 4).
 *   2. Evaluate sin(r) and cos(r) on the reduced interval via the
 *      shared __internal_fpmp2_sin_kernel / __internal_fpmp2_cos_kernel.
 *   3. tan has period pi, so only the LSB of q matters:
 *        q even  ->  tan(x) =  sin(r) / cos(r)
 *        q odd   ->  tan(x) = -cos(r) / sin(r)        (= -cot(r))
 *      The full quadrant-mod-4 sign dance used by sincos is unnecessary
 *      here because tan(x + pi) = tan(x) absorbs the q == 2,3 sign
 *      flips that sincos performs on its sin/cos outputs.
 *
 * One shared reduction + one + sin kernel + one cos kernel + one fp32mp2 division.  Reusing the
 * already-tuned sin/cos kernels inherits their ~46-bit accuracy
 * envelope without having to fit a separate tan polynomial.
 *
 * Singularities at x === pi/2 (mod pi) produce +-inf through the q-odd
 * branch when sin(r) underflows to zero (matches the IEEE
 * convention; signed-infinity direction follows the rounded reduced
 * argument).  Inf / NaN inputs propagate to NaN through the reduction.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_tan(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  int __quadrant;
  float __r_hi, __r_lo;
  __internal_fpmp2_trig_reduction(__x_hi, __x_lo, &__quadrant, &__r_hi, &__r_lo);

  float __s_hi, __s_lo, __c_hi, __c_lo;
  __internal_fpmp2_sin_kernel(__r_hi, __r_lo, &__s_hi, &__s_lo);
  __internal_fpmp2_cos_kernel(__r_hi, __r_lo, &__c_hi, &__c_lo);

  using __mp2_t = fpmp2<float>;
  __mp2_t __s(__s_hi, __s_lo);
  __mp2_t __c(__c_hi, __c_lo);

  __mp2_t __result = (__quadrant & 1) ? __mp2_t(-__c / __s) : __mp2_t(__s / __c);

  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Tangent tan(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_tan(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(tan, _CCCL_FPMP_TANQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(tan)

/*
 * ====================================================================
 * atan(x) - arc tangent
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Arc tangent atan(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *  ---- atan(x) ----
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_atan(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  const bool __is_neg = __x_hi < float(0);
  __ffloat __x(__x_hi, __x_lo);
  __ffloat __absx = __is_neg ? -__x : __x;

  /* |x| > 1: use atan(x) = pi/2 - atan(1/x).  This includes |x| = inf,
   * which gives 1/x = 0, atan(0) = 0, result = pi/2. */
  const bool __large = __absx.hi() > float(1);
  __ffloat __a       = __large ? (__ffloat(float(1)) / __absx) : __absx;

  __ffloat __r;
  __internal_fpmp2_atan_kernel<float>(__a, &__r);

  if (__large)
  {
    constexpr __ffloat __PIO2(1.5707963267948966); /* pi/2 split into hi+lo */
    __r = __PIO2 - __r;
  }
  if (__is_neg)
  {
    __r = -__r;
  }

  *__res_hi = __r.hi();
  *__res_lo = __r.lo();
}

/*
 * --------------------------------------------------------------------
 * Arc tangent atan(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_atan(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(atan, _CCCL_FPMP_ATANQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(atan)

/*
 * ====================================================================
 * atan2(y, x) - arc tangent of y/x, quadrant aware
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Arc tangent atan2(y, x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *  ---- atan2(y, x) ----
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_atan2(
  const float __y_hi,
  const float __y_lo,
  const float __x_hi,
  const float __x_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* Signed-zero / signed-infinity safe sign probes via the sign bit
   * (a plain `x_hi < 0` test would return false for -0.0). */
  const uint32_t __x_bits = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const uint32_t __y_bits = ::cuda::std::bit_cast<uint32_t>(__y_hi);
  const bool __x_is_neg   = (__x_bits & 0x80000000U) != 0U;
  const bool __y_is_neg   = (__y_bits & 0x80000000U) != 0U;

  /* NaN propagation: any NaN component (in either hi or lo) forces
   * a NaN result.  Use self-inequality so the test doesn't falsely
   * fire on Inf + (-Inf) intermediates. */
  const bool __x_has_nan = (__x_hi != __x_hi) || (__x_lo != __x_lo);
  const bool __y_has_nan = (__y_hi != __y_hi) || (__y_lo != __y_lo);
  if (__x_has_nan || __y_has_nan)
  {
    const float __nan_val = __x_has_nan ? (__x_hi + __x_lo) : (__y_hi + __y_lo);
    *__res_hi             = __nan_val;
    *__res_lo             = __nan_val;
    return;
  }

  __ffloat __y(__y_hi, __y_lo);
  __ffloat __x(__x_hi, __x_lo);
  __ffloat __ay = __y_is_neg ? -__y : __y;
  __ffloat __ax = __x_is_neg ? -__x : __x;

  /* |a| == +inf  <->  bit-pattern 0x7f800000 (with sign bit already
   * stripped by the abs above). */
  const bool __x_is_inf = (::cuda::std::bit_cast<uint32_t>(__ax.hi()) == 0x7f800000U);
  const bool __y_is_inf = (::cuda::std::bit_cast<uint32_t>(__ay.hi()) == 0x7f800000U);

  /* Special cases.  IEEE-754 + C99 sectionF.10.1.4 atan2 semantics:
   *   atan2(+-0, +0)    = +-0           (preserves sign of y)
   *   atan2(+-0, -0)    = +-pi
   *   atan2(+-0, x>0)   = +-0
   *   atan2(+-0, x<0)   = +-pi
   *   atan2(+-inf, +-inf)    = +-pi/4, +-3pi/4  (signs decided per quadrant)
   *   atan2(+-inf, x finite)  = +-pi/2
   *   atan2(y finite, +inf)  = +-0
   *   atan2(y finite, -inf)  = +-pi
   *
   * NOTE on signed zero handling.  The test-framework reference is
   * computed as `atan2(double(y_hi+y_lo), double(x_hi+x_lo))`, which
   * collapses each fp32mp2 input to a single fp64 value before atan2.
   * The collapsed value's sign is the sign of the *sum*, which can
   * differ from the sign of `hi` alone (e.g. `(+0,-0)` collapses to
   * `+0`, but `(-0,-0)` collapses to `-0`).  To match the reference
   * we therefore probe the collapsed sign for any argument whose
   * `hi` is zero and route the sign decisions through that. */
  constexpr __ffloat __PI(3.141592653589793);
  constexpr __ffloat __PIO2(1.5707963267948966);
  constexpr __ffloat __pio4(0.7853981633974483);
  constexpr __ffloat __PI3O4(2.356194490192345); /* 3pi/4 */

  /* Effective (collapsed) sign of y, used whenever y_hi == 0.  When
   * y_hi != 0, IEEE `y_is_neg` is the right answer because the
   * collapsed sign matches `hi`'s sign for any normal value. */
  const float __y_sum         = __y_hi + __y_lo;
  const uint32_t __y_sum_bits = ::cuda::std::bit_cast<uint32_t>(__y_sum);
  const bool __y_eff_neg      = (__y_sum_bits & 0x80000000U) != 0U;

  __ffloat __r;
  if (__ax.hi() == float(0) && __ay.hi() == float(0))
  {
    /* Both magnitudes "zero" at the high component.  The reference
     * still distinguishes +-0 by the collapsed sign of x, so honour
     * that here:  x_collapsed >= +0 -> r = +0;  x_collapsed = -0 -> r = pi
     * (the framework's `atan2(+-0, -0)` returns +-pi). */
    const float __x_sum         = __x_hi + __x_lo;
    const uint32_t __x_sum_bits = ::cuda::std::bit_cast<uint32_t>(__x_sum);
    const bool __x_eff_neg      = (__x_sum_bits & 0x80000000U) != 0U;
    __r                         = __x_eff_neg ? __PI : __ffloat(float(0));
  }
  else if (__x_is_inf && __y_is_inf)
  {
    /* Both infinite: 45deg / 135deg depending on x sign. */
    __r = __x_is_neg ? __PI3O4 : __pio4;
  }
  else if (__y_is_inf)
  {
    /* |y| = inf, |x| finite:  result = +-pi/2 (sign from y). */
    __r = __PIO2;
  }
  else if (__x_is_inf)
  {
    /* |x| = inf, |y| finite:  result = +-0 or +-pi depending on sign of x.
     * Skipping the division avoids NaN from `finite / Inf` in
     * fp32mp2's renormalisation step. */
    __r = __x_is_neg ? __PI : __ffloat(float(0));
  }
  else
  {
    /* Generic finite path: atan(num/den), then octant fixup. */
    const bool __y_gt_x = __ay.hi() > __ax.hi();
    __ffloat __num      = __y_gt_x ? __ax : __ay;
    __ffloat __den      = __y_gt_x ? __ay : __ax;
    __ffloat __t        = div<fpmp2_accuracy::def>(__num, __den);
    __internal_fpmp2_atan_kernel<float>(__t, &__t);

    if (__y_gt_x)
    {
      /* |y| > |x|:  result = +-pi/2 -/+ atan(|x|/|y|) */
      __r = __x_is_neg ? (__PIO2 + __t) : (__PIO2 - __t);
    }
    else if (__x_is_neg)
    {
      /* |y| <= |x|, x < 0:  result = pi - atan(|y|/|x|) */
      __r = __PI - __t;
    }
    else
    {
      /* |y| <= |x|, x >= 0:  result =     atan(|y|/|x|) */
      __r = __t;
    }
  }

  /* Apply sign of y (mirror across x-axis).  When y_hi is exactly
   * zero, `y_is_neg` reflects only the sign bit of `hi`, but the
   * reference's `double(y_hi+y_lo)` collapse may yield a different
   * sign, so use `y_eff_neg` for the y_hi == 0 case. */
  const bool __y_apply_neg = (__y_hi == float(0)) ? __y_eff_neg : __y_is_neg;
  if (__y_apply_neg)
  {
    __r = -__r;
  }

  *__res_hi = __r.hi();
  *__res_lo = __r.lo();
}

/*
 * --------------------------------------------------------------------
 * Arc tangent atan2(y, x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 * Note: On CUDA device, _CCCL_FPMP_ATAN2Q widens through double atan2 (no fp128 intrinsic); _CCCL_FPMP_CBRTQ is
 * reconstructed from __nv_fp128_pow.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_atan2(
  const double __y_hi,
  const double __y_lo,
  const double __x_hi,
  const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
#  if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1)
  __fpmp_fp128 __res = _CCCL_FPMP_ATAN2Q(__fpmp2_to_quad(__y_hi, __y_lo), __fpmp2_to_quad(__x_hi, __x_lo));
  __fpmp2_from_quad(__res, __res_hi, __res_lo);
#  else
  double __res = ::cuda::std::atan2(__fpmp2_to_double(__y_hi, __y_lo), __fpmp2_to_double(__x_hi, __x_lo));
  __fpmp2_from_double(__res, __res_hi, __res_lo);
#  endif
} // __internal_fpmp2_atan2

_CCCL_FPMP_MATH_DISPATCH_2A_YX(atan2)

/*
 * ====================================================================
 * asin(x) - arc sine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Arc sine asin(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *  ---- asin(x) ----
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_asin(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fpmp2<float>;

  const bool __is_neg = __x_hi < float(0);
  __ffloat __x(__x_hi, __x_lo);
  __ffloat __absx = __is_neg ? -__x : __x;

  /* Crossover at |x| ~= 0.575 (threshold is
   * the boundary above which the small-branch polynomial loses
   * conditioning and the large-branch sqrt reconstruction wins). */
  constexpr float __branch = float(0.575f);

  __ffloat __r;
  if (__absx.hi() < __branch)
  {
    /* Small branch: asin(|x|) = |x| + |x|*(|x|^2*P(|x|^2)) */
    __ffloat __a2 = __absx * __absx;
    __ffloat __p;
    __internal_fpmp2_asin_poly<float>(__a2, &__p);
    __r = renormalize(__absx + __absx * (__a2 * __p));
  }
  else
  {
    /* Large branch: y = (1 - |x|)/2,
     *   asin(|x|) = pi/2 - 2*sqrty*(1 + y*P(y))
     * sqrt(y) returns NaN for y < 0 (i.e., |x| > 1), so NaN
     * propagates through the rest of the chain naturally. */
    __ffloat __y = __ffloat(float(0.5f)) - __absx * __ffloat(float(0.5f));
    float __sy_hi, __sy_lo;
    __fpmp2_sqrt(__y.hi(), __y.lo(), &__sy_hi, &__sy_lo);
    __ffloat __sy(__sy_hi, __sy_lo);

    __ffloat __p;
    __internal_fpmp2_asin_poly<float>(__y, &__p);

    constexpr __ffloat __PIO2(1.5707963267948966);
    __r = renormalize(__PIO2 - __ffloat(float(2)) * __sy * (__ffloat(float(1)) + __y * __p));
  }

  if (__is_neg)
  {
    __r = -__r;
  }
  *__res_hi = __r.hi();
  *__res_lo = __r.lo();
}

/*
 * --------------------------------------------------------------------
 * Arc sine asin(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_asin(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(asin, _CCCL_FPMP_ASINQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(asin)

/*
 * ====================================================================
 * acos(x) - arc cosine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Arc cosine acos(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *  ---- acos(x) ----
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_acos(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fpmp2<float>;

  const bool __is_neg = __x_hi < float(0);
  __ffloat __x(__x_hi, __x_lo);
  __ffloat __absx = __is_neg ? -__x : __x;

  constexpr float __branch = float(0.575f);
  constexpr __ffloat __PI(3.141592653589793);
  constexpr __ffloat __PIO2(1.5707963267948966);

  __ffloat __r;
  if (__absx.hi() < __branch)
  {
    /* Small branch: reuse asin polynomial.
     *   acos(x) = pi/2 - asin(x)   (sign of x already in asin) */
    __ffloat __a2 = __absx * __absx;
    __ffloat __p;
    __internal_fpmp2_asin_poly<float>(__a2, &__p);
    __ffloat __asin_abs = renormalize(__absx + __absx * (__a2 * __p));
    __r                 = __is_neg ? renormalize(__PIO2 + __asin_abs) : renormalize(__PIO2 - __asin_abs);
  }
  else
  {
    /* Large branch:
     *   y = 1 - |x|;   acos(|x|) = sqrt(2y)*(1 + y*P(y))
     *   x < 0  ->  acos(x) = pi - acos(|x|)
     * Polynomial P(y) is evaluated by `__internal_fpmp2_acos_poly`
     * (analogous to `__internal_fpmp2_asin_poly` used by the small
     *  branch and by asin). */
    __ffloat __y     = __ffloat(float(1)) - __absx;
    __ffloat __two_y = __ffloat(float(2)) * __y;
    float __s_hi, __s_lo;
    __fpmp2_sqrt(__two_y.hi(), __two_y.lo(), &__s_hi, &__s_lo);
    __ffloat __s(__s_hi, __s_lo);

    __ffloat __p;
    __internal_fpmp2_acos_poly<float>(__y, &__p);
    __ffloat __acos_abs = renormalize(__s + __s * (__y * __p));
    __r                 = __is_neg ? renormalize(__PI - __acos_abs) : __acos_abs;
  }

  *__res_hi = __r.hi();
  *__res_lo = __r.lo();
}

/*
 * --------------------------------------------------------------------
 * Arc cosine acos(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_acos(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(acos, _CCCL_FPMP_ACOSQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(acos)

/*
 * ====================================================================
 * sinpi(x) - sin(pi * x)
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Sine of pi*x sinpi(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 * The CUDA intrinsic on the device, sin(pi * x) on the host.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sinpi(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __xd   = static_cast<double>(__mp2_t(__x_hi, __x_lo));
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (__r = ::sinpi(__xd);), (__r = ::cuda::std::sin(__xd * ::cuda::std::__numbers<double>::__pi());))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Sine of pi*x sinpi(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sinpi(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  double __xd = __fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (__fpmp2_from_double(::sinpi(__xd), __res_hi, __res_lo);),
    (__fpmp2_from_double(::cuda::std::sin(__xd * ::cuda::std::__numbers<double>::__pi()), __res_hi, __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_1A(sinpi)

/*
 * ====================================================================
 * cospi(x) - cos(pi * x)
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Cosine of pi*x cospi(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cospi(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __xd   = static_cast<double>(__mp2_t(__x_hi, __x_lo));
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (__r = ::cospi(__xd);), (__r = ::cuda::std::cos(__xd * ::cuda::std::__numbers<double>::__pi());))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Cosine of pi*x cospi(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cospi(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  double __xd = __fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (__fpmp2_from_double(::cospi(__xd), __res_hi, __res_lo);),
    (__fpmp2_from_double(::cuda::std::cos(__xd * ::cuda::std::__numbers<double>::__pi()), __res_hi, __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_1A(cospi)

/*
 * ====================================================================
 * sincospi(x, &s, &c) - sin(pi * x) and cos(pi * x)
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Sine and cosine of pi*x sincospi(x, &s, &c) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincospi(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __xd   = static_cast<double>(__mp2_t(__x_hi, __x_lo));
  double __sd;
  double __cd;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (::sincospi(__xd, &__sd, &__cd);), ({
                      double __xpi = __xd * ::cuda::std::__numbers<double>::__pi();
                      __sd         = ::cuda::std::sin(__xpi);
                      __cd         = ::cuda::std::cos(__xpi);
                    }))
  __mp2_t __s(__sd), __c(__cd);
  *__sin_hi = __s.hi();
  *__sin_lo = __s.lo();
  *__cos_hi = __c.hi();
  *__cos_lo = __c.lo();
}

/*
 * --------------------------------------------------------------------
 * Sine and cosine of pi*x sincospi(x, &s, &c) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincospi(
  const double __x_hi,
  const double __x_lo,
  double* __sin_hi,
  double* __sin_lo,
  double* __cos_hi,
  double* __cos_lo) noexcept
{
  double __xd = __fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    ({
      double __sd;
      double __cd;
      ::sincospi(__xd, &__sd, &__cd);
      __fpmp2_from_double(__sd, __sin_hi, __sin_lo);
      __fpmp2_from_double(__cd, __cos_hi, __cos_lo);
    }),
    ({
      double __xpi = __xd * ::cuda::std::__numbers<double>::__pi();
      __fpmp2_from_double(::cuda::std::sin(__xpi), __sin_hi, __sin_lo);
      __fpmp2_from_double(::cuda::std::cos(__xpi), __cos_hi, __cos_lo);
    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A_2OUT(sincospi)

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_TRIG_H
