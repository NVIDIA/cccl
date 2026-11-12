//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
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
#include <cuda/std/__bit/countl.h> // countl_zero for the Payne-Hanek normalization
#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/copysign.h>
#include <cuda/std/__cmath/isfinite.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/numbers>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * ====================================================================
 * Internal kernels for sin, cos and sincos (fp32mp2)
 * ====================================================================
 * Algorithm:
 *   1. Argument reduction: x = n*(pi/2) + r, |r| <= pi/4
 *      - Tiny (|x| < pi/4): no reduction
 *      - Fast (|x| < 2^20): Cody-Waite with a 4-piece pi/2 (94 bits),
 *        exact products (two_mult_fma) and error-free sums down to the
 *        third-order terms, so r keeps ~46 bits even when x lies within
 *        2^-47 |x| of a multiple of pi/2
 *      - Large (|x| >= 2^20): Payne-Hanek using an integer 2/pi table (256
 *        bits), combining x_hi and x_lo fractions in 96-bit fixed point
 *        before converting to fp32mp2 (pure fp32 arithmetic, no fp64).
 *        Delivers ~46 bits in the reduced argument r down to |r| ~ 2^-46;
 *        final precision can be lower for tan near singularities, where
 *        small input quantization is amplified by tan'.
 *   2. Evaluate sin(r) and cos(r) via Taylor polynomials in fp32mp2
 *      sin: 8 terms (x through x^15), cos: 9 terms (1 through x^16),
 *      one shared kernel (shared r^2, Fast2Sum Horner steps)
 *   3. Map to correct quadrant using n mod 4
 *      sincos computes both kernels; sin/cos call sincos internally
 * ====================================================================
 */

/*
 * Payne-Hanek stage 1: |a| * (2/pi) via integer arithmetic; returns the
 * 2-bit quadrant and a 94-bit fraction in [0, 1) as three words (f0 most
 * significant; the two lowest bits of f2 are zero).  Does NOT apply the
 * > 0.5 adjustment -- the caller handles that.
 *
 * 2/pi is held to 256 bits (eight words, least significant first).  For
 * |a| < 2^128 the fraction window sits at most 128 bits below the top of
 * the product, so three words plus one more for the shift always exist.
 * The earlier 64-bit fraction limited the reduced argument to 2^-63
 * absolute, i.e. 46 bits only for |r| >= 2^-16 (31-33 bits on the pattern
 * dataset); 94 bits keep 46 bits down to |r| ~ 2^-46.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__internal_fpmp2_ph_frac(_FpType __a_hi, unsigned* __q_out, uint32_t* __f0, uint32_t* __f1, uint32_t* __f2) noexcept
{
  constexpr unsigned int __i2opi[] = {
    0xdebbc561U,
    0xfe5163abU,
    0x3c439041U,
    0xdb629599U,
    0xf534ddc0U,
    0xfc2757d1U,
    0x4e441529U,
    0xa2f9836eU,
  };

  uint32_t __ia = ::cuda::std::bit_cast<uint32_t>(__a_hi);
  uint32_t __result[9];
  uint32_t __hi = 0;

  const int __e = (int) ((__ia >> 23U) & 0xFFU) - 128;
  __ia          = (__ia << 8U) | 0x80000000U;

  for (int __iq = 0; __iq < 8; __iq++)
  {
    uint64_t __p   = (uint64_t) __i2opi[__iq] * __ia + __hi;
    __result[__iq] = (uint32_t) __p;
    __hi           = (uint32_t) (__p >> 32);
  }
  __result[8] = __hi;

  /* Extract the window holding the quadrant and the fraction: three
   * words w3 (most significant), w2, w1, shifted left by e mod 32 for
   * e >= 0 (|a| >= 2) and right by -e for e < 0.
   */
  uint32_t __w3, __w2, __w1;
  if (__e >= 0)
  {
    const uint32_t __ue  = (uint32_t) __e;
    const uint32_t __idx = 6U - (__ue >> 5U); /* 3 <= idx <= 6 */
    __w3                 = __result[__idx + 2];
    __w2                 = __result[__idx + 1];
    __w1                 = __result[__idx];
    const uint32_t __w0  = __result[__idx - 1];
    const uint32_t __sh  = __ue & 31U;
    if (__sh != 0U)
    {
      const uint32_t __q = 32U - __sh;
      __w3               = (__w3 << __sh) | (__w2 >> __q);
      __w2               = (__w2 << __sh) | (__w1 >> __q);
      __w1               = (__w1 << __sh) | (__w0 >> __q);
    }
  }
  else
  {
    int __r = -__e;
    __w3    = __result[8];
    __w2    = __result[7];
    __w1    = __result[6];
    while (__r >= 32) /* at most four rounds (|a| >= 2^-126) */
    {
      __w1 = __w2;
      __w2 = __w3;
      __w3 = 0U;
      __r -= 32;
    }
    if (__r != 0)
    {
      const uint32_t __ur = (uint32_t) __r;
      const uint32_t __q  = 32U - __ur;
      __w1                = (__w1 >> __ur) | (__w2 << __q);
      __w2                = (__w2 >> __ur) | (__w3 << __q);
      __w3                = __w3 >> __ur;
    }
  }

  *__q_out = __w3 >> 30U;
  *__f0    = (__w3 << 2U) | (__w2 >> 30U);
  *__f1    = (__w2 << 2U) | (__w1 >> 30U);
  *__f2    = (__w1 << 2U);
}

/*
 * Payne-Hanek stage 2: convert a nonzero 96-bit unsigned fraction in
 * [0, 0.5) (after the > 0.5 adjustment) to an fp32mp2 angle by
 * multiplying by pi/2, using 96 bits of pi/4 and keeping the top 64 bits
 * of the product (16 guard bits beyond the 48 the pair holds; the low
 * limb is rounded to nearest).
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_frac_to_angle(
  uint32_t __f0, uint32_t __f1, uint32_t __f2, uint32_t __s, _FpType* __r_hi, _FpType* __r_lo) noexcept
{
  /* Normalize: shift so the MSB of f0 is 1.  countl_zero is well defined
   * for 0 (it yields 32), unlike __builtin_clz, and lowers to the clz
   * instruction on device. */
  uint32_t __lz = 0U;
  if (__f0 == 0U)
  {
    __f0 = __f1;
    __f1 = __f2;
    __f2 = 0U;
    __lz += 32U;
    if (__f0 == 0U)
    {
      __f0 = __f1;
      __f1 = 0U;
      __lz += 32U;
    }
  }
  const uint32_t __z = (uint32_t) ::cuda::std::countl_zero(__f0);
  if (__z != 0U)
  {
    __f0 = (__f0 << __z) | (__f1 >> (32U - __z));
    __f1 = (__f1 << __z) | (__f2 >> (32U - __z));
    __f2 = __f2 << __z;
    __lz += __z;
  }

  /* Multiply by pi/2 using 96 bits of pi/4 = 0x0.C90FDAA2_2168C234_C4C6628B...
   * The *2 (pi/4 -> pi/2) is in biased_exp = 127 - lz.  Partial products
   * by weight: 2^-64: f0 p0; 2^-96: f0 p1, f1 p0; 2^-128: f0 p2, f1 p1,
   * f2 p0 (only their high halves reach the 2^-96 column). */
  constexpr uint32_t __pio4_0 = 0xC90FDAA2U;
  constexpr uint32_t __pio4_1 = 0x2168C234U;
  constexpr uint32_t __pio4_2 = 0xC4C6628BU;

  const uint64_t __t00 = (uint64_t) __f0 * __pio4_0;
  const uint64_t __t01 = (uint64_t) __f0 * __pio4_1;
  const uint64_t __t10 = (uint64_t) __f1 * __pio4_0;
  const uint64_t __t02 = (uint64_t) __f0 * __pio4_2;
  const uint64_t __t11 = (uint64_t) __f1 * __pio4_1;
  const uint64_t __t20 = (uint64_t) __f2 * __pio4_0;
  const uint64_t __col =
    (__t01 & 0xFFFFFFFFULL) + (__t10 & 0xFFFFFFFFULL) + (__t02 >> 32) + (__t11 >> 32) + (__t20 >> 32);
  const uint64_t __combined = __t00 + (__t01 >> 32) + (__t10 >> 32) + (__col >> 32);

  uint32_t __rhi = (uint32_t) (__combined >> 32);
  uint32_t __rlo = (uint32_t) __combined;

  if ((int32_t) __rhi > 0)
  {
    __rhi = (__rhi << 1) | (__rlo >> 31);
    __rlo = __rlo << 1;
    __lz++;
  }

  /* Convert to fp32mp2 */
  const uint32_t __biased_exp = 127U - __lz;
  const uint32_t __f1_bits    = __s | (__biased_exp << 23) | ((__rhi >> 8) & 0x7FFFFFU);

  const uint32_t __rem       = (__rhi << 24) | (__rlo >> 8);
  const uint32_t __rem_extra = __rlo << 24;

  if (__rem == 0U)
  {
    *__r_hi = ::cuda::std::bit_cast<_FpType>(__f1_bits);
    *__r_lo = _FpType(0);
    return;
  }

  const uint32_t __rlz = (uint32_t) ::cuda::std::countl_zero(__rem);
  uint32_t __rem_norm  = (__rlz > 0U) ? ((__rem << __rlz) | (__rem_extra >> (32U - __rlz))) : __rem;
  int __biased_exp2    = (int) __biased_exp - 24 - (int) __rlz;

  /* Round the low limb to nearest: 24 of its 32 bits are kept. */
  const uint64_t __rr = (uint64_t) __rem_norm + 0x80U;
  if ((__rr >> 32) != 0U)
  {
    __rem_norm = 0x80000000U;
    __biased_exp2 += 1;
  }
  else
  {
    __rem_norm = (uint32_t) __rr;
  }

  if (__biased_exp2 < 1)
  {
    *__r_hi = ::cuda::std::bit_cast<_FpType>(__f1_bits);
    *__r_lo = _FpType(0);
  }
  else
  {
    const uint32_t __f2_bits = __s | ((uint32_t) __biased_exp2 << 23) | ((__rem_norm >> 8) & 0x7FFFFFU);
    *__r_hi                  = ::cuda::std::bit_cast<_FpType>(__f1_bits);
    *__r_lo                  = ::cuda::std::bit_cast<_FpType>(__f2_bits);
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
     * pi/2 = c1 + c2 + c3 + c4 as four floats (94 bits, residual
     * 8.5e-32); every n * c_k is split exactly with two_mult_fma, so
     * the reduction is exact down to its third-order terms:
     *
     * First order.  x_hi - ph is exact (Sterbenz: x_hi and n * pi/2
     * agree to within pi/4), and the three remaining first-order terms
     * -pl, x_lo, -p2h are folded with error-free sums, so
     * (s, e2 + e3 + e4) represents x - n * (c1 + c2) exactly.
     *
     * Second order.  e2, e3, e4 are O(2^-24 |s_k|) and p2l, p3h are
     * O(2^-48 |x|).  Their sum must not round at the scale of the
     * partial sums s_k: when x lies close to a multiple of pi/2 those
     * are far larger than r, which is how the previous accumulation in
     * fp32mp2_high lost up to 11 bits (37 bits at |x| ~ 1e4, 25 bits at
     * |x| ~ 4e5).  e2 + e3 + e4 goes through error-free sums; p2l + p3h
     * is added in float unless |r| < 2^-22 |x_hi| (one argument in ~4
     * million), where it is folded exactly as well.
     *
     * Third order (p3l, n * c4 and the sum errors of the second order)
     * is below 2^-70 |x| and is summed in float.  The returned pair is
     * not renormalized; the kernel tolerates that.
     */
    constexpr _FpType __c1 = _FpType(0x1.921fb40000000p+0);
    constexpr _FpType __c2 = _FpType(0x1.4442d00000000p-24);
    constexpr _FpType __c3 = _FpType(0x1.84698a0000000p-48);
    constexpr _FpType __c4 = _FpType(-0x1.cceba40000000p-74);

    const int __n       = ::cuda::__fpmp_fp2int_rn(__x_hi * _FpType(0x1.45f306p-1f));
    const _FpType __n_f = ::cuda::__fpmp_int2fp_rn<_FpType>(__n);

    _FpType __pl, __p2l, __p3l;
    const _FpType __ph  = ::cuda::__fpmp_two_mult_fma(__n_f, __c1, &__pl);
    const _FpType __p2h = ::cuda::__fpmp_two_mult_fma(__n_f, __c2, &__p2l);
    const _FpType __p3h = ::cuda::__fpmp_two_mult_fma(__n_f, __c3, &__p3l);

    _FpType __e2, __e3, __e4;
    _FpType __s = ::cuda::__fpmp_sub_rn(__x_hi, __ph); /* exact */
    __s         = ::cuda::__fpmp_two_sum(__s, -__pl, &__e2);
    __s         = ::cuda::__fpmp_two_sum(__s, __x_lo, &__e3);
    __s         = ::cuda::__fpmp_two_sum(__s, -__p2h, &__e4);

    _FpType __f1, __f2;
    _FpType __t     = ::cuda::__fpmp_two_sum(__e2, __e3, &__f1);
    __t             = ::cuda::__fpmp_two_sum(__t, __e4, &__f2);
    _FpType __third = (__f1 + __f2) - ::cuda::__fpmp_fma_rn(__n_f, __c4, __p3l);

    const _FpType __abs_s = (__s < _FpType(0)) ? -__s : __s;
    _FpType __r_hi_v, __r_lo_v;
    if (__abs_s >= __abs_hi * _FpType(0x1p-22f))
    {
      __third  = __third - (__p2l + __p3h);
      __r_hi_v = ::cuda::__fpmp_fast_two_sum(__s, __t, &__r_lo_v);
    }
    else
    {
      _FpType __f3, __f4;
      __t      = ::cuda::__fpmp_two_sum(__t, -__p2l, &__f3);
      __t      = ::cuda::__fpmp_two_sum(__t, -__p3h, &__f4);
      __third  = __third + (__f3 + __f4);
      __r_hi_v = ::cuda::__fpmp_two_sum(__s, __t, &__r_lo_v);
    }

    *__quadrant = __n;
    *__r_hi     = __r_hi_v;
    *__r_lo     = __r_lo_v + __third;
  }
  else
  {
    /* -- Slow path: |x_hi| >= 2^20 -- */

    /* Payne-Hanek: combine the 2/pi fractions of x_hi and x_lo in
     * 96-bit fixed point BEFORE the pi/2 multiply, to avoid precision
     * loss from floating-point cancellation.
     */
    uint32_t __f0, __f1, __f2;
    unsigned __q_hi;
    ::cuda::__internal_fpmp2_ph_frac(__x_hi, &__q_hi, &__f0, &__f1, &__f2);

    uint32_t __x_hi_sign = ::cuda::std::bit_cast<uint32_t>(__x_hi) & 0x80000000U;
    int __q              = (int) __q_hi;

    /* Add the x_lo contribution in fixed point.  |x_lo| <= |x_hi| * 2^-24
     * can still span many quadrants, and even a small |x_lo| can
     * dominate the fraction when the result angle is near zero, so
     * handle ALL non-zero x_lo.
     */
    if (__x_lo != _FpType(0))
    {
      _FpType __abs_lo       = (__x_lo < _FpType(0)) ? -__x_lo : __x_lo;
      uint32_t __abs_lo_bits = ::cuda::std::bit_cast<uint32_t>(__abs_lo);
      bool __same_sign       = (__x_lo > _FpType(0)) == (__x_hi > _FpType(0));

      uint32_t __g0 = 0U, __g1 = 0U, __g2 = 0U;
      unsigned __q_lo = 0;

      if (__abs_lo_bits >= 0x00800000U)
      {
        ::cuda::__internal_fpmp2_ph_frac(__abs_lo, &__q_lo, &__g0, &__g1, &__g2);
      }

      if (__same_sign)
      {
        __q += (int) __q_lo;
        const uint64_t __s2 = (uint64_t) __f2 + __g2;
        const uint64_t __s1 = (uint64_t) __f1 + __g1 + (__s2 >> 32);
        const uint64_t __s0 = (uint64_t) __f0 + __g0 + (__s1 >> 32);
        __f2                = (uint32_t) __s2;
        __f1                = (uint32_t) __s1;
        __f0                = (uint32_t) __s0;
        if ((__s0 >> 32) != 0U)
        {
          __q++;
        }
      }
      else
      {
        /* f - g modulo 2^96: when f < g the wrapped value is the correct
         * fraction of the next lower quadrant. */
        __q -= (int) __q_lo;
        const bool __ge     = (__f0 > __g0) || (__f0 == __g0 && (__f1 > __g1 || (__f1 == __g1 && __f2 >= __g2)));
        const uint64_t __d2 = (uint64_t) __f2 - __g2;
        const uint64_t __d1 = (uint64_t) __f1 - __g1 - (__d2 >> 63);
        const uint64_t __d0 = (uint64_t) __f0 - __g0 - (__d1 >> 63);
        __f2                = (uint32_t) __d2;
        __f1                = (uint32_t) __d1;
        __f0                = (uint32_t) __d0;
        if (!__ge)
        {
          __q--;
        }
      }
    }

    const uint32_t __top_bit = __f0 >> 31U;
    __q += (int) __top_bit;
    if (__x_hi_sign != 0U)
    {
      __q = 0U - (unsigned) __q;
    }

    if (__top_bit != 0U)
    {
      __f0 = ~__f0;
      __f1 = ~__f1;
      __f2 = ~__f2;
      __x_hi_sign ^= 0x80000000U;
    }

    if ((__f0 | __f1 | __f2) == 0U)
    {
      *__quadrant = (int) __q;
      *__r_hi     = _FpType(0);
      *__r_lo     = _FpType(0);
      return;
    }

    *__quadrant = (int) __q;
    ::cuda::__internal_fpmp2_frac_to_angle(__f0, __f1, __f2, __x_hi_sign, __r_hi, __r_lo);
  }
}

/*
 * sin and cos kernel for |r| <= pi/4 (plus the slack of a float-estimated
 * quadrant), both from one call since every caller needs both:
 *   sin(r) = r + r * (u * Qs(u)),  Qs(u) = -1/3! + u/5! - ... + u^6 (-1/15!)
 *   cos(r) = 1 + u * Qc(u),        Qc(u) = -1/2! + u/4! - ... + u^7 (1/16!)
 * with u = r^2 shared.  A coefficient can stay a plain float only when
 * one float ulp of it, times its power of r at |r| = pi/4, stays below
 * ~2^-50 of the result: that holds from 1/11! (sin) and 1/12! (cos) on
 * (1/10! as a float still cost cos ~1.5 u^2 at |r| = pi/4, the difference
 * between 45 and 46 bits on the test datasets).
 * 1/8! stored as a float cost cos 20 u^2 of coefficient rounding plus
 * ~50 u^2 of Horner rounding at |r| = pi/4 -- the 41-bit floor that sin,
 * cos, tan, sinpi and cospi all showed -- and 1/9! as a float cost sin
 * ~7 u^2.  The ff Horner steps use the Fast2Sum variant (|v*u| is at
 * most 3 % of the next coefficient), and the final additions to r and
 * to 1 do the same, which pays for the two extra ff terms.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincos_kernel(
  const float __r_hi, const float __r_lo, float* __s_hi, float* __s_lo, float* __c_hi, float* __c_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  constexpr __ffloat __sin_c[7] = {
    __ffloat(-1.6666666666666666e-01), /* [0] -1/3!  */
    __ffloat(8.3333333333333333e-03), /* [1]  1/5!  */
    __ffloat(-1.9841269841269841e-04), /* [2] -1/7!  */
    __ffloat(2.7557319223985893e-06), /* [3]  1/9!  (last float-float term) */
    /* high-order M = 3 entries: .lo() == 0 by construction */
    __ffloat(-2.5052108385441719e-08f), /* [4] -1/11! */
    __ffloat(1.6059043836821615e-10f), /* [5]  1/13! */
    __ffloat(-7.6471637318198165e-13f) /* [6] -1/15! */
  };
  constexpr __ffloat __cos_c[8] = {
    __ffloat(-5.0000000000000000e-01), /* [0] -1/2!  */
    __ffloat(4.1666666666666667e-02), /* [1]  1/4!  */
    __ffloat(-1.3888888888888889e-03), /* [2] -1/6!  */
    __ffloat(2.4801587301587302e-05), /* [3]  1/8!  */
    __ffloat(-2.7557319223985893e-07), /* [4] -1/10! (last float-float term) */
    /* high-order M = 3 entries: .lo() == 0 by construction */
    __ffloat(2.0876756987868099e-09f), /* [5]  1/12! */
    __ffloat(-1.1470745597729725e-11f), /* [6] -1/14! */
    __ffloat(4.7794773323873853e-14f) /* [7]  1/16! */
  };

  const __ffloat __r(__r_hi, __r_lo);
  const __ffloat __u = __r * __r;

  /* sin(r) = r + r * (u * Qs(u)); |r * u * Qs| <= 0.13 |r|, so Fast2Sum applies */
  const __ffloat __qs = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 3>(__u, __sin_c);
  const __ffloat __ps = (__qs * __u) * __r;
  float __sl;
  const float __sh = ::cuda::__fpmp_fast_two_sum(__r_hi, __ps.hi(), &__sl);
  __sl             = __sl + (__r_lo + __ps.lo());
  *__s_hi          = ::cuda::__fpmp_fast_two_sum(__sh, __sl, __s_lo);

  /* cos(r) = 1 + u * Qc(u); |u * Qc| <= 0.32 */
  const __ffloat __qc = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 3>(__u, __cos_c);
  const __ffloat __pc = __qc * __u;
  float __cl;
  const float __ch = ::cuda::__fpmp_fast_two_sum(1.0f, __pc.hi(), &__cl);
  __cl             = __cl + __pc.lo();
  *__c_hi          = ::cuda::__fpmp_fast_two_sum(__ch, __cl, __c_lo);
}

/*
 * ====================================================================
 * Internal kernels for asin, acos, atan and atan2 (fp32mp2)
 * ====================================================================
 *
 * All four functions are built on shared polynomial kernels evaluated in
 * fp32mp2 arithmetic.  The asin/acos kernels run in pure compensated
 * Horner (M = 0): on their ranges even the highest coefficients carry
 * float-rounding noise above the fp32mp2 ulp when stored as `float`.
 * The atan kernel's argument is folded once by its callers (see kernel 1),
 * which shrinks its polynomial to 13 coefficients with the highest in
 * plain float.
 *
 *   atan(x):   |x| <= sqrt(1/3)          -> kernel 1 on |x|
 *              sqrt(1/3) < |x| < sqrt(3) -> pi/4 + kernel 1 on (|x|-1)/(|x|+1)
 *              |x| >= sqrt(3)            -> pi/2 - kernel 1 on 1/|x|
 *              one refined division in every case, selected branch-free.
 *
 *   atan2(y,x):  one refined division of min/max, or of (|y|-|x|)/(|y|+|x|)
 *              when the ratio exceeds sqrt(1/3), then kernel 1 and the
 *              pi/4, pi/2 and pi anchors.
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

/* ---- (kernel 1) atan on |t| <= sqrt(1/3):  atan(t) = t + t * v * P(v), v = t^2 ----
 *
 * P is a 13-coefficient near-minimax fit of (atan(sqrt v)/sqrt v - 1)/v on
 * [0, 1/3] (Chebyshev nodes, error 2^-54.9); the highest coefficient is a
 * plain float.  The callers bring their argument into this range with one
 * fold and a single refined division:
 *     sqrt(1/3) < a < sqrt(3):  atan(a) = pi/4 + atan((a - 1) / (a + 1)),
 *                               |(a - 1)/(a + 1)| <= 2 - sqrt(3) = 0.268
 *     a >= sqrt(3):             atan(a) = pi/2 - atan(1 / a)
 * (atan2 folds the ratio |y|/|x| the same way, directly from y and x.)
 * The previous kernel covered |a| <= 1 with 19 coefficients in compensated
 * Horner.  Only the small correction w = atan(t) - t is returned; the
 * callers compose base + t + w with the helper below.
 * _FpType is unused -- the kernel is fp32mp2-only -- but keeping it a
 * template spares every translation unit that never calls it an
 * unused-function warning. */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_atan_kernel(const fp32mp2_low& __t, fp32mp2_low* __result) noexcept
{
  using __ffloat = fp32mp2_low;

  constexpr __ffloat __atan_c[13] = {
    __ffloat(-3.33333333333333204e-01), /* c0  */
    __ffloat(+1.99999999999893707e-01), /* c1  */
    __ffloat(-1.42857142839203616e-01), /* c2  */
    __ffloat(+1.11111109916979270e-01), /* c3  */
    __ffloat(-9.09090494297958851e-02), /* c4  */
    __ffloat(+7.69222148497070296e-02), /* c5  */
    __ffloat(-6.66550850221702129e-02), /* c6  */
    __ffloat(+5.87182313264359781e-02), /* c7  */
    __ffloat(-5.19665181466080761e-02), /* c8  */
    __ffloat(+4.46653875216280818e-02), /* c9  */
    __ffloat(-3.42420037510217240e-02), /* c10 */
    __ffloat(+1.97768457382240768e-02), /* c11 */
    /* high-order M = 1 entry: .lo() == 0 by construction */
    __ffloat(-6.07076482944944856e-03f), /* c12 */
  };

  const __ffloat __v = __t * __t;
  const __ffloat __p = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 1>(__v, __atan_c);
  *__result          = __t * (__v * __p); /* w = atan(t) - t, at most 0.1 |t| */
}

/* ---- composition  r = base + s * (t + w)  for atan, atan2 and asin ----
 * base is a float pair (0, pi/4, pi/2, 3pi/4 or pi), t a possibly
 * unnormalized pair with |t| < base unless base = 0, and w a pair with
 * |w| < |t| (atan: the reduced argument and the kernel's correction
 * atan(t) - t; asin: 2 sqrt(y) and 2 sqrt(y) y P(y)); s = -1 negates both.  Every high-word sum is
 * exact (Fast2Sum applies: |t| < base unless base = 0, |w| < |t|) and the
 * low words are accumulated with error-free sums and folded back into the
 * high word, so the result rounds once at half an ulp of its own low word
 * (~0.5 u^2); summing the low words in plain float cost up to 4 u^2 just
 * above the fold point, where the result is only twice the reduced
 * argument. */
template <typename _FpType>
_CCCL_FPMP_CORE_API fp32mp2_low __internal_fpmp2_compose_base_tw(
  const float __b_hi, const float __b_lo, const fp32mp2_low& __t, const fp32mp2_low& __w, const bool __negate) noexcept
{
  using __ffloat   = fp32mp2_low;
  const float __th = __negate ? -__t.hi() : __t.hi();
  const float __tl = __negate ? -__t.lo() : __t.lo();
  const float __wh = __negate ? -__w.hi() : __w.hi();
  const float __wl = __negate ? -__w.lo() : __w.lo();
  float __e1, __e2, __e3, __f1, __f2, __f3;
  const float __s1 = ::cuda::__fpmp_fast_two_sum(__b_hi, __th, &__e1);
  const float __s2 = ::cuda::__fpmp_fast_two_sum(__s1, __wh, &__e2);
  /* exact sum of the three half-ulp-sized low words, folded back into the
   * high word before anything rounds: left as a float it reached three
   * half-ulps and its single rounding cost up to 3 u^2 */
  float __p        = ::cuda::__fpmp_two_sum(__e1, __tl, &__f1);
  __p              = ::cuda::__fpmp_two_sum(__p, __e2, &__f2);
  __p              = ::cuda::__fpmp_two_sum(__p, __b_lo, &__f3);
  const float __s3 = ::cuda::__fpmp_fast_two_sum(__s2, __p, &__e3);
  const float __lo = __e3 + (__wl + ((__f1 + __f2) + __f3));
  float __r_lo;
  const float __r_hi = ::cuda::__fpmp_fast_two_sum(__s3, __lo, &__r_lo);
  return __ffloat(__r_hi, __r_lo);
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

  __ffloat __q = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__y_fast, __asin_c);
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

  __ffloat __q = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__y_fast, __acos_c);
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
  ::cuda::__internal_fpmp2_trig_reduction(__x_hi, __x_lo, &__quadrant, &__r_hi, &__r_lo);

  float __s_hi, __s_lo, __c_hi, __c_lo;
  ::cuda::__internal_fpmp2_sincos_kernel(__r_hi, __r_lo, &__s_hi, &__s_lo, &__c_hi, &__c_lo);

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
  ::cuda::__fpmp2_sincos(__x_hi, __x_lo, __res_hi, __res_lo, &__c_hi, &__c_lo);
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
  ::cuda::__fpmp2_sincos(__x_hi, __x_lo, &__s_hi, &__s_lo, __res_hi, __res_lo);
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
 *      shared __internal_fpmp2_sincos_kernel.
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
  ::cuda::__internal_fpmp2_trig_reduction(__x_hi, __x_lo, &__quadrant, &__r_hi, &__r_lo);

  float __s_hi, __s_lo, __c_hi, __c_lo;
  ::cuda::__internal_fpmp2_sincos_kernel(__r_hi, __r_lo, &__s_hi, &__s_lo, &__c_hi, &__c_lo);

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

  /* Three ranges, one refined division (exact-remainder variant) in all of
   * them so the warp never diverges; see kernel 1 for the identities.
   * |x| - 1 is exact for |x| in [1/2, 2] (Sterbenz), |x| + 1 is an exact
   * pair.  The reciprocal's denominator is capped at 2^64: above it the
   * correction atan(1/|x|) is below 2^-64 of the result, and the cap keeps
   * |x| = inf out of the division (which used to return NaN there). */
  constexpr float __sqrt13(0x1.279a74p-1f); /* sqrt(1/3) */
  constexpr float __sqrt3(0x1.bb67aep+0f);
  constexpr float __pio4_hi(0x1.921fb6p-1f);
  constexpr float __pio4_lo(-0x1.777a5cp-26f);
  const float __ah  = __absx.hi();
  const float __al  = __absx.lo();
  const bool __fold = (__ah > __sqrt13) && (__ah < __sqrt3);
  /* Written so that NaN selects the direct range (every comparison false) and flows through to
   * the result; the negated forms sent it to the reciprocal range, whose cap made it pi/2. */
  const bool __inv  = __ah >= __sqrt3;
  const bool __cap  = __ah >= 0x1p64f;

  float __nl, __de;
  const float __nh = ::cuda::__fpmp_fast_two_sum(::cuda::__fpmp_sub_rn(__ah, 1.0f), __al, &__nl);
  const float __dh = ::cuda::__fpmp_two_sum(__ah, 1.0f, &__de);
  const float __dl = __de + __al;

  const float __num_hi = __inv ? 1.0f : (__fold ? __nh : __ah);
  const float __num_lo = __inv ? 0.0f : (__fold ? __nl : __al);
  const float __den_hi = __inv ? (__cap ? 0x1p64f : __ah) : (__fold ? __dh : 1.0f);
  const float __den_lo = __inv ? (__cap ? 0.0f : __al) : (__fold ? __dl : 0.0f);
  float __q_hi, __q_lo;
  ::cuda::__fpmp_math_div_refined<true>(__num_hi, __num_lo, __den_hi, __den_lo, &__q_hi, &__q_lo);
  const __ffloat __t = (__fold || __inv) ? __ffloat(__q_hi, __q_lo) : __absx;

  __ffloat __w;
  ::cuda::__internal_fpmp2_atan_kernel<float>(__t, &__w);

  /* r = base + (t + w), or pi/2 - (t + w) for the reciprocal range */
  const float __b_hi = __inv ? __pio4_hi * 2.0f : (__fold ? __pio4_hi : 0.0f);
  const float __b_lo = __inv ? __pio4_lo * 2.0f : (__fold ? __pio4_lo : 0.0f);
  __ffloat __r       = ::cuda::__internal_fpmp2_compose_base_tw<float>(__b_hi, __b_lo, __t, __w, __inv);
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
    /* Generic finite path.  Bring the larger magnitude into [2^-100, 2^100]
     * first (rare, exact scaling): it keeps |y| + |x| finite and the
     * approximate reciprocal in range (atan2 used to return 0 for
     * |x| > 2^126).  Then one refined division of either min/max or, when
     * the ratio exceeds sqrt(1/3), of (|y| - |x|) / (|y| + |x|), whose
     * magnitude is at most 2 - sqrt(3):
     *     atan2(|y|, |x|) = pi/4 + atan((|y| - |x|) / (|y| + |x|))
     * holds on both sides of the diagonal.  Kernel 1 then sees |t| <=
     * sqrt(1/3). */
    constexpr float __sqrt13(0x1.279a74p-1f);
    constexpr float __pio4_hi(0x1.921fb6p-1f);
    constexpr float __pio4_lo(-0x1.777a5cp-26f);
    const float __big0 = (__ay.hi() > __ax.hi()) ? __ay.hi() : __ax.hi();
    if (__big0 > 0x1p100f || __big0 < 0x1p-100f)
    {
      const float __sc = (__big0 > 1.0f) ? 0x1p-64f : 0x1p64f;
      __ay             = __ffloat(__ay.hi() * __sc, __ay.lo() * __sc);
      __ax             = __ffloat(__ax.hi() * __sc, __ax.lo() * __sc);
    }
    const bool __y_gt_x  = __ay.hi() > __ax.hi();
    const __ffloat __big = __y_gt_x ? __ay : __ax;
    const __ffloat __sml = __y_gt_x ? __ax : __ay;
    const bool __fold    = __sml.hi() > __big.hi() * __sqrt13;
    float __e1, __e2;
    const float __dh = ::cuda::__fpmp_two_sum(__ay.hi(), -__ax.hi(), &__e1);
    const float __dl = __e1 + (__ay.lo() - __ax.lo());
    const float __sh = ::cuda::__fpmp_two_sum(__ay.hi(), __ax.hi(), &__e2);
    const float __sl = __e2 + (__ay.lo() + __ax.lo());
    float __t_hi, __t_lo;
    ::cuda::__fpmp_math_div_refined<true>(
      __fold ? __dh : __sml.hi(),
      __fold ? __dl : __sml.lo(),
      __fold ? __sh : __big.hi(),
      __fold ? __sl : __big.lo(),
      &__t_hi,
      &__t_lo);
    const __ffloat __t(__t_hi, __t_lo);
    __ffloat __w;
    ::cuda::__internal_fpmp2_atan_kernel<float>(__t, &__w);

    /* First octant: atan2(|y|, |x|) = pi/4 + (t + w) folded, pi/2 - (t + w)
     * for |y| > |x|, else t + w; x < 0 mirrors it to pi - that, i.e. the
     * base becomes 3pi/4, pi/2 or pi and the sign of (t + w) flips. */
    constexpr float __pi3o4_hi(0x1.2d97c80000000p+1f);
    constexpr float __pi3o4_lo(-0x1.99bc5c0000000p-28f);
    const bool __neg_w = (!__fold && __y_gt_x) != __x_is_neg;
    float __b_hi, __b_lo;
    if (__fold)
    {
      __b_hi = __x_is_neg ? __pi3o4_hi : __pio4_hi;
      __b_lo = __x_is_neg ? __pi3o4_lo : __pio4_lo;
    }
    else if (__y_gt_x)
    {
      __b_hi = __pio4_hi * 2.0f;
      __b_lo = __pio4_lo * 2.0f;
    }
    else
    {
      __b_hi = __x_is_neg ? __pio4_hi * 4.0f : 0.0f;
      __b_lo = __x_is_neg ? __pio4_lo * 4.0f : 0.0f;
    }
    __r = ::cuda::__internal_fpmp2_compose_base_tw<float>(__b_hi, __b_lo, __t, __w, __neg_w);
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
  __fpmp_fp128 __res =
    _CCCL_FPMP_ATAN2Q(::cuda::__fpmp2_to_quad(__y_hi, __y_lo), ::cuda::__fpmp2_to_quad(__x_hi, __x_lo));
  ::cuda::__fpmp2_from_quad(__res, __res_hi, __res_lo);
#  else
  double __res =
    ::cuda::std::atan2(::cuda::__fpmp2_to_double(__y_hi, __y_lo), ::cuda::__fpmp2_to_double(__x_hi, __x_lo));
  ::cuda::__fpmp2_from_double(__res, __res_hi, __res_lo);
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
    ::cuda::__internal_fpmp2_asin_poly<float>(__a2, &__p);
    __r = renormalize(__absx + __absx * (__a2 * __p));
  }
  else
  {
    /* Large branch: y = (1 - |x|)/2,
     *   asin(|x|) = pi/2 - 2*sqrty*(1 + y*P(y))
     * sqrt(y) returns NaN for y < 0 (i.e., |x| > 1), so NaN
     * propagates through the rest of the chain naturally. */
    __ffloat __y = 0.5f - __absx * 0.5f;
    float __sy_hi, __sy_lo;
    ::cuda::__fpmp2_sqrt(__y.hi(), __y.lo(), &__sy_hi, &__sy_lo);
    __ffloat __sy(__sy_hi, __sy_lo);

    __ffloat __p;
    ::cuda::__internal_fpmp2_asin_poly<float>(__y, &__p);

    /* 2 sqrt(y) (1 + y P) = 2 sqrt(y) + 2 sqrt(y) (y P): the second term is
     * at most 8 % of the first, so its own roundings are immaterial, and
     * pi/2 - (first + second) is composed with exact low words (the
     * cancellation just above the branch point amplifies every rounding
     * here by 1.5x; the products as float-float operations and plain
     * low-word sums cost ~1 u^2 each, i.e. 44-45 bits). */
    const __ffloat __c = __sy * (__y * __p);
    constexpr float __pio2_hi(0x1.921fb6p+0f);
    constexpr float __pio2_lo(-0x1.777a5cp-25f);
    const fp32mp2_low __r_low = ::cuda::__internal_fpmp2_compose_base_tw<float>(
      __pio2_hi,
      __pio2_lo,
      fp32mp2_low(__sy_hi + __sy_hi, __sy_lo + __sy_lo),
      fp32mp2_low(__c.hi() + __c.hi(), __c.lo() + __c.lo()),
      true);
    __r = __ffloat(__r_low.hi(), __r_low.lo());
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
    ::cuda::__internal_fpmp2_asin_poly<float>(__a2, &__p);
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
    __ffloat __y     = 1.0f - __absx;
    __ffloat __two_y = __y * 2.0f;
    float __s_hi, __s_lo;
    ::cuda::__fpmp2_sqrt(__two_y.hi(), __two_y.lo(), &__s_hi, &__s_lo);
    __ffloat __s(__s_hi, __s_lo);

    __ffloat __p;
    ::cuda::__internal_fpmp2_acos_poly<float>(__y, &__p);
    /* s (1 + y P) = s + s (y P), the second term at most 4 % of the first: one Fast2Sum. */
    const __ffloat __c = __s * (__y * __p);
    float __al;
    const float __ah    = ::cuda::__fpmp_fast_two_sum(__s_hi, __c.hi(), &__al);
    __ffloat __acos_abs = renormalize(__ffloat(__ah, __al + (__s_lo + __c.lo())));
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
 * sinpi(x), cospi(x), sincospi(x) - sin(pi*x) and cos(pi*x)
 * ====================================================================
 * fp32mp2 reduces x = n + f, |f| <= 1/2, by splitting the limb pair.
 * f is folded into [-1/4, 1/4], scaled by pi, and evaluated with the
 * sincos Taylor kernels. No radian argument reduction is involved.
 * fp64mp2 remains a double-precision fallback.
 */

_CCCL_FPMP_CORE_API void __internal_fpmp2_sincospi(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept;

/*
 * --------------------------------------------------------------------
 * Sine of pi*x sinpi(x) (fp32mp2)
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sinpi(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  float __c_hi, __c_lo;
  ::cuda::__internal_fpmp2_sincospi(__x_hi, __x_lo, __res_hi, __res_lo, &__c_hi, &__c_lo);
}

/*
 * --------------------------------------------------------------------
 * Sine of pi*x sinpi(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sinpi(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  double __xd = ::cuda::__fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::sinpi(__xd), __res_hi, __res_lo);),
    (::cuda::__fpmp2_from_double(::cuda::std::sin(__xd * ::cuda::std::__numbers<double>::__pi()), __res_hi, __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_1A(sinpi)

/*
 * ====================================================================
 * cospi(x) - cos(pi * x)
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Cosine of pi*x cospi(x) (fp32mp2)
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cospi(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  float __s_hi, __s_lo;
  ::cuda::__internal_fpmp2_sincospi(__x_hi, __x_lo, &__s_hi, &__s_lo, __res_hi, __res_lo);
}

/*
 * --------------------------------------------------------------------
 * Cosine of pi*x cospi(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cospi(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  double __xd = ::cuda::__fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::cospi(__xd), __res_hi, __res_lo);),
    (::cuda::__fpmp2_from_double(::cuda::std::cos(__xd * ::cuda::std::__numbers<double>::__pi()), __res_hi, __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_1A(cospi)

/*
 * ====================================================================
 * sincospi(x, &s, &c) - sin(pi * x) and cos(pi * x)
 * ====================================================================
 */

/*
 * pi * f as a float-float, for the fraction f of sinpi/cospi (|f| <= 1/4,
 * or any f for the tiny-argument path).  pi = pi1 + pi2 + pi3 + pi4 is the
 * doubled 4-piece pi/2 of the Cody-Waite reduction (doubling is exact).
 * The three first-order products are split exactly; the two smaller ones
 * are folded into the leading one with Fast2Sum (each is below 2^-24 of
 * it), the second-order parts are summed with error-free sums and the
 * third-order parts in float, so the angle is accurate to ~1 u^2.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_pi_times(const float __f_hi, const float __f_lo, float* __a_hi, float* __a_lo) noexcept
{
  constexpr float __pi1 = 2.0f * 0x1.921fb40000000p+0f;
  constexpr float __pi2 = 2.0f * 0x1.4442d00000000p-24f;
  constexpr float __pi3 = 2.0f * 0x1.84698a0000000p-48f;

  float __pl, __ql, __ul;
  const float __ph = ::cuda::__fpmp_two_mult_fma(__f_hi, __pi1, &__pl);
  const float __qh = ::cuda::__fpmp_two_mult_fma(__f_lo, __pi1, &__ql);
  const float __uh = ::cuda::__fpmp_two_mult_fma(__f_hi, __pi2, &__ul);

  float __a1, __a2, __g1, __g2;
  float __a = ::cuda::__fpmp_fast_two_sum(__ph, __qh, &__a1);
  __a       = ::cuda::__fpmp_fast_two_sum(__a, __uh, &__a2);
  float __t = ::cuda::__fpmp_two_sum(__a1, __a2, &__g1);
  __t       = ::cuda::__fpmp_two_sum(__t, __pl, &__g2);
  const float __third =
    ((__g1 + __g2) + (__ql + __ul)) + ::cuda::__fpmp_fma_rn(__f_lo, __pi2, ::cuda::__fpmp_mul_rn(__f_hi, __pi3));

  *__a_hi = ::cuda::__fpmp_fast_two_sum(__a, __t + __third, __a_lo);
}

/*
 * Split a float into a nearby integer and an exact residual in [-1/2, 1/2].
 * The integer is returned mod 2; only its parity changes sin(pi*x) and cos(pi*x).
 */
[[nodiscard]] _CCCL_FPMP_CORE_API int __internal_fpmp2_pi_limb(float __x, float* __f) noexcept
{
  // |x| >= 2^23: ulp >= 1, so a finite float is an integer and the fraction is 0.
  // |x| >= 2^24: ulp >= 2, so it is an even integer and its parity is 0.
  constexpr float __integral_threshold      = 0x1p23f;
  constexpr float __even_integral_threshold = 0x1p24f;

  const float __abs = ::cuda::std::fabs(__x);
  if (__abs >= __even_integral_threshold)
  {
    *__f = 0.0f;
    return 0;
  }
  if (__abs >= __integral_threshold)
  {
    const int __n = ::cuda::__fpmp_fp2int_rn(__x);
    *__f          = 0.0f;
    return __n & 1;
  }

  int __n          = ::cuda::__fpmp_fp2int_rn(__x);
  const float __nf = ::cuda::__fpmp_int2fp_rn<float>(__n);
  float __frac     = __x - __nf;
  if (__frac > 0.5f)
  {
    ++__n;
    __frac -= 1.0f;
  }
  else if (__frac < -0.5f)
  {
    --__n;
    __frac += 1.0f;
  }
  *__f = __frac;
  return __n & 1;
}

/*
 * --------------------------------------------------------------------
 * Sine and cosine of pi*x sincospi(x, &s, &c) (fp32mp2)
 * --------------------------------------------------------------------
 * x = n + f, |f| <= 1/2. Fold f into [-1/4, 1/4] so pi*f is inside the
 * Taylor kernels, then apply the cofunction identity and the sign (-1)^n.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincospi(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept
{
  const float __abs_hi = ::cuda::std::fabs(__x_hi);
  // Inf and NaN share an all-ones exponent. isfinite(-0) is true, so this does not
  // treat a signed zero as non-finite the way a raw sign-bit check would.
  if (!::cuda::std::isfinite(__x_hi) || !::cuda::std::isfinite(__x_lo))
  {
    *__sin_hi = ::cuda::std::numeric_limits<float>::quiet_NaN();
    *__sin_lo = 0.0f;
    *__cos_hi = ::cuda::std::numeric_limits<float>::quiet_NaN();
    *__cos_lo = 0.0f;
    return;
  }

  /* |x| < 2^-64: sinpi(x) = pi * x to fp32mp2 precision (the cubic term is
   * below 2^-128 of it) and cospi(x) = 1.  The products are formed on x
   * scaled by 2^64 so that their low parts stay normal even for a
   * denormal x -- unscaled they flush and the result degrades to float
   * precision (23 bits on the pattern dataset) -- then scaled back in two
   * steps so a denormal result is rounded once. */
  if (__abs_hi < 0x1p-64f)
  {
    float __a_hi, __a_lo;
    ::cuda::__internal_fpmp2_pi_times(__x_hi * 0x1p64f, __x_lo * 0x1p64f, &__a_hi, &__a_lo);
    *__sin_hi = (__a_hi * 0x1p-32f) * 0x1p-32f;
    *__sin_lo = (__a_lo * 0x1p-32f) * 0x1p-32f;
    *__cos_hi = 1.0f;
    *__cos_lo = 0.0f;
    return;
  }

  int __odd;
  float __f_hi;
  float __f_lo;
  // Same thresholds as __internal_fpmp2_pi_limb: below 2^23 the high limb still has a fraction.
  constexpr float __integral_threshold      = 0x1p23f;
  constexpr float __even_integral_threshold = 0x1p24f;
  if (__abs_hi < __integral_threshold)
  {
    int __n           = ::cuda::__fpmp_fp2int_rn(__x_hi);
    const float __n_f = ::cuda::__fpmp_int2fp_rn<float>(__n);
    const float __e   = __x_hi - __n_f;
    float __fl        = 0.0f;
    float __fh        = ::cuda::__fpmp_two_sum(__e, __x_lo, &__fl);
    if (__fh > 0.5f || (__fh == 0.5f && __fl > 0.0f))
    {
      ++__n;
      float __err = 0.0f;
      __fh        = ::cuda::__fpmp_two_sum(__fh, -1.0f, &__err);
      __fl += __err;
    }
    else if (__fh < -0.5f || (__fh == -0.5f && __fl < 0.0f))
    {
      --__n;
      float __err = 0.0f;
      __fh        = ::cuda::__fpmp_two_sum(__fh, 1.0f, &__err);
      __fl += __err;
    }
    __f_hi = ::cuda::__fpmp_two_sum(__fh, __fl, &__f_lo);
    __odd  = __n & 1;
  }
  else
  {
    const int __odd_hi = (__abs_hi < __even_integral_threshold) ? (::cuda::__fpmp_fp2int_rn(__x_hi) & 1) : 0;
    float __lo_f       = 0.0f;
    const int __odd_lo = ::cuda::__internal_fpmp2_pi_limb(__x_lo, &__lo_f);
    __odd              = __odd_hi ^ __odd_lo;
    __f_hi             = __lo_f;
    __f_lo             = 0.0f;
  }

  // Captured before the fold. CUDA's contract is sinpi(-0) == -0. two_sum of that
  // fraction yields +0, and the sine kernel keeps the sign it is given, so the
  // input sign has to be put back when the fraction is exactly zero. The odd
  // negation stays on the cosine and is not applied to that sine zero.
  const bool __exact_integer = (__f_hi == 0.0f) && (__f_lo == 0.0f);

  /* Fold into [-1/4, 1/4]. q == 1 uses sin=cos, cos=sin; q == 3 uses sin=-cos, cos=sin. */
  int __q = 0;
  if (__f_hi > 0.25f || (__f_hi == 0.25f && __f_lo > 0.0f))
  {
    __q         = 1;
    float __err = 0.0f;
    __f_hi      = ::cuda::__fpmp_two_sum(0.5f, -__f_hi, &__err);
    __f_lo      = __err - __f_lo;
  }
  else if (__f_hi < -0.25f || (__f_hi == -0.25f && __f_lo < 0.0f))
  {
    __q         = 3;
    float __err = 0.0f;
    __f_hi      = ::cuda::__fpmp_two_sum(__f_hi, 0.5f, &__err);
    __f_lo += __err;
  }
  {
    const float __renorm = ::cuda::__fpmp_two_sum(__f_hi, __f_lo, &__f_lo);
    __f_hi               = __renorm;
  }

  /* angle = pi * f, see __internal_fpmp2_pi_times */
  float __angle_hi, __angle_lo;
  ::cuda::__internal_fpmp2_pi_times(__f_hi, __f_lo, &__angle_hi, &__angle_lo);

  float __s_hi, __s_lo, __c_hi, __c_lo;
  ::cuda::__internal_fpmp2_sincos_kernel(__angle_hi, __angle_lo, &__s_hi, &__s_lo, &__c_hi, &__c_lo);

  if (__q == 1)
  {
    const float __t_hi = __s_hi;
    const float __t_lo = __s_lo;
    __s_hi             = __c_hi;
    __s_lo             = __c_lo;
    __c_hi             = __t_hi;
    __c_lo             = __t_lo;
  }
  else if (__q == 3)
  {
    const float __t_hi = __s_hi;
    const float __t_lo = __s_lo;
    __s_hi             = -__c_hi;
    __s_lo             = -__c_lo;
    __c_hi             = __t_hi;
    __c_lo             = __t_lo;
  }
  if (__odd != 0)
  {
    if (!__exact_integer)
    {
      __s_hi = -__s_hi;
      __s_lo = -__s_lo;
    }
    __c_hi = -__c_hi;
    __c_lo = -__c_lo;
  }
  if (__exact_integer)
  {
    __s_hi = ::cuda::std::copysign(__s_hi, __x_hi);
    __s_lo = ::cuda::std::copysign(__s_lo, __x_hi);
  }

  *__sin_hi = __s_hi;
  *__sin_lo = __s_lo;
  *__cos_hi = __c_hi;
  *__cos_lo = __c_lo;
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
  double __xd = ::cuda::__fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    ({
      double __sd;
      double __cd;
      ::sincospi(__xd, &__sd, &__cd);
      ::cuda::__fpmp2_from_double(__sd, __sin_hi, __sin_lo);
      ::cuda::__fpmp2_from_double(__cd, __cos_hi, __cos_lo);
    }),
    ({
      double __xpi = __xd * ::cuda::std::__numbers<double>::__pi();
      ::cuda::__fpmp2_from_double(::cuda::std::sin(__xpi), __sin_hi, __sin_lo);
      ::cuda::__fpmp2_from_double(::cuda::std::cos(__xpi), __cos_hi, __cos_lo);
    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A_2OUT(sincospi)

#endif // _CCCL_FPMP_USE_LIB
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_TRIG_H
