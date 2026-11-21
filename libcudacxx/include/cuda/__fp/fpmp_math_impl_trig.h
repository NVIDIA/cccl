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

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
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
#include <cuda/std/__cmath/signbit.h>
#include <cuda/std/__floating_point/constants.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/numbers>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
#ifndef _CCCL_FPMP_USE_LIB

/*
 * ====================================================================
 * Internal kernels for sin, cos and sincos (fp32mp2)
 * ====================================================================
 * Two sets of kernels: fpmp2_accuracy::low and mid (the default) take the
 * shorter one described first; fpmp2_accuracy::high takes the Taylor kernel
 * and the fast-path reduction it was measured with (the __internal_fpmp2_high_
 * entry points), which err by about 1u^2 less at the worst argument, in more
 * operations. The tiny and Payne-Hanek paths of the reduction, the quadrant
 * mapping and the special values are the same at every level.
 *
 * Algorithm (low and mid):
 *   1. Argument reduction: x = n*(pi/2) + r
 *      - Tiny (|x| < pi/4): no reduction
 *      - Fast (|x| < 2^20): Cody-Waite with a 4-piece pi/2 (94 bits, within
 *        8.5e-32). n*C1 is subtracted by one exact fma, n*C2 and n*C3 as exact
 *        products, and the third-order terms are added as an exact triple, so r
 *        keeps ~46 bits even when x lies within 2^-47 |x| of a multiple of pi/2.
 *        n is x_hi * fl(2/pi) rounded to an integer, so |r| <= 0.908
 *      - Large (|x| >= 2^20): Payne-Hanek using an integer 2/pi table (256
 *        bits), combining x_hi and x_lo fractions in 96-bit fixed point
 *        before converting to fp32mp2 (pure fp32 arithmetic, no fp64),
 *        |r| <= pi/4. Delivers ~46 bits in the reduced argument r down to
 *        |r| ~ 2^-46; final precision can be lower for tan near singularities,
 *        where small input quantization is amplified by tan'.
 *   2. sin(r) = r (1 + r^2 S(r^2)) and cos(r) = 1 + r^2 C(r^2), with S and C
 *      minimax polynomials of degree 6 on |r| <= 0.91: three coefficients
 *      as pairs, four in single precision
 *   3. Map to correct quadrant using n mod 4. sincos and tan evaluate both
 *      kernels; sin and cos evaluate one, its coefficients selected by the
 *      parity of n
 *
 * fpmp2_accuracy::high differs in:
 *   1. Fast (|x| < 2^20): n*C1 as an exact product subtracted by a TwoSum,
 *      and the remaining terms accumulated in fp32mp2_high
 *   2. sin(r) and cos(r) via Taylor polynomials in fp32mp2, sin: 8 terms
 *      (x through x^15), cos: 9 terms (1 through x^16), one shared kernel
 *      (shared r^2, Fast2Sum Horner steps)
 *   3. sincos computes both kernels; sin and cos call sincos
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
 * Returns quadrant (mod 4) and reduced argument r, |r| <= 0.908.
 *
 * Three paths:
 *   Tiny:  |x| < pi/4 -> no reduction
 *   Fast:  |x| < 2^20 -> Cody-Waite with a 4-piece pi/2, |r| <= 0.908
 *   Large: |x| >= 2^20 -> Payne-Hanek (integer 2/pi table), |r| <= pi/4
 *
 * `__high` (a constant at every call) selects how the fast path accumulates
 * its terms: the form fpmp2_accuracy::high was measured with, or the shorter
 * one of the other levels. Both use the same n and the same pieces of pi/2.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_trig_reduction(
  _FpType __x_hi, _FpType __x_lo, int* __quadrant, _FpType* __r_hi, _FpType* __r_lo, const bool __high) noexcept
{
  using __afloat = fp32mp2_high;

  // The sign bit is cleared, not negated away: -0 is below pi/4 too, and comes back as -0.
  uint32_t __abs_bits = ::cuda::std::bit_cast<uint32_t>(__x_hi) & 0x7FFFFFFFU;

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
     * pi/2 split into 4 float pieces, whose sum misses pi/2 by 8.5e-32.
     *
     * n is x_hi * fl(2/pi) rounded to an integer. Over every float x_hi in
     * [pi/4, 2^20) that leaves |x_hi - n*pi/2| <= 0.8767, and |x_lo| <= 2^-5, so
     * |r| <= 0.908: the kernels are fitted on |r| <= 0.91.
     */
    constexpr _FpType __c1 = _FpType(1.5707962512969971e+000);
    constexpr _FpType __c2 = _FpType(7.5497894158615964e-008);
    constexpr _FpType __c3 = _FpType(5.3903029534742384e-015);
    constexpr _FpType __c4 = _FpType(-0x1.cceba4p-74);

    int __n       = ::cuda::__fpmp_fp2int_rn(__x_hi * _FpType(0x1.45f306p-1f));
    _FpType __n_f = ::cuda::__fpmp_int2fp_rn<_FpType>(__n);

    *__quadrant = __n;
    if (__high)
    {
      /* fpmp2_accuracy::high: the accumulation its kernel was measured with.
       * C1 has 2 trailing zero mantissa bits, making n*C1 exact for |n| < 2^12
       * via two_mult_fma, and accurate for larger n; two_sum gives the exact
       * x_hi - ph = s + e, and the remaining corrections are accumulated in
       * fp32mp2_high.
       */
      /* Exact product n*C1 = ph + pl */
      _FpType __pl;
      _FpType __ph = ::cuda::__fpmp_two_mult_fma(__n_f, __c1, &__pl);

      /* Exact subtraction x_hi - ph = s + e */
      _FpType __e;
      _FpType __s = ::cuda::__fpmp_two_sum(__x_hi, -__ph, &__e);

      /* Build result as fp32mp2_high from exact (s, e),
       * then accumulate corrections with full precision.
       */
      __afloat __result(__s, __e);
      /* Subtracting pl and adding x_lo as single limbs gives the same values as adding them as
       * pairs with a zero low word, in fewer operations, and makes sin, cos, sincos and tan 4-10%
       * faster on sm_100 and sm_120. On sm_90, nvcc schedules the single-limb form worse: sin,
       * cos and sincos ran 1-4% slower than with the pairs (tan 4% faster), so sm_90 keeps them.
       */
      NV_IF_ELSE_TARGET(NV_IS_EXACTLY_SM_90,
                        (__result = __result + __afloat(-__pl); __result = __result + __afloat(__x_lo);),
                        (__result = __result - __pl; __result = __result + __x_lo;))

      /* Exact product n*C2 = nC2_hi + nC2_lo via two_mult_fma */
      _FpType __n_c2_lo;
      _FpType __n_c2_hi = ::cuda::__fpmp_two_mult_fma(__n_f, __c2, &__n_c2_lo);
      __result          = __result - __afloat(__n_c2_hi, __n_c2_lo);

      /* nC3_hi + nC3_lo = n*(C3 + C4) to within 2^-47 * n*C3: n*C3 exactly, and n*C4 added
       * to its low word by one fma. Near a multiple of pi/2, r is far smaller than n*C3, so
       * neither rounding n*C3 (error up to 2^-24 * n*C3) nor cutting pi/2 at C3
       * (error n * 9.5e-23) keeps the error small relative to r.
       */
      _FpType __n_c3_lo;
      _FpType __n_c3_hi = ::cuda::__fpmp_two_mult_fma(__n_f, __c3, &__n_c3_lo);
      __n_c3_lo         = ::cuda::__fpmp_fma_rn(__n_f, __c4, __n_c3_lo);
      __result          = __result - __afloat(__n_c3_hi, __n_c3_lo);

      *__r_hi = __result.hi();
      *__r_lo = __result.lo();
    }
    else
    {
      /* s = x_hi - n*C1, exact in one fma: |s| < 1 (n*|pi/2 - C1| < 0.06), C1 is a
       * multiple of 2^-22, and so is x_hi from 2 on. Below 2, n <= 1 and x_hi is a
       * multiple of 2^-23, of 2^-24 below 1, where |s| > 1/2. Checked for every
       * float x_hi in [pi/4, 2^20).
       */
      const _FpType __s = ::cuda::__fpmp_fma_rn(__n_f, -__c1, __x_hi);

      /* n*C2 = h2 + l2 exactly. h3 + l3 = n*(C3 + C4) to within 2^-49 * n*C3: n*C3
       * exactly, and n*C4 added to its low word by one fma.
       */
      _FpType __l2;
      const _FpType __h2 = ::cuda::__fpmp_two_mult_fma(__n_f, __c2, &__l2);
      _FpType __l3;
      const _FpType __h3 = ::cuda::__fpmp_two_mult_fma(__n_f, __c3, &__l3);
      __l3               = ::cuda::__fpmp_fma_rn(__n_f, __c4, __l3);

      /* x_lo - h2 exactly, formed off the path of s. Adding s is DWPlusFP: within
       * 2u^2 of the sum whatever cancels.
       */
      _FpType __a_lo;
      const _FpType __a_hi = ::cuda::__fpmp_two_sum(__x_lo, -__h2, &__a_lo);
      __afloat __result(__a_hi, __a_lo);
      __result = __result + __s;

      /* -(l2 + h3 + l3) = u + v + w exactly. Near a multiple of pi/2 the sum so far
       * cancels against u, and v and l3 are what is left of r: rounded into one float
       * they would cost another 2^-49 * n*C3 there, as much as l3 itself is off by.
       * w is below ulp(v) / 2 and goes into the low word.
       */
      _FpType __v;
      _FpType __w;
      const _FpType __u = ::cuda::__fpmp_two_sum(-__l2, -__h3, &__v);
      __v               = ::cuda::__fpmp_two_sum(__v, -__l3, &__w);
      __result          = __result + __afloat(__u, __v);

      *__r_hi = __result.hi();
      *__r_lo = ::cuda::__fpmp_add_rn(__result.lo(), __w);
    }
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
 * Kernel polynomial: the unnormalized pair z = u (k1 + k2 u + ... + k7 u^6) for u = x^2, where
 *   sin(x) = x (1 + z)  with the sine coefficients,
 *   cos(x) = 1 + z      with the cosine coefficients (k1 = -1/2),
 * both minimax on |x| <= 0.91 (Sollya fpminimax, k1..k3 as 48-bit pairs and k4..k7 as single
 * floats; relative error below 2^-57.4 and 2^-55.4). `__cosine` selects the set, at compile
 * time when it is a constant.
 *
 * k4..k7 run in single precision. The rounding error of the last of those steps is up to
 * 2^-25 * k4 * u^4, or 2^-42 of cos(x); it is recovered (k4 - w is exact, as |t u| < k4 / 100)
 * and carried into the pair steps. Each pair step adds its product to a coefficient at least
 * 14 times larger, so the sum is a Fast2Sum. x_lo^2 and the products of two low words are
 * dropped.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_trig_poly(
  const bool __cosine, _FpType __x_hi, _FpType __x_lo, _FpType* __z_hi, _FpType* __z_lo) noexcept
{
  // k1 hi, k1 lo, k2 hi, k2 lo, k3 hi, k3 lo, k4, k5, k6, k7
  constexpr float __sin_k[10] = {
    -0x1.555556p-3f,
    0x1.555554p-28f,
    0x1.111112p-7f,
    -0x1.dddbe4p-32f,
    -0x1.a01a02p-13f,
    0x1.780028p-39f,
    0x1.71de3cp-19f,
    -0x1.ae65dp-26f,
    0x1.61c79ap-33f,
    -0x1.e3e4bap-41f,
  };
  constexpr float __cos_k[10] = {
    -0x1p-1f,
    0.0f,
    0x1.555556p-5f,
    -0x1.5554dp-30f,
    -0x1.6c16c2p-10f,
    0x1.27164cp-35f,
    0x1.a01a02p-16f,
    -0x1.27e528p-22f,
    0x1.1effcap-29f,
    -0x1.9745dcp-37f,
  };
  float __k[10];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int __i = 0; __i < 10; ++__i)
  {
    __k[__i] = __cosine ? __cos_k[__i] : __sin_k[__i];
  }

  float __p_lo;
  float __e;

  // u = x^2
  const float __u_hi = ::cuda::__fpmp_two_mult_fma(__x_hi, __x_hi, &__p_lo);
  const float __u_lo = ::cuda::__fpmp_fma_rn(::cuda::__fpmp_add_rn(__x_hi, __x_hi), __x_lo, __p_lo);

  // w + d = k4 + u (k5 + u (k6 + u k7)), d the rounding error of the last step
  float __t       = ::cuda::__fpmp_fma_rn(__k[9], __u_hi, __k[8]);
  __t             = ::cuda::__fpmp_fma_rn(__t, __u_hi, __k[7]);
  const float __w = ::cuda::__fpmp_fma_rn(__t, __u_hi, __k[6]);
  const float __d = ::cuda::__fpmp_fma_rn(__t, __u_hi, ::cuda::__fpmp_sub_rn(__k[6], __w));

  // q3 = k3 + u (w + d)
  float __p_hi        = ::cuda::__fpmp_two_mult_fma(__w, __u_hi, &__p_lo);
  __p_lo              = ::cuda::__fpmp_fma_rn(__w, __u_lo, __p_lo);
  __p_lo              = ::cuda::__fpmp_fma_rn(__d, __u_hi, __p_lo);
  const float __q3_hi = ::cuda::__fpmp_fast_two_sum(__k[4], __p_hi, &__e);
  const float __q3_lo = ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__k[5], __p_lo), __e);

  // q2 = k2 + u q3
  __p_hi              = ::cuda::__fpmp_two_mult_fma(__q3_hi, __u_hi, &__p_lo);
  __p_lo              = ::cuda::__fpmp_fma_rn(__q3_hi, __u_lo, __p_lo);
  __p_lo              = ::cuda::__fpmp_fma_rn(__q3_lo, __u_hi, __p_lo);
  const float __q2_hi = ::cuda::__fpmp_fast_two_sum(__k[2], __p_hi, &__e);
  const float __q2_lo = ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__k[3], __p_lo), __e);

  // q1 = k1 + u q2
  __p_hi              = ::cuda::__fpmp_two_mult_fma(__q2_hi, __u_hi, &__p_lo);
  __p_lo              = ::cuda::__fpmp_fma_rn(__q2_hi, __u_lo, __p_lo);
  __p_lo              = ::cuda::__fpmp_fma_rn(__q2_lo, __u_hi, __p_lo);
  const float __q1_hi = ::cuda::__fpmp_fast_two_sum(__k[0], __p_hi, &__e);
  const float __q1_lo = ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__k[1], __p_lo), __e);

  // z = u q1
  *__z_hi = ::cuda::__fpmp_two_mult_fma(__q1_hi, __u_hi, &__p_lo);
  __p_lo  = ::cuda::__fpmp_fma_rn(__q1_hi, __u_lo, __p_lo);
  *__z_lo = ::cuda::__fpmp_fma_rn(__q1_lo, __u_hi, __p_lo);
}

/*
 * cos(x) when `__cosine`, else sin(x), for |x| <= 0.91: a (1 + z) with a = 1 or a = x and
 * the matching coefficients. One polynomial serves sin(x) and cos(x) of a reduced argument
 * whose quadrant picks the function. |z| < 1/2, so a + a z is a Fast2Sum.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_sin_or_cos_kernel(
  const bool __cosine, _FpType __x_hi, _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  float __z_hi;
  float __z_lo;
  ::cuda::__internal_fpmp2_trig_poly(__cosine, __x_hi, __x_lo, &__z_hi, &__z_lo);

  const float __a_hi = __cosine ? 1.0f : __x_hi;
  const float __a_lo = __cosine ? 0.0f : __x_lo;

  float __y_lo;
  float __e;
  const float __y_hi = ::cuda::__fpmp_two_mult_fma(__a_hi, __z_hi, &__y_lo);
  __y_lo             = ::cuda::__fpmp_fma_rn(__a_hi, __z_lo, __y_lo);
  __y_lo             = ::cuda::__fpmp_fma_rn(__a_lo, __z_hi, __y_lo);
  const float __r_hi = ::cuda::__fpmp_fast_two_sum(__a_hi, __y_hi, &__e);
  const float __r_lo = ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__a_lo, __y_lo), __e);
  *__res_hi          = ::cuda::__fpmp_fast_two_sum(__r_hi, __r_lo, __res_lo);
}

/*
 * Sin kernel: sin(x) = x (1 + z) for |x| <= 0.91.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sin_kernel(_FpType __x_hi, _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_sin_or_cos_kernel(false, __x_hi, __x_lo, __res_hi, __res_lo);
}

/*
 * Cos kernel: cos(x) = 1 + z for |x| <= 0.91. The same value as
 * __internal_fpmp2_sin_or_cos_kernel gives with a = 1, without its products by 1 and 0.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cos_kernel(_FpType __x_hi, _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  float __z_hi;
  float __z_lo;
  ::cuda::__internal_fpmp2_trig_poly(true, __x_hi, __x_lo, &__z_hi, &__z_lo);

  float __e;
  const float __r_hi = ::cuda::__fpmp_fast_two_sum(1.0f, __z_hi, &__e);
  const float __r_lo = ::cuda::__fpmp_add_rn(__z_lo, __e);
  *__res_hi          = ::cuda::__fpmp_fast_two_sum(__r_hi, __r_lo, __res_lo);
}

/*
 * fpmp2_accuracy::high: sin and cos kernel for |r| <= pi/4 (plus the slack of a
 * float-estimated quadrant), both from one call since every caller needs both:
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
 * This table-driven kernel serves fpmp2_accuracy::low and the default level;
 * fpmp2_accuracy::high has the polynomial kernels further down (__internal_fpmp2_high_*),
 * 1 to 1.5 bits more accurate for two to three times the instructions. Both sit behind the
 * same entry code, so the special values are those of one function.
 *
 * All four functions find the angle of a point (M, m), 0 <= m <= M, from one table:
 *
 *   atan(m / M) = atan(c) + atan(t),   c = i/16,   t = (m - c M) / (M + c m).
 *
 * i is the integer part of 16 m/M + 1/4 (from a single-precision quotient), which leaves
 * -1/64 <= t <= 3/64 and makes the leading words of m and c M cancel exactly (Sterbenz:
 * m / (c M) is within [3/4, 7/4] for i >= 1). The asymmetric interval keeps the first table
 * entry from cancelling half of its own magnitude against t.
 *
 *   atan(x), atan2(y, x):  (M, m) = (|x|, |y|) sorted, one pair division for t.
 *   asin(x), acos(x):      (M, m) = (|x|, sqrt((1 - |x|)(1 + |x|))) sorted, a point of the
 *                          unit circle. There t is replaced by the sine of the same angle,
 *                          (m - c M) cos(atan(c)), a product by a table entry instead of a
 *                          division, and the kernel is asin instead of atan. For i = 0 that
 *                          is m itself, so asin(x) for small x does not depend on the root.
 *
 * The octant is folded into the table: its four rows hold atan(c), pi/2 - atan(c),
 * pi/2 + atan(c) and pi - atan(c), and the odd rows take -t. The rows at c = 0 and c = 1
 * are also the exact results for zero and infinite arguments.
 *
 * Both kernels are t (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8), minimax on |t| <= 0.04689
 * (Sollya fpminimax, k1 and k2 as 48-bit pairs, k3 and k4 as single floats; relative error
 * below 2^-56.3 for atan and 2^-58.4 for asin).
 *
 * NaN, infinite and out-of-domain arguments are answered on entry: nothing below relies on
 * them propagating through the arithmetic.
 * ====================================================================
 */

struct alignas(8) __fpmp2_atan_entry
{
  float __hi;
  float __lo;
};

// Rounded from 400-bit values (Sollya). A static local of a function with internal linkage, like
// the tables of exp and log and for the same reason.
[[nodiscard]] _CCCL_HOST_DEVICE static _CCCL_FORCEINLINE __fpmp2_atan_entry
__fpmp2_atan_table(const int __row, const int __i) noexcept
{
  static constexpr __fpmp2_atan_entry __table[5][17] = {
    // atan(i/16)
    {
      {0.0f, 0.0f},
      {0x1.ff55bcp-5f, -0x1.1a6042p-30f},
      {0x1.fd5baap-4f, -0x1.54f424p-30f},
      {0x1.7b97b4p-3f, 0x1.79cb6p-28f},
      {0x1.f5b76p-3f, -0x1.b4dfc8p-29f},
      {0x1.362774p-2f, -0x1.1f0286p-27f},
      {0x1.6f6194p-2f, 0x1.e4defp-30f},
      {0x1.a64eecp-2f, 0x1.e611fep-29f},
      {0x1.dac67p-2f, 0x1.586ed4p-28f},
      {0x1.0657eap-1f, -0x1.6499e6p-26f},
      {0x1.1e00bap-1f, 0x1.7bdfd6p-26f},
      {0x1.345f02p-1f, -0x1.98e422p-28f},
      {0x1.4978fap-1f, 0x1.934f7p-28f},
      {0x1.5d5898p-1f, 0x1.c5a6c6p-27f},
      {0x1.700a7cp-1f, 0x1.5e118cp-27f},
      {0x1.819d0cp-1f, -0x1.1d4eb6p-26f},
      {0x1.921fb6p-1f, -0x1.777a5cp-26f},
    },
    // pi/2 - atan(i/16)
    {
      {0x1.921fb6p0f, -0x1.777a5cp-25f},
      {0x1.822508p0f, -0x1.2ea75ap-25f},
      {0x1.7249fap0f, 0x1.532d44p-25f},
      {0x1.62acbep0f, 0x1.594c38p-25f},
      {0x1.5368cap0f, -0x1.5c2c6p-25f},
      {0x1.4495d8p0f, 0x1.a08c8ap-26f},
      {0x1.36475p0f, 0x1.e57aaep-27f},
      {0x1.288bfap0f, 0x1.a8920cp-27f},
      {0x1.1b6e1ap0f, -0x1.a28838p-25f},
      {0x1.0ef3cp0f, 0x1.3ad296p-25f},
      {0x1.031f58p0f, -0x1.ab5242p-28f},
      {0x1.efe068p-1f, 0x1.77444ep-26f},
      {0x1.dac67p-1f, 0x1.586ed4p-27f},
      {0x1.c6e6d2p-1f, 0x1.71bf18p-29f},
      {0x1.b434eep-1f, 0x1.8809fep-28f},
      {0x1.a2a26p-1f, -0x1.d1a604p-26f},
      {0x1.921fb6p-1f, -0x1.777a5cp-26f},
    },
    // pi/2 + atan(i/16)
    {
      {0x1.921fb6p0f, -0x1.777a5cp-25f},
      {0x1.a21a64p0f, -0x1.c04d6p-25f},
      {0x1.b1f57p0f, -0x1.0887f8p-27f},
      {0x1.c192acp0f, -0x1.2103c4p-27f},
      {0x1.d0d6a2p0f, -0x1.92c85ap-25f},
      {0x1.dfa992p0f, 0x1.031406p-27f},
      {0x1.edf81ap0f, 0x1.2f5936p-26f},
      {0x1.fbb37p0f, 0x1.4dcd86p-26f},
      {0x1.0468a8p1f, 0x1.59c9bep-24f},
      {0x1.0aa5d4p1f, 0x1.eb1c58p-24f},
      {0x1.10900ap1f, -0x1.5cc538p-24f},
      {0x1.16279cp1f, -0x1.d54b7p-24f},
      {0x1.1b6e1ap1f, -0x1.a28838p-24f},
      {0x1.2066p1f, 0x1.7cf7aap-24f},
      {0x1.25127ap1f, -0x1.1ff5fap-25f},
      {0x1.29771ep1f, -0x1.0310dcp-24f},
      {0x1.2d97c8p1f, -0x1.99bc5cp-28f},
    },
    // pi - atan(i/16)
    {
      {0x1.921fb6p1f, -0x1.777a5cp-24f},
      {0x1.8a225ep1f, 0x1.59de48p-25f},
      {0x1.8234d8p1f, -0x1.2268c6p-28f},
      {0x1.7a663ap1f, -0x1.e2e26p-29f},
      {0x1.72c44p1f, -0x1.69d35ep-24f},
      {0x1.6b5ac6p1f, 0x1.ac65f4p-24f},
      {0x1.643382p1f, 0x1.80f228p-24f},
      {0x1.5d55d8p1f, -0x1.0d55dap-25f},
      {0x1.56c6e8p1f, -0x1.8d014ap-24f},
      {0x1.5089bap1f, 0x1.e1ac1cp-24f},
      {0x1.4a9f86p1f, 0x1.298daep-24f},
      {0x1.4507f4p1f, 0x1.a213e6p-24f},
      {0x1.3fc176p1f, 0x1.6f50acp-24f},
      {0x1.3ac99p1f, -0x1.b02f36p-24f},
      {0x1.361d16p1f, 0x1.730dc6p-26f},
      {0x1.31b872p1f, 0x1.9fb2a2p-25f},
      {0x1.2d97c8p1f, -0x1.99bc5cp-28f},
    },
    // cos(atan(i/16))
    {
      {1.0f, 0.0f},
      {0x1.ff00cp-1f, -0x1.3ee8fcp-26f},
      {0x1.fc0bd8p-1f, 0x1.141e3cp-26f},
      {0x1.f73b06p-1f, -0x1.3e058ep-30f},
      {0x1.f0b684p-1f, 0x1.1a55e4p-26f},
      {0x1.e8b19ap-1f, -0x1.ed38d4p-27f},
      {0x1.df668ap-1f, -0x1.20672cp-27f},
      {0x1.d51292p-1f, 0x1.fd94eap-26f},
      {0x1.c9f25cp-1f, 0x1.6ffb76p-27f},
      {0x1.be3f26p-1f, 0x1.7e31cp-29f},
      {0x1.b22cd6p-1f, -0x1.24f05p-26f},
      {0x1.a5e8d2p-1f, 0x1.6be074p-26f},
      {0x1.99999ap-1f, -0x1.99999ap-27f},
      {0x1.8d5edp-1f, 0x1.c3bcap-27f},
      {0x1.8151bcp-1f, -0x1.e40478p-27f},
      {0x1.7585e6p-1f, 0x1.03fcdep-26f},
      {0x1.6a09e6p-1f, 0x1.9fcef4p-27f},
    },
  };
  return __table[__row][__i];
}

/*
 * atan(m / M) for pairs 0 <= m <= M (to within rounding), M normal, folded into `__row` of
 * the table; `__rho` is m / M in single precision. With `__unit`, (M, m) is a point of the
 * unit circle and the reduced argument is a sine (see above).
 *
 * In the kernel, k3 and k4 run in single precision; the step at k2 adds its product to the
 * low word of k2, which is exact enough because the product is below 2^-9 of k2. The
 * remaining steps are pair steps whose sums are Fast2Sums: each adds a product to a term
 * at least 2^8 times larger.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_atan_kernel(
  const bool __unit,
  const _FpType __m_hi,
  const _FpType __m_lo,
  const _FpType __M_hi,
  const _FpType __M_lo,
  const _FpType __rho,
  const int __row,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  const float __k1_hi = __unit ? 0x1.555556p-3f : -0x1.555556p-2f;
  const float __k1_lo = __unit ? -0x1.5555fcp-28f : 0x1.5556a8p-27f;
  const float __k2_hi = __unit ? 0x1.333334p-4f : 0x1.99999ap-3f;
  const float __k2_lo = __unit ? -0x1.834838p-29f : -0x1.f455e8p-29f;
  const float __k3    = __unit ? 0x1.6db664p-5f : -0x1.2491dp-3f;
  const float __k4    = __unit ? 0x1.f3bf64p-6f : 0x1.c51d9ap-4f;

  float __e;

  const int __i   = static_cast<int>(::cuda::__fpmp_fma_rn(16.0f, __rho, 0.25f));
  const float __c = ::cuda::__fpmp_mul_rn(static_cast<float>(__i), 0.0625f);

  // n = m - c M: the high words cancel exactly, and the difference is renormalized because
  // its low word can exceed what is left of the high one
  float __cM_lo;
  const float __cM_hi = ::cuda::__fpmp_two_mult_fma(__c, __M_hi, &__cM_lo);
  __cM_lo             = ::cuda::__fpmp_fma_rn(__c, __M_lo, __cM_lo);
  float __n_lo;
  const float __n_hi =
    ::cuda::__fpmp_two_sum(::cuda::__fpmp_sub_rn(__m_hi, __cM_hi), ::cuda::__fpmp_sub_rn(__m_lo, __cM_lo), &__n_lo);

  float __t_hi;
  float __t_lo;
  if (__unit)
  {
    // t = n cos(atan(c))
    const __fpmp2_atan_entry __cos = ::cuda::__fpmp2_atan_table(4, __i);
    __t_hi                         = ::cuda::__fpmp_two_mult_fma(__n_hi, __cos.__hi, &__t_lo);
    __t_lo                         = ::cuda::__fpmp_fma_rn(__n_hi, __cos.__lo, __t_lo);
    __t_lo                         = ::cuda::__fpmp_fma_rn(__n_lo, __cos.__hi, __t_lo);
  }
  else
  {
    // t = n / (M + c m)
    float __cm_lo;
    const float __cm_hi = ::cuda::__fpmp_two_mult_fma(__c, __m_hi, &__cm_lo);
    __cm_lo             = ::cuda::__fpmp_fma_rn(__c, __m_lo, __cm_lo);
    const float __d_hi  = ::cuda::__fpmp_fast_two_sum(__M_hi, __cm_hi, &__e);
    const float __d_lo  = ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__M_lo, __cm_lo), __e);
    ::cuda::__fpmp2_div(__n_hi, __n_lo, __d_hi, __d_lo, &__t_hi, &__t_lo);
  }
  if ((__row & 1) != 0)
  {
    __t_hi = -__t_hi;
    __t_lo = -__t_lo;
  }

  // u = t^2
  float __u_lo;
  const float __u_hi = ::cuda::__fpmp_two_mult_fma(__t_hi, __t_hi, &__u_lo);
  __u_lo             = ::cuda::__fpmp_fma_rn(::cuda::__fpmp_add_rn(__t_hi, __t_hi), __t_lo, __u_lo);

  // q2 = k2 + u (k3 + u k4), as k2_hi plus a low word
  const float __w     = ::cuda::__fpmp_fma_rn(__k4, __u_hi, __k3);
  const float __q2_lo = ::cuda::__fpmp_fma_rn(__w, __u_hi, __k2_lo);

  // q1 = k1 + u q2
  float __p_lo;
  const float __p_hi  = ::cuda::__fpmp_two_mult_fma(__k2_hi, __u_hi, &__p_lo);
  __p_lo              = ::cuda::__fpmp_fma_rn(__q2_lo, __u_hi, __p_lo);
  __p_lo              = ::cuda::__fpmp_fma_rn(__k2_hi, __u_lo, __p_lo);
  const float __q1_hi = ::cuda::__fpmp_fast_two_sum(__k1_hi, __p_hi, &__e);
  const float __q1_lo = ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__k1_lo, __p_lo), __e);

  // z = u q1
  const float __z_hi = ::cuda::__fpmp_two_mult_fma(__q1_hi, __u_hi, &__p_lo);
  __p_lo             = ::cuda::__fpmp_fma_rn(__q1_hi, __u_lo, __p_lo);
  const float __z_lo = ::cuda::__fpmp_fma_rn(__q1_lo, __u_hi, __p_lo);

  // y = t z
  const float __y_hi = ::cuda::__fpmp_two_mult_fma(__t_hi, __z_hi, &__p_lo);
  __p_lo             = ::cuda::__fpmp_fma_rn(__t_hi, __z_lo, __p_lo);
  const float __y_lo = ::cuda::__fpmp_fma_rn(__t_lo, __z_hi, __p_lo);

  // table entry + t + y: the entry is zero or exceeds |t|, and |y| < 2^-9 |t|
  const __fpmp2_atan_entry __s = ::cuda::__fpmp2_atan_table(__row, __i);
  float __e2;
  const float __h1 = ::cuda::__fpmp_fast_two_sum(__s.__hi, __t_hi, &__e);
  const float __h2 = ::cuda::__fpmp_fast_two_sum(__h1, __y_hi, &__e2);
  const float __l  = ::cuda::__fpmp_add_rn(
    ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__e, __e2), __s.__lo), ::cuda::__fpmp_add_rn(__t_lo, __y_lo));
  *__res_hi = ::cuda::__fpmp_fast_two_sum(__h2, __l, __res_lo);
}

/*
 * The pair s = sqrt((1 - a)(1 + a)) for a pair 0 <= a <= 1, and min(a, s) / max(a, s) in
 * single precision (`__rho`).
 *
 * 1 - a and 1 + a are exact up to the rounding of their low words, so the product keeps its
 * relative accuracy as a approaches 1. The root is y ~ sqrt(v) and h ~ 1 / (2 sqrt(v)) in
 * single precision, then the residual v - y^2 (exact in its high word) times h. The host takes
 * y from sqrtf; the device refines its approximate reciprocal square root by one coupled
 * Newton step on y and h, and since that instruction flushes subnormals to zero, a product
 * below 2^-64 is scaled first.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_asin_cofunction(
  const _FpType __a_hi, const _FpType __a_lo, _FpType* __s_hi, _FpType* __s_lo, _FpType* __rho) noexcept
{
  float __e;
  float __om_lo;
  const float __h     = ::cuda::__fpmp_fast_two_sum(1.0f, -__a_hi, &__e);
  const float __om_hi = ::cuda::__fpmp_fast_two_sum(__h, ::cuda::__fpmp_sub_rn(__e, __a_lo), &__om_lo);
  const float __op_hi = ::cuda::__fpmp_fast_two_sum(1.0f, __a_hi, &__e);
  const float __op_lo = ::cuda::__fpmp_add_rn(__e, __a_lo);

  float __v_lo;
  float __v_hi = ::cuda::__fpmp_two_mult_fma(__om_hi, __op_hi, &__v_lo);
  __v_lo       = ::cuda::__fpmp_fma_rn(__om_hi, __op_lo, __v_lo);
  __v_lo       = ::cuda::__fpmp_fma_rn(__om_lo, __op_hi, __v_lo);

  const bool __tiny  = __v_hi < 0x1p-64f;
  const float __up   = __tiny ? 0x1p64f : 1.0f;
  const float __down = __tiny ? 0x1p-32f : 1.0f;
  __v_hi             = ::cuda::__fpmp_mul_rn(__v_hi, __up);
  __v_lo             = ::cuda::__fpmp_mul_rn(__v_lo, __up);

  // v = 0 only for a = 1
  float __y;
  float __hh;
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    ({
      const float __r0 = (__v_hi > 0.0f) ? ::cuda::__fpmp_rsqrt_rn(__v_hi) : 0.0f;
      const float __y0 = ::cuda::__fpmp_mul_rn(__v_hi, __r0);
      const float __h0 = ::cuda::__fpmp_mul_rn(0.5f, __r0);
      const float __d  = ::cuda::__fpmp_fma_rn(-__y0, __h0, 0.5f);
      __y              = ::cuda::__fpmp_fma_rn(__y0, __d, __y0);
      __hh             = ::cuda::__fpmp_fma_rn(__h0, __d, __h0);
    }),
    ({
      __y  = ::sqrtf(__v_hi);
      __hh = (__v_hi > 0.0f) ? 0.5f / __y : 0.0f;
    }))
  float __resid = ::cuda::__fpmp_fma_rn(-__y, __y, __v_hi);
  __resid       = ::cuda::__fpmp_add_rn(__resid, __v_lo);

  *__s_hi = ::cuda::__fpmp_mul_rn(__y, __down);
  *__s_lo = ::cuda::__fpmp_mul_rn(::cuda::__fpmp_mul_rn_nocontract(__resid, __hh), __down);

  // a > s: s / a, the only case the scaling above can apply to; else a / s, with 1 / s = 2 h
  *__rho = (__a_hi > *__s_hi) ? ::cuda::__fpmp_mul_rn(*__s_hi, ::cuda::__fpmp_rcp_rn(__a_hi))
                              : ::cuda::__fpmp_mul_rn(__a_hi, ::cuda::__fpmp_add_rn(__hh, __hh));
}

/*
 * ====================================================================
 * fpmp2_accuracy::high kernels for asin, acos, atan and atan2 (fp32mp2)
 * ====================================================================
 *
 * The high level evaluates polynomials in fp32mp2 arithmetic instead of
 * the table: 2 to 3.4 u^2 at most where the table-driven kernel reaches
 * 4 to 7, for two to three times the instructions.  The asin/acos kernels
 * run in pure compensated Horner (M = 0): on their ranges even the
 * highest coefficients carry float-rounding noise above the fp32mp2 ulp
 * when stored as `float`.  The atan kernel's argument is folded once by
 * its callers (see kernel 1), which shrinks its polynomial to 13
 * coefficients with the highest in plain float.
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
 * These kernels see the arguments the shared entry code leaves them: a
 * finite, in-domain |x| (atan: at most 2^60), or for atan2 a pair of
 * magnitudes already scaled so that the larger lies in [1, 2^64).
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
 * Only the small correction w = atan(t) - t is returned; the callers
 * compose base + t + w with the helper below.
 * _FpType is unused -- the kernel is fp32mp2-only -- but keeping it a
 * template spares every translation unit that never calls it an
 * unused-function warning. */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_atan_kernel(const fp32mp2_low& __t, fp32mp2_low* __result) noexcept
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
_CCCL_FPMP_CORE_API fp32mp2_low __internal_fpmp2_high_compose_base_tw(
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
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_asin_poly(const fpmp2<_FpType>& __y, fpmp2<_FpType>* __result) noexcept
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
 * Companion to `__internal_fpmp2_high_asin_poly` for the |x| >= 0.575 branch
 * of acos.  Evaluates the 13-coefficient minimax
 * P(y) such that, for y = 1 - |x|, acos(|x|) = sqrt(2y)*(1 + y*P(y)).
 * Same fp32mp2_low internal evaluation as the asin kernel -- no
 * per-op renormalisation, single conversion in/out around the call. */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_acos_poly(const fpmp2<_FpType>& __y, fpmp2<_FpType>* __result) noexcept
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

/* ---- sqrt(v) for the asin/acos large branches, v = (1 - |x|)/2 or 2 (1 - |x|) ----
 * The pair square root loses its low word below about 2^-100 and, on the
 * device, returns NaN for a subnormal argument; v gets there for a pair
 * (1, -tiny).  Below 2^-64 the argument is scaled by 2^64 and the root by
 * 2^-32, both exact; elsewhere the factors are 1 and the result is the
 * root's own. */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_asin_sqrt(const _FpType __v_hi, const _FpType __v_lo, _FpType* __s_hi, _FpType* __s_lo) noexcept
{
  const bool __tiny  = __v_hi < 0x1p-64f;
  const float __up   = __tiny ? 0x1p64f : 1.0f;
  const float __down = __tiny ? 0x1p-32f : 1.0f;
  float __r_hi, __r_lo;
  ::cuda::__fpmp2_sqrt(::cuda::__fpmp_mul_rn(__v_hi, __up), ::cuda::__fpmp_mul_rn(__v_lo, __up), &__r_hi, &__r_lo);
  *__s_hi = ::cuda::__fpmp_mul_rn(__r_hi, __down);
  *__s_lo = ::cuda::__fpmp_mul_rn(__r_lo, __down);
}

/* ---- atan(a) for a pair 0 <= a <= 2^60 ---- */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_atan_abs(const _FpType __ah, const _FpType __al, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* Three ranges, one refined division (exact-remainder variant) in all of
   * them so the warp never diverges; see kernel 1 for the identities.
   * |x| - 1 is exact for |x| in [1/2, 2] (Sterbenz), |x| + 1 is an exact
   * pair.  The entry code answers |x| > 2^60, where the correction
   * atan(1/|x|) is below 2^-60 of the result, and with it |x| = inf. */
  constexpr float __sqrt13(0x1.279a74p-1f); /* sqrt(1/3) */
  constexpr float __sqrt3(0x1.bb67aep+0f);
  constexpr float __pio4_hi(0x1.921fb6p-1f);
  constexpr float __pio4_lo(-0x1.777a5cp-26f);
  const bool __fold = (__ah > __sqrt13) && (__ah < __sqrt3);
  const bool __inv  = __ah >= __sqrt3;

  float __nl, __de;
  const float __nh = ::cuda::__fpmp_fast_two_sum(::cuda::__fpmp_sub_rn(__ah, 1.0f), __al, &__nl);
  const float __dh = ::cuda::__fpmp_two_sum(__ah, 1.0f, &__de);
  const float __dl = __de + __al;

  const float __num_hi = __inv ? 1.0f : (__fold ? __nh : __ah);
  const float __num_lo = __inv ? 0.0f : (__fold ? __nl : __al);
  const float __den_hi = __inv ? __ah : (__fold ? __dh : 1.0f);
  const float __den_lo = __inv ? __al : (__fold ? __dl : 0.0f);
  float __q_hi, __q_lo;
  ::cuda::__fpmp_math_div_refined<true>(__num_hi, __num_lo, __den_hi, __den_lo, &__q_hi, &__q_lo);
  const __ffloat __t = (__fold || __inv) ? __ffloat(__q_hi, __q_lo) : __ffloat(__ah, __al);

  __ffloat __w;
  ::cuda::__internal_fpmp2_high_atan_kernel<float>(__t, &__w);

  /* r = base + (t + w), or pi/2 - (t + w) for the reciprocal range */
  const float __b_hi = __inv ? __pio4_hi * 2.0f : (__fold ? __pio4_hi : 0.0f);
  const float __b_lo = __inv ? __pio4_lo * 2.0f : (__fold ? __pio4_lo : 0.0f);
  const __ffloat __r = ::cuda::__internal_fpmp2_high_compose_base_tw<float>(__b_hi, __b_lo, __t, __w, __inv);

  *__res_hi = __r.hi();
  *__res_lo = __r.lo();
}

/* ---- atan2(|y|, |x|) reflected for x < 0: the angle in [0, pi] of the point (+-|x|, |y|) ----
 * for finite magnitudes, not both zero, the larger in [1, 2^64). */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_atan2_abs(
  const _FpType __ay_hi,
  const _FpType __ay_lo,
  const _FpType __ax_hi,
  const _FpType __ax_lo,
  const bool __x_is_neg,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;
  const __ffloat __ay(__ay_hi, __ay_lo);
  const __ffloat __ax(__ax_hi, __ax_lo);

  /* One refined division of either min/max or, when the ratio exceeds
   * sqrt(1/3), of (|y| - |x|) / (|y| + |x|), whose magnitude is at most
   * 2 - sqrt(3):
   *     atan2(|y|, |x|) = pi/4 + atan((|y| - |x|) / (|y| + |x|))
   * holds on both sides of the diagonal.  Kernel 1 then sees |t| <=
   * sqrt(1/3).  The scaling of the entry code keeps |y| + |x| finite and
   * the approximate reciprocal in range. */
  constexpr float __sqrt13(0x1.279a74p-1f);
  constexpr float __pio4_hi(0x1.921fb6p-1f);
  constexpr float __pio4_lo(-0x1.777a5cp-26f);
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
  ::cuda::__internal_fpmp2_high_atan_kernel<float>(__t, &__w);

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
  const __ffloat __r = ::cuda::__internal_fpmp2_high_compose_base_tw<float>(__b_hi, __b_lo, __t, __w, __neg_w);

  *__res_hi = __r.hi();
  *__res_lo = __r.lo();
}

/* ---- asin(a) for a pair 0 <= a <= 1 ---- */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_asin_abs(
  const _FpType __a_hi, const _FpType __a_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  using __ffloat = fpmp2<float>;
  const __ffloat __absx(__a_hi, __a_lo);

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
    ::cuda::__internal_fpmp2_high_asin_poly<float>(__a2, &__p);
    __r = renormalize(__absx + __absx * (__a2 * __p));
  }
  else
  {
    /* Large branch: y = (1 - |x|)/2,
     *   asin(|x|) = pi/2 - 2*sqrty*(1 + y*P(y)) */
    __ffloat __y = 0.5f - __absx * 0.5f;
    float __sy_hi, __sy_lo;
    ::cuda::__internal_fpmp2_high_asin_sqrt<float>(__y.hi(), __y.lo(), &__sy_hi, &__sy_lo);
    __ffloat __sy(__sy_hi, __sy_lo);

    __ffloat __p;
    ::cuda::__internal_fpmp2_high_asin_poly<float>(__y, &__p);

    /* 2 sqrt(y) (1 + y P) = 2 sqrt(y) + 2 sqrt(y) (y P): the second term is
     * at most 8 % of the first, so its own roundings are immaterial, and
     * pi/2 - (first + second) is composed with exact low words (the
     * cancellation just above the branch point amplifies every rounding
     * here by 1.5x; the products as float-float operations and plain
     * low-word sums cost ~1 u^2 each, i.e. 44-45 bits). */
    const __ffloat __c = __sy * (__y * __p);
    constexpr float __pio2_hi(0x1.921fb6p+0f);
    constexpr float __pio2_lo(-0x1.777a5cp-25f);
    const fp32mp2_low __r_low = ::cuda::__internal_fpmp2_high_compose_base_tw<float>(
      __pio2_hi,
      __pio2_lo,
      fp32mp2_low(__sy_hi + __sy_hi, __sy_lo + __sy_lo),
      fp32mp2_low(__c.hi() + __c.hi(), __c.lo() + __c.lo()),
      true);
    __r = __ffloat(__r_low.hi(), __r_low.lo());
  }

  *__res_hi = __r.hi();
  *__res_lo = __r.lo();
}

/* ---- acos(+-a) for a pair 0 <= a <= 1 ---- */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_acos_abs(
  const _FpType __a_hi, const _FpType __a_lo, const bool __is_neg, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  using __ffloat = fpmp2<float>;
  const __ffloat __absx(__a_hi, __a_lo);

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
    ::cuda::__internal_fpmp2_high_asin_poly<float>(__a2, &__p);
    __ffloat __asin_abs = renormalize(__absx + __absx * (__a2 * __p));
    __r                 = __is_neg ? renormalize(__PIO2 + __asin_abs) : renormalize(__PIO2 - __asin_abs);
  }
  else
  {
    /* Large branch:
     *   y = 1 - |x|;   acos(|x|) = sqrt(2y)*(1 + y*P(y))
     *   x < 0  ->  acos(x) = pi - acos(|x|)
     * Polynomial P(y) is evaluated by `__internal_fpmp2_high_acos_poly`
     * (analogous to `__internal_fpmp2_high_asin_poly` used by the small
     *  branch and by asin). */
    __ffloat __y     = 1.0f - __absx;
    __ffloat __two_y = __y * 2.0f;
    float __s_hi, __s_lo;
    ::cuda::__internal_fpmp2_high_asin_sqrt<float>(__two_y.hi(), __two_y.lo(), &__s_hi, &__s_lo);
    __ffloat __s(__s_hi, __s_lo);

    __ffloat __p;
    ::cuda::__internal_fpmp2_high_acos_poly<float>(__y, &__p);
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
 * ====================================================================
 * sincos(x, &s, &c) - sine and cosine of the same argument
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Sine and cosine sincos(x, &s, &c) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * sincos for fp32mp2: compute sin(x) and cos(x) simultaneously.
 * Shared argument reduction, separate sin/cos kernels on |r| <= 0.91
 * (fpmp2_accuracy::high: the Taylor kernel), quadrant-based swap and
 * sign adjustment. `__high` is a constant at every call.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincos_core(
  const float __x_hi,
  const float __x_lo,
  float* __sin_hi,
  float* __sin_lo,
  float* __cos_hi,
  float* __cos_lo,
  const bool __high) noexcept
{
  int __quadrant;
  float __r_hi, __r_lo;
  ::cuda::__internal_fpmp2_trig_reduction(__x_hi, __x_lo, &__quadrant, &__r_hi, &__r_lo, __high);

  float __s_hi, __s_lo, __c_hi, __c_lo;
  if (__high)
  {
    ::cuda::__internal_fpmp2_sincos_kernel(__r_hi, __r_lo, &__s_hi, &__s_lo, &__c_hi, &__c_lo);
  }
  else
  {
    ::cuda::__internal_fpmp2_sin_kernel(__r_hi, __r_lo, &__s_hi, &__s_lo);
    ::cuda::__internal_fpmp2_cos_kernel(__r_hi, __r_lo, &__c_hi, &__c_lo);
  }

  const int __q = __quadrant & 3;

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

  // sin(-0) is -0; the kernels' x + x z is +0 there. Both limbs are tested: a low-accuracy
  // difference can carry a zero high limb over a nonzero low one.
  *__sin_hi = (__x_hi == 0.0f && __x_lo == 0.0f) ? __x_hi : __s_hi;
  *__sin_lo = __s_lo;
  *__cos_hi = __c_hi;
  *__cos_lo = __c_lo;
}

/* fpmp2_accuracy::low and mid */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincos(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept
{
  ::cuda::__internal_fpmp2_sincos_core(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo, false);
}

/* fpmp2_accuracy::high: the Taylor kernel and the reduction it was measured with */
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_sincos(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept
{
  ::cuda::__internal_fpmp2_sincos_core(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo, true);
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

/* fp64mp2 has one sincos for every level */
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_sincos(
  const double __x_hi,
  const double __x_lo,
  double* __sin_hi,
  double* __sin_lo,
  double* __cos_hi,
  double* __cos_lo) noexcept
{
  ::cuda::__internal_fpmp2_sincos(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A_2OUT(sincos)
_CCCL_FPMP_MATH_DISPATCH_1A_2OUT(high_sincos)

/*
 * ====================================================================
 * sin(x) - sine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Sine sin(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * sin for fp32mp2: the reduction of sincos, then one kernel. sin(x) is +-sin(r) in an even
 * quadrant and +-cos(r) in an odd one, so the quadrant's parity selects the coefficients of a
 * single polynomial. The result is the sine sincos returns.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sin(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  int __quadrant;
  float __r_hi, __r_lo;
  ::cuda::__internal_fpmp2_trig_reduction(__x_hi, __x_lo, &__quadrant, &__r_hi, &__r_lo, false);

  float __s_hi, __s_lo;
  ::cuda::__internal_fpmp2_sin_or_cos_kernel((__quadrant & 1) != 0, __r_hi, __r_lo, &__s_hi, &__s_lo);

  const bool __negate = (__quadrant & 2) != 0;
  __s_hi              = __negate ? -__s_hi : __s_hi;
  __s_lo              = __negate ? -__s_lo : __s_lo;

  // sin(-0) is -0; the kernel's x + x z is +0 there.
  *__res_hi = (__x_hi == 0.0f && __x_lo == 0.0f) ? __x_hi : __s_hi;
  *__res_lo = __s_lo;
}

/* fpmp2_accuracy::high: the sine of the high sincos (the Taylor kernel evaluates both) */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_sin(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  float __c_hi, __c_lo;
  ::cuda::__internal_fpmp2_high_sincos(__x_hi, __x_lo, __res_hi, __res_lo, &__c_hi, &__c_lo);
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

/* fp64mp2 has one sin for every level */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_sin(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_sin(__x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(sin)
_CCCL_FPMP_MATH_DISPATCH_1A(high_sin)

/*
 * ====================================================================
 * cos(x) - cosine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Cosine cos(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * cos for fp32mp2: the reduction of sincos, then one kernel. cos(x) is +-cos(r) in an even
 * quadrant and +-sin(r) in an odd one, so the quadrant's parity selects the coefficients of a
 * single polynomial. The result is the cosine sincos returns.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cos(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  int __quadrant;
  float __r_hi, __r_lo;
  ::cuda::__internal_fpmp2_trig_reduction(__x_hi, __x_lo, &__quadrant, &__r_hi, &__r_lo, false);

  float __c_hi, __c_lo;
  ::cuda::__internal_fpmp2_sin_or_cos_kernel((__quadrant & 1) == 0, __r_hi, __r_lo, &__c_hi, &__c_lo);

  /* Negate in quadrants 1 and 2 (mod 4). */
  const bool __negate = ((__quadrant + 1) & 2) != 0;
  *__res_hi           = __negate ? -__c_hi : __c_hi;
  *__res_lo           = __negate ? -__c_lo : __c_lo;
}

/* fpmp2_accuracy::high: the cosine of the high sincos (the Taylor kernel evaluates both) */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_cos(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  float __s_hi, __s_lo;
  ::cuda::__internal_fpmp2_high_sincos(__x_hi, __x_lo, &__s_hi, &__s_lo, __res_hi, __res_lo);
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

/* fp64mp2 has one cos for every level */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_cos(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_cos(__x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(cos)
_CCCL_FPMP_MATH_DISPATCH_1A(high_cos)

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
 *   1. Reduce x to r, |r| <= 0.908, via the shared
 *      __internal_fpmp2_trig_reduction; this also returns the quadrant
 *      index q (modulo 4).
 *   2. Evaluate sin(r) and cos(r) on the reduced interval via the
 *      shared __internal_fpmp2_sin_kernel / __internal_fpmp2_cos_kernel
 *      (fpmp2_accuracy::high: __internal_fpmp2_sincos_kernel).
 *   3. tan has period pi, so only the LSB of q matters:
 *        q even  ->  tan(x) =  sin(r) / cos(r)
 *        q odd   ->  tan(x) = -cos(r) / sin(r)        (= -cot(r))
 *      The full quadrant-mod-4 sign dance used by sincos is unnecessary
 *      here because tan(x + pi) = tan(x) absorbs the q == 2,3 sign
 *      flips that sincos performs on its sin/cos outputs.
 *
 * One shared reduction + one sin kernel + one cos kernel + one fp32mp2 division.  Reusing the
 * sin/cos kernels inherits their accuracy without having to fit a separate tan polynomial.
 *
 * Singularities at x === pi/2 (mod pi) produce +-inf through the q-odd
 * branch when sin(r) underflows to zero (matches the IEEE
 * convention; signed-infinity direction follows the rounded reduced
 * argument).  Inf / NaN inputs propagate to NaN through the reduction.
 * `__high` is a constant at every call.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_tan_core(
  const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo, const bool __high) noexcept
{
  int __quadrant;
  float __r_hi, __r_lo;
  ::cuda::__internal_fpmp2_trig_reduction(__x_hi, __x_lo, &__quadrant, &__r_hi, &__r_lo, __high);

  float __s_hi, __s_lo, __c_hi, __c_lo;
  if (__high)
  {
    ::cuda::__internal_fpmp2_sincos_kernel(__r_hi, __r_lo, &__s_hi, &__s_lo, &__c_hi, &__c_lo);
  }
  else
  {
    ::cuda::__internal_fpmp2_sin_kernel(__r_hi, __r_lo, &__s_hi, &__s_lo);
    ::cuda::__internal_fpmp2_cos_kernel(__r_hi, __r_lo, &__c_hi, &__c_lo);
  }

  using __mp2_t = fpmp2<float>;
  __mp2_t __s(__s_hi, __s_lo);
  __mp2_t __c(__c_hi, __c_lo);

  // sin(r) / cos(r), or -cos(r) / sin(r) in an odd quadrant: one division of the selected operands
  const bool __odd       = (__quadrant & 1) != 0;
  const __mp2_t __result = (__odd ? -__c : __s) / (__odd ? __s : __c);

  // tan(-0) is -0; the quotient of the kernels is +0 there.
  *__res_hi = (__x_hi == 0.0f && __x_lo == 0.0f) ? __x_hi : __result.hi();
  *__res_lo = __result.lo();
}

/* fpmp2_accuracy::low and mid */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_tan(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_tan_core(__x_hi, __x_lo, __res_hi, __res_lo, false);
}

/* fpmp2_accuracy::high: the Taylor kernel and the reduction it was measured with */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_tan(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_tan_core(__x_hi, __x_lo, __res_hi, __res_lo, true);
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

/* fp64mp2 has one tan for every level */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_tan(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_tan(__x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(tan)
_CCCL_FPMP_MATH_DISPATCH_1A(high_tan)

/*
 * ====================================================================
 * atan(x) - arc tangent
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Arc tangent atan(x) (fp32mp2)
 * --------------------------------------------------------------------
 * atan(|x|) is the angle of (1, |x|), or of (|x|, 1) taken from pi/2 for |x| > 1, through
 * the shared kernel; with `__high`, the polynomial kernel of fpmp2_accuracy::high on |x|.
 * Beyond 2^60, and for an infinite x, the result is +-pi/2 to the last bit; a NaN gives NaN
 * in both limbs. The sign is applied last, so atan(-0) = -0. The entry code is the same at
 * every accuracy level.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_atan_core(
  const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo, const bool __high) noexcept
{
  const uint32_t __x_bits = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const bool __neg        = (__x_bits & 0x80000000u) != 0u;
  const float __a_hi      = ::cuda::std::bit_cast<float>(__x_bits & 0x7FFFFFFFu);
  const float __a_lo      = __neg ? -__x_lo : __x_lo;

  float __r_hi;
  float __r_lo;
  if (!(__a_hi <= 0x1p60f))
  {
    if (__a_hi != __a_hi)
    {
      *__res_hi = __x_hi;
      *__res_lo = __x_hi;
      return;
    }
    __r_hi = ::cuda::__fpmp2_atan_table(1, 0).__hi;
    __r_lo = ::cuda::__fpmp2_atan_table(1, 0).__lo;
  }
  else if (__high)
  {
    ::cuda::__internal_fpmp2_high_atan_abs(__a_hi, __a_lo, &__r_hi, &__r_lo);
  }
  else
  {
    // |x| > 1: the angle is taken from pi/2
    const bool __swap = __a_hi > 1.0f;
    const float __rho = __swap ? ::cuda::__fpmp_rcp_rn(__a_hi) : __a_hi;
    ::cuda::__internal_fpmp2_atan_kernel(
      false,
      __swap ? 1.0f : __a_hi,
      __swap ? 0.0f : __a_lo,
      __swap ? __a_hi : 1.0f,
      __swap ? __a_lo : 0.0f,
      __rho,
      __swap ? 1 : 0,
      &__r_hi,
      &__r_lo);
  }

  *__res_hi = __neg ? -__r_hi : __r_hi;
  *__res_lo = __neg ? -__r_lo : __r_lo;
}

_CCCL_FPMP_CORE_API void
__internal_fpmp2_atan(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_atan_core(__x_hi, __x_lo, __res_hi, __res_lo, false);
}

/* fpmp2_accuracy::high: the polynomial kernel behind the same entry code */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_atan(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_atan_core(__x_hi, __x_lo, __res_hi, __res_lo, true);
}

/*
 * A NaN result of the fp64mp2 inverse trigonometric functions goes into both limbs, as in
 * their fp32mp2 counterparts.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_nan_both_limbs(double* __res_hi, double* __res_lo) noexcept
{
  if (*__res_hi != *__res_hi)
  {
    *__res_lo = *__res_hi;
  }
}

/*
 * --------------------------------------------------------------------
 * Arc tangent atan(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_atan(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  if (__x_hi == 0.0 && __x_lo == 0.0) // atan(+-0) = +-0: the conversions do not keep the sign of zero
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
    return;
  }
  _CCCL_FPMP_CALL_FP64MP2_MATH(atan, _CCCL_FPMP_ATANQ, __x_hi, __x_lo, __res_hi, __res_lo);
  ::cuda::__internal_fpmp2_nan_both_limbs(__res_hi, __res_lo);
}

/* fp64mp2 has one atan for every level */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_atan(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_atan(__x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(atan)
_CCCL_FPMP_MATH_DISPATCH_1A(high_atan)

/*
 * ====================================================================
 * atan2(y, x) - arc tangent of y/x, quadrant aware
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Arc tangent atan2(y, x) (fp32mp2)
 * --------------------------------------------------------------------
 * The angle of (max(|x|, |y|), min(|x|, |y|)) through the shared kernel, in the table row
 * of its octant, with the sign of y applied last; with `__high`, the polynomial kernel of
 * fpmp2_accuracy::high on the same scaled magnitudes. Special values follow C99 F.10.1.4:
 *
 *   atan2(+-0, x)   = +-0 for x > 0 or +0, +-pi for x < 0 or -0
 *   atan2(y, +-0)   = +-pi/2 for y != 0
 *   atan2(+-inf, x) = +-pi/2 for finite x, +-pi/4 for +inf, +-3pi/4 for -inf
 *   atan2(y, +inf)  = +-0, atan2(y, -inf) = +-pi for finite y
 *   NaN in either argument gives NaN in both limbs
 *
 * These are the table entries at c = 0 and c = 1 of the same row. Signs are those of the
 * high limbs. A low-accuracy difference can carry a zero high limb over a nonzero low one
 * ((1, 0) - (1, 2^-30) is (0, -2^-30)): such an operand is its low limb.
 *
 * The operands are scaled by a common power of two that brings the larger one into
 * [1, 2^64): its reciprocal then stays normal (the device's approximate reciprocal flushes
 * subnormals to zero), the divisor cannot overflow, and the quotient keeps its low word
 * whenever the result has room for one. Subnormal operands are ordinary inputs. The entry
 * code, special values and scaling included, is the same at every accuracy level.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_atan2_core(
  float __y_hi, float __y_lo, float __x_hi, float __x_lo, float* __res_hi, float* __res_lo, const bool __high) noexcept
{
  // An operand with a zero high limb over a nonzero low limb is that low limb. One test of the
  // bit patterns keeps this off the path of ordinary operands.
  const bool __x_hi_zero = (::cuda::std::bit_cast<uint32_t>(__x_hi) << 1) == 0u;
  const bool __y_hi_zero = (::cuda::std::bit_cast<uint32_t>(__y_hi) << 1) == 0u;
  if (__x_hi_zero | __y_hi_zero)
  {
    if (__x_hi == 0.0f && __x_lo != 0.0f)
    {
      __x_hi = __x_lo;
      __x_lo = 0.0f;
    }
    if (__y_hi == 0.0f && __y_lo != 0.0f)
    {
      __y_hi = __y_lo;
      __y_lo = 0.0f;
    }
  }

  const uint32_t __x_bits  = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const uint32_t __y_bits  = ::cuda::std::bit_cast<uint32_t>(__y_hi);
  const uint32_t __ax_bits = __x_bits & 0x7FFFFFFFu;
  const uint32_t __ay_bits = __y_bits & 0x7FFFFFFFu;
  const bool __x_neg       = (__x_bits & 0x80000000u) != 0u;
  const bool __y_neg       = (__y_bits & 0x80000000u) != 0u;

  // |y| > |x|: the angle is taken from the y axis
  const bool __swap = __ay_bits > __ax_bits;
  const int __row   = (__x_neg ? 3 : 0) ^ (__swap ? 1 : 0);

  float __r_hi;
  float __r_lo;
  if (__ax_bits >= 0x7F800000u || __ay_bits >= 0x7F800000u || (__ax_bits | __ay_bits) == 0u)
  {
    if (__ax_bits > 0x7F800000u || __ay_bits > 0x7F800000u)
    {
      *__res_hi = __x_hi + __y_hi;
      *__res_lo = *__res_hi;
      return;
    }
    // Both infinite: the diagonal. Otherwise an axis: one operand infinite, or both zero.
    const __fpmp2_atan_entry __s =
      ::cuda::__fpmp2_atan_table(__row, (__ax_bits == __ay_bits && __ax_bits != 0u) ? 16 : 0);
    __r_hi = __s.__hi;
    __r_lo = __s.__lo;
  }
  else
  {
    const float __ax_hi = ::cuda::std::bit_cast<float>(__ax_bits);
    const float __ay_hi = ::cuda::std::bit_cast<float>(__ay_bits);
    const float __ax_lo = __x_neg ? -__x_lo : __x_lo;
    const float __ay_lo = __y_neg ? -__y_lo : __y_lo;
    float __m_hi        = __swap ? __ax_hi : __ay_hi;
    float __m_lo        = __swap ? __ax_lo : __ay_lo;
    float __M_hi        = __swap ? __ay_hi : __ax_hi;
    float __M_lo        = __swap ? __ay_lo : __ax_lo;

    const float __scale = (__M_hi < 1.0f) ? 0x1p64f : ((__M_hi >= 0x1p64f) ? 0x1p-64f : 1.0f);
    __m_hi              = ::cuda::__fpmp_mul_rn(__m_hi, __scale);
    __m_lo              = ::cuda::__fpmp_mul_rn(__m_lo, __scale);
    __M_hi              = ::cuda::__fpmp_mul_rn(__M_hi, __scale);
    __M_lo              = ::cuda::__fpmp_mul_rn(__M_lo, __scale);
    if (__M_hi < 1.0f) // both operands below 2^-64
    {
      __m_hi = ::cuda::__fpmp_mul_rn(__m_hi, 0x1p86f);
      __m_lo = ::cuda::__fpmp_mul_rn(__m_lo, 0x1p86f);
      __M_hi = ::cuda::__fpmp_mul_rn(__M_hi, 0x1p86f);
      __M_lo = ::cuda::__fpmp_mul_rn(__M_lo, 0x1p86f);
    }

    if (__high)
    {
      // back to (|y|, |x|): the polynomial kernel does its own sorting and folding
      ::cuda::__internal_fpmp2_high_atan2_abs(
        __swap ? __M_hi : __m_hi,
        __swap ? __M_lo : __m_lo,
        __swap ? __m_hi : __M_hi,
        __swap ? __m_lo : __M_lo,
        __x_neg,
        &__r_hi,
        &__r_lo);
    }
    else
    {
      const float __rho = ::cuda::__fpmp_mul_rn(__m_hi, ::cuda::__fpmp_rcp_rn(__M_hi));
      ::cuda::__internal_fpmp2_atan_kernel(false, __m_hi, __m_lo, __M_hi, __M_lo, __rho, __row, &__r_hi, &__r_lo);
    }
  }

  *__res_hi = __y_neg ? -__r_hi : __r_hi;
  *__res_lo = __y_neg ? -__r_lo : __r_lo;
}

_CCCL_FPMP_CORE_API void __internal_fpmp2_atan2(
  const float __y_hi,
  const float __y_lo,
  const float __x_hi,
  const float __x_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_atan2_core(__y_hi, __y_lo, __x_hi, __x_lo, __res_hi, __res_lo, false);
}

/* fpmp2_accuracy::high: the polynomial kernel behind the same entry code */
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_atan2(
  const float __y_hi,
  const float __y_lo,
  const float __x_hi,
  const float __x_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_atan2_core(__y_hi, __y_lo, __x_hi, __x_lo, __res_hi, __res_lo, true);
}

/*
 * --------------------------------------------------------------------
 * Arc tangent atan2(y, x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 * Note: On CUDA device, _CCCL_FPMP_ATAN2Q widens through double atan2 (no fp128 intrinsic); _CCCL_FPMP_CBRTQ is
 * reconstructed from __nv_fp128_pow.
 *
 * The conversions do not keep the sign of a zero y, so that case is answered here:
 * atan2(+-0, x) = +-0 for x > 0 or +0, +-pi for x < 0 or -0. The sign of x is that of its high
 * limb, or of its low limb under a zero high limb (a low-accuracy difference).
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_atan2(
  const double __y_hi,
  const double __y_lo,
  const double __x_hi,
  const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  if (__y_hi == 0.0 && __y_lo == 0.0 && __x_hi == __x_hi)
  {
    const bool __x_neg = ::cuda::std::signbit((__x_hi == 0.0 && __x_lo != 0.0) ? __x_lo : __x_hi);
    const bool __y_neg = ::cuda::std::signbit(__y_hi);
    const double __hi  = __x_neg ? 0x1.921fb54442d18p+1 : 0.0; // pi = __hi + __lo
    const double __lo  = __x_neg ? 0x1.1a62633145c07p-53 : 0.0;
    *__res_hi          = __y_neg ? -__hi : __hi;
    *__res_lo          = __y_neg ? -__lo : __lo;
    return;
  }
#  if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1)
  __fpmp_fp128 __res =
    _CCCL_FPMP_ATAN2Q(::cuda::__fpmp2_to_quad(__y_hi, __y_lo), ::cuda::__fpmp2_to_quad(__x_hi, __x_lo));
  ::cuda::__fpmp2_from_quad(__res, __res_hi, __res_lo);
#  else
  double __res =
    ::cuda::std::atan2(::cuda::__fpmp2_to_double(__y_hi, __y_lo), ::cuda::__fpmp2_to_double(__x_hi, __x_lo));
  ::cuda::__fpmp2_from_double(__res, __res_hi, __res_lo);
#  endif
  ::cuda::__internal_fpmp2_nan_both_limbs(__res_hi, __res_lo);
} // __internal_fpmp2_atan2

/* fp64mp2 has one atan2 for every level */
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_atan2(
  const double __y_hi,
  const double __y_lo,
  const double __x_hi,
  const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_atan2(__y_hi, __y_lo, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_2A_YX(atan2)
_CCCL_FPMP_MATH_DISPATCH_2A_YX(high_atan2)

/*
 * ====================================================================
 * asin(x) - arc sine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Arc sine asin(x) (fp32mp2)
 * --------------------------------------------------------------------
 * asin(|x|) is the angle of the unit-circle point (s, |x|), s = sqrt((1 - |x|)(1 + |x|)),
 * through the shared kernel; with `__high`, the polynomial kernel of fpmp2_accuracy::high on
 * |x|. The sign is applied last, so asin(-0) = -0. |x| > 1 (decided on both limbs) and NaN
 * give NaN in both limbs. The entry code is the same at every accuracy level.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_asin_core(
  const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo, const bool __high) noexcept
{
  const uint32_t __x_bits = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const bool __neg        = (__x_bits & 0x80000000u) != 0u;
  const float __a_hi      = ::cuda::std::bit_cast<float>(__x_bits & 0x7FFFFFFFu);
  const float __a_lo      = __neg ? -__x_lo : __x_lo;

  if (!(__a_hi <= 1.0f) || (__a_hi == 1.0f && __a_lo > 0.0f))
  {
    *__res_hi = (__a_hi != __a_hi) ? __x_hi : ::cuda::std::__fp_nan<float>();
    *__res_lo = *__res_hi;
    return;
  }

  float __r_hi;
  float __r_lo;
  if (__high)
  {
    ::cuda::__internal_fpmp2_high_asin_abs(__a_hi, __a_lo, &__r_hi, &__r_lo);
  }
  else
  {
    float __s_hi;
    float __s_lo;
    float __rho;
    ::cuda::__internal_fpmp2_asin_cofunction(__a_hi, __a_lo, &__s_hi, &__s_lo, &__rho);

    // |x| > s: the angle is taken from pi/2
    const bool __swap = __a_hi > __s_hi;
    ::cuda::__internal_fpmp2_atan_kernel(
      true,
      __swap ? __s_hi : __a_hi,
      __swap ? __s_lo : __a_lo,
      __swap ? __a_hi : __s_hi,
      __swap ? __a_lo : __s_lo,
      __rho,
      __swap ? 1 : 0,
      &__r_hi,
      &__r_lo);
  }

  *__res_hi = __neg ? -__r_hi : __r_hi;
  *__res_lo = __neg ? -__r_lo : __r_lo;
}

_CCCL_FPMP_CORE_API void
__internal_fpmp2_asin(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_asin_core(__x_hi, __x_lo, __res_hi, __res_lo, false);
}

/* fpmp2_accuracy::high: the polynomial kernel behind the same entry code */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_asin(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_asin_core(__x_hi, __x_lo, __res_hi, __res_lo, true);
}

/*
 * --------------------------------------------------------------------
 * asin and acos of an fp64mp2 x with 1/2 <= |x| <= 1
 * --------------------------------------------------------------------
 * There the functions depend on 1 - |x|: acos(|x|) = h = 2 asin(sqrt((1 - |x|) / 2)), which is
 * small near 1, and asin(|x|) = pi/2 - h, acos(-|x|) = pi - h. The sum of the limbs in
 * binary128 (or binary64) drops a low limb beyond its precision, and with it 1 - |x|, so
 * 1 - |x| is formed from the limbs instead: the high limbs subtract exactly. `__form` selects
 * h (0), pi - h (1) or pi/2 - h (2). |x| > 1 gives NaN through the square root.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_acos_from_one(
  const double __a_hi, const double __a_lo, const int __form, double* __res_hi, double* __res_lo) noexcept
{
#  if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1)
  // pi to 113 bits, as the sum of three doubles
  const __fpmp_fp128 __pi =
    static_cast<__fpmp_fp128>(0x1.921fb54442d18p+1) + static_cast<__fpmp_fp128>(0x1.1a62633145c07p-53)
    + static_cast<__fpmp_fp128>(-0x1.f1976b7ed8fbcp-109);
  const __fpmp_fp128 __two = static_cast<__fpmp_fp128>(2);
  const __fpmp_fp128 __d   = static_cast<__fpmp_fp128>(1.0 - __a_hi) - static_cast<__fpmp_fp128>(__a_lo);
  // sqrt(1/2) to 113 bits, as the sum of three doubles. sqrt(d) sqrt(1/2) rather than sqrt(d / 2):
  // where the device has the binary128 type but computes these functions in double, d / 2 rounds
  // to zero as a double for the smallest subnormal d.
  const __fpmp_fp128 __sqrt_half =
    static_cast<__fpmp_fp128>(0x1.6a09e667f3bcdp-1) + static_cast<__fpmp_fp128>(-0x1.bdd3413b26456p-55)
    + static_cast<__fpmp_fp128>(0x1.57d3e3adec175p-109);
  const __fpmp_fp128 __h = __two * _CCCL_FPMP_ASINQ(_CCCL_FPMP_SQRTQ(__d) * __sqrt_half);
  ::cuda::__fpmp2_from_quad((__form == 0) ? __h : ((__form == 1) ? __pi - __h : __pi / __two - __h), __res_hi, __res_lo);
#  else
  constexpr double __pi = 0x1.921fb54442d18p+1;
  const double __d      = (1.0 - __a_hi) - __a_lo;
  // sqrt(1/2) sqrt(d) rather than sqrt(d / 2), which underflows for the smallest subnormal d
  const double __h = 2.0 * ::asin(0x1.6a09e667f3bcdp-1 * ::sqrt(__d));
  ::cuda::__fpmp2_from_double((__form == 0) ? __h : ((__form == 1) ? __pi - __h : 0.5 * __pi - __h), __res_hi, __res_lo);
#  endif
  ::cuda::__internal_fpmp2_nan_both_limbs(__res_hi, __res_lo);
}

/*
 * --------------------------------------------------------------------
 * Arc sine asin(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 * asin(+-0) = +-0 is answered here (the conversions lose the sign of zero), and |x| >= 1/2
 * goes through 1 - |x|, which also decides the domain on both limbs.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_asin(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  if (__x_hi == 0.0 && __x_lo == 0.0)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
    return;
  }
  const bool __neg    = __x_hi < 0.0;
  const double __a_hi = __neg ? -__x_hi : __x_hi;
  if (__a_hi >= 0.5)
  {
    ::cuda::__internal_fpmp2_acos_from_one(__a_hi, __neg ? -__x_lo : __x_lo, 2, __res_hi, __res_lo);
    if (__neg)
    {
      *__res_hi = -*__res_hi;
      *__res_lo = -*__res_lo;
    }
    return;
  }
  _CCCL_FPMP_CALL_FP64MP2_MATH(asin, _CCCL_FPMP_ASINQ, __x_hi, __x_lo, __res_hi, __res_lo);
  ::cuda::__internal_fpmp2_nan_both_limbs(__res_hi, __res_lo);
}

/* fp64mp2 has one asin for every level */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_asin(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_asin(__x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(asin)
_CCCL_FPMP_MATH_DISPATCH_1A(high_asin)

/*
 * ====================================================================
 * acos(x) - arc cosine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Arc cosine acos(x) (fp32mp2)
 * --------------------------------------------------------------------
 * acos(x) is the angle of the unit-circle point (x, s), s = sqrt((1 - |x|)(1 + |x|)),
 * through the shared kernel, in the table rows of the right or the left half plane; with
 * `__high`, the polynomial kernel of fpmp2_accuracy::high on |x|. Either way the result
 * stays accurate for x near 1, where it is small, and acos(1) = +0. |x| > 1 (decided on
 * both limbs) and NaN give NaN in both limbs. The entry code is the same at every accuracy
 * level.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_acos_core(
  const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo, const bool __high) noexcept
{
  const uint32_t __x_bits = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const bool __neg        = (__x_bits & 0x80000000u) != 0u;
  const float __a_hi      = ::cuda::std::bit_cast<float>(__x_bits & 0x7FFFFFFFu);
  const float __a_lo      = __neg ? -__x_lo : __x_lo;

  if (!(__a_hi <= 1.0f) || (__a_hi == 1.0f && __a_lo > 0.0f))
  {
    *__res_hi = (__a_hi != __a_hi) ? __x_hi : ::cuda::std::__fp_nan<float>();
    *__res_lo = *__res_hi;
    return;
  }

  if (__high)
  {
    ::cuda::__internal_fpmp2_high_acos_abs(__a_hi, __a_lo, __neg, __res_hi, __res_lo);
    return;
  }

  float __s_hi;
  float __s_lo;
  float __rho;
  ::cuda::__internal_fpmp2_asin_cofunction(__a_hi, __a_lo, &__s_hi, &__s_lo, &__rho);

  // s > |x|: the angle is taken from pi/2
  const bool __swap = __s_hi > __a_hi;
  ::cuda::__internal_fpmp2_atan_kernel(
    true,
    __swap ? __a_hi : __s_hi,
    __swap ? __a_lo : __s_lo,
    __swap ? __s_hi : __a_hi,
    __swap ? __s_lo : __a_lo,
    __rho,
    (__neg ? 3 : 0) ^ (__swap ? 1 : 0),
    __res_hi,
    __res_lo);
}

_CCCL_FPMP_CORE_API void
__internal_fpmp2_acos(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_acos_core(__x_hi, __x_lo, __res_hi, __res_lo, false);
}

/* fpmp2_accuracy::high: the polynomial kernel behind the same entry code */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_acos(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_acos_core(__x_hi, __x_lo, __res_hi, __res_lo, true);
}

/*
 * --------------------------------------------------------------------
 * Arc cosine acos(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 * |x| >= 1/2 goes through 1 - |x|, which keeps acos accurate where it is small (x near 1) and
 * decides the domain on both limbs.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_acos(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  const bool __neg    = __x_hi < 0.0;
  const double __a_hi = __neg ? -__x_hi : __x_hi;
  if (__a_hi >= 0.5)
  {
    ::cuda::__internal_fpmp2_acos_from_one(__a_hi, __neg ? -__x_lo : __x_lo, __neg ? 1 : 0, __res_hi, __res_lo);
    return;
  }
  _CCCL_FPMP_CALL_FP64MP2_MATH(acos, _CCCL_FPMP_ACOSQ, __x_hi, __x_lo, __res_hi, __res_lo);
  ::cuda::__internal_fpmp2_nan_both_limbs(__res_hi, __res_lo);
}

/* fp64mp2 has one acos for every level */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_acos(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_acos(__x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(acos)
_CCCL_FPMP_MATH_DISPATCH_1A(high_acos)

/*
 * ====================================================================
 * sinpi(x), cospi(x), sincospi(x) - sin(pi*x) and cos(pi*x)
 * ====================================================================
 * fp32mp2 reduces x = n + f, |f| <= 1/2, by splitting the limb pair.
 * f is folded into [-1/4, 1/4], scaled by pi, and evaluated with the
 * sincos kernels: the Taylor kernel at fpmp2_accuracy::high (the
 * __internal_fpmp2_high_ entry points), the minimax kernels otherwise.
 * No radian argument reduction is involved.
 * fp64mp2 remains a double-precision fallback, the same at every level.
 */

_CCCL_FPMP_CORE_API void __internal_fpmp2_sincospi(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept;
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_sincospi(
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

/* fpmp2_accuracy::high */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_sinpi(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  float __c_hi, __c_lo;
  ::cuda::__internal_fpmp2_high_sincospi(__x_hi, __x_lo, __res_hi, __res_lo, &__c_hi, &__c_lo);
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

/* fp64mp2 has one sinpi for every level */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_sinpi(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_sinpi(__x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(sinpi)
_CCCL_FPMP_MATH_DISPATCH_1A(high_sinpi)

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

/* fpmp2_accuracy::high */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_cospi(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  float __s_hi, __s_lo;
  ::cuda::__internal_fpmp2_high_sincospi(__x_hi, __x_lo, &__s_hi, &__s_lo, __res_hi, __res_lo);
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

/* fp64mp2 has one cospi for every level */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_high_cospi(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_cospi(__x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(cospi)
_CCCL_FPMP_MATH_DISPATCH_1A(high_cospi)

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
 * kernels' range, then apply the cofunction identity and the sign (-1)^n.
 * `__high` (a constant at every call) selects the Taylor kernel of
 * fpmp2_accuracy::high; everything else is the same at every level.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincospi_core(
  const float __x_hi,
  const float __x_lo,
  float* __sin_hi,
  float* __sin_lo,
  float* __cos_hi,
  float* __cos_lo,
  const bool __high) noexcept
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
    // sinpi(-0) is -0; the sum of the products is +0 there.
    *__sin_hi = (__x_hi == 0.0f && __x_lo == 0.0f) ? __x_hi : (__a_hi * 0x1p-32f) * 0x1p-32f;
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
  if (__high)
  {
    ::cuda::__internal_fpmp2_sincos_kernel(__angle_hi, __angle_lo, &__s_hi, &__s_lo, &__c_hi, &__c_lo);
  }
  else
  {
    ::cuda::__internal_fpmp2_sin_kernel(__angle_hi, __angle_lo, &__s_hi, &__s_lo);
    ::cuda::__internal_fpmp2_cos_kernel(__angle_hi, __angle_lo, &__c_hi, &__c_lo);
  }

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
    // 0 - c, not -c: cospi is +0 at every half-integer (IEEE 754-2019, 9.2). There the cosine
    // is the kernel's sine of a zero fraction, +0, and the subtraction leaves it +0.
    __c_hi = 0.0f - __c_hi;
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

/* fpmp2_accuracy::low and mid */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincospi(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept
{
  ::cuda::__internal_fpmp2_sincospi_core(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo, false);
}

/* fpmp2_accuracy::high: the Taylor kernel */
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_sincospi(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept
{
  ::cuda::__internal_fpmp2_sincospi_core(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo, true);
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

/* fp64mp2 has one sincospi for every level */
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_sincospi(
  const double __x_hi,
  const double __x_lo,
  double* __sin_hi,
  double* __sin_lo,
  double* __cos_hi,
  double* __cos_lo) noexcept
{
  ::cuda::__internal_fpmp2_sincospi(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A_2OUT(sincospi)
_CCCL_FPMP_MATH_DISPATCH_1A_2OUT(high_sincospi)

#endif // _CCCL_FPMP_USE_LIB
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_TRIG_H
