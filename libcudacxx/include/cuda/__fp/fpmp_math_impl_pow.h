//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_POW_H
#define _CUDA___FP_FPMP_MATH_IMPL_POW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_pow.h - fpmp2 power / root functions (pow, cbrt, rcbrt, hypot, norm/rnorm)
    ==================================================================================================
*/

#include <cuda/__fp/fpmp_math_impl.h>
// Sibling families whose kernels this family calls (exp/log are used by pow).
#include <cuda/__fp/fpmp_math_impl_exp.h>
#include <cuda/std/__floating_point/constants.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * ====================================================================
 * pow(x, y) - x raised to the power y
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Power function pow(x, y) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Algorithm:
 *
 *    pow(a, b) = exp(b * log(|a|)),   log|a| = e ln 2 + L
 *
 * with the exponent part b e ln 2 handled exactly and the exp kernel
 * entered with a directly reduced argument (see the main path), and a
 * sign fixup for a < 0 with integer b.  Dedicated fp32mp2 throughout -
 * no fp64 operations in the main path.
 *
 * Integer-b detection:
 *   b is integer iff b.lo == 0 AND truncf(b.hi) == b.hi.
 *   b is odd integer iff b is integer AND |b.hi| < 2^24 AND
 *   ((int32_t)b.hi & 1) != 0.  Above 2^24 every float-representable
 *   b.hi is automatically even (the LSB has weight >= 2).
 *
 * No b-clamping is needed. For fp32mp2, |log(a)| <= ~88 for any
 * finite a > 0, so `b * loga` overflows to +-Inf only when the true
 * result truly overflows fp32 - and the dedicated `__fpmp2_exp`
 * handles +-Inf input via its existing saturation paths.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_pow(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  /* ---- (1,2) pow(1,b) = pow(a,0) = 1, highest priority per IEEE 754-2008 ---- */
  if ((__a_hi == 1.0f && __a_lo == 0.0f) || (__b_hi == 0.0f && __b_lo == 0.0f))
  {
    *__res_hi = 1.0f;
    *__res_lo = 0.0f;
    return;
  }

  /* ---- (3) NaN propagation ---- */
  if ((__a_hi != __a_hi) || (__b_hi != __b_hi))
  {
    *__res_hi = __a_hi + __b_hi;
    *__res_lo = 0.0f;
    return;
  }

  /* ---- (4) integer / odd-integer b detection ---- */
  bool __b_is_int     = false;
  bool __b_is_odd_int = false;
  {
    const float __b_trunc = ::cuda::__fpmp_internal_trunc<float>(__b_hi);
    if (__b_lo == 0.0f && __b_trunc == __b_hi)
    {
      __b_is_int             = true;
      const float __abs_b_hi = __b_hi < 0.0f ? -__b_hi : __b_hi;
      if (__abs_b_hi < 0x1.0p+24f) /* parity only meaningful below 2^24 */
      {
        __b_is_odd_int = (static_cast<int32_t>(__b_hi) & 1) != 0;
      }
    }
  }

  const bool __a_is_neg  = (__a_hi < 0.0f) || (__a_hi == 0.0f && __a_lo < 0.0f);
  const float __abs_a_hi = __a_is_neg ? -__a_hi : __a_hi;
  const float __abs_a_lo = __a_is_neg ? -__a_lo : __a_lo;

  /* ---- (5) a == 0 ---- */
  if (__abs_a_hi == 0.0f && __abs_a_lo == 0.0f)
  {
    if (__b_hi < 0.0f)
    {
      const float __sign = (__a_is_neg && __b_is_odd_int) ? -1.0f : 1.0f;
      *__res_hi          = __sign * ::cuda::std::__fp_inf<float>();
    }
    else
    {
      *__res_hi = (__a_is_neg && __b_is_odd_int) ? -0.0f : 0.0f;
    }
    *__res_lo = 0.0f;
    return;
  }

  /* ---- (6) negative base with non-integer exponent ---- */
  if (__a_is_neg && !__b_is_int)
  {
    *__res_hi = ::cuda::std::__fp_nan<float>();
    *__res_lo = 0.0f;
    return;
  }

  /* ---- (7) |a| = Inf ---- */
  if (__abs_a_hi == ::cuda::std::__fp_inf<float>())
  {
    const float __sign = (__a_is_neg && __b_is_odd_int) ? -1.0f : 1.0f;
    *__res_hi          = (__b_hi > 0.0f) ? __sign * ::cuda::std::__fp_inf<float>() : __sign * 0.0f;
    *__res_lo          = 0.0f;
    return;
  }

  /* ---- (8) |b| = Inf ---- */
  if (__b_hi == ::cuda::std::__fp_inf<float>() || __b_hi == -::cuda::std::__fp_inf<float>())
  {
    /* IEEE 754: pow(-1, +-Inf) = 1.  pow(+1, ...) already handled at (1). */
    if (__abs_a_hi == 1.0f && __abs_a_lo == 0.0f)
    {
      *__res_hi = 1.0f;
      *__res_lo = 0.0f;
      return;
    }
    const bool __abs_a_gt_one = (__abs_a_hi > 1.0f) || (__abs_a_hi == 1.0f && __abs_a_lo > 0.0f);
    *__res_hi                 = ((__b_hi > 0.0f) == __abs_a_gt_one) ? ::cuda::std::__fp_inf<float>() : 0.0f;
    *__res_lo                 = 0.0f;
    return;
  }

  /* ---- (9) main path: exp(b * log|a|) with the exponent part of log|a|
   * handled exactly.
   *
   * log|a| = e ln 2 + L with |L| <= 0.35 from the shared log core; the
   * previous composition exp(b * log|a|) formed log|a| and the product
   * as float-float values, so the ~2 u^2 of the product was amplified
   * by |b log|a|| in the result (41 bits on the normal dataset at
   * b log|a| = -35).  Here the argument of the exp kernel is reduced
   * directly:
   *     t = b e ln 2 + b L,   N = round(t / ln 2),
   *     r = (b e - N) ln 2 + b L,
   * where b e is an exact sum of two products, b e - N is formed with
   * error-free sums, (b e - N) ln 2 is an exact product with cross terms, and b L is an
   * exact product plus its cross terms.  What remains is
   * the error of L itself and of b L, amplified by |b L| <= 0.35 |b|
   * instead of by |t|.  Results below 2^-102 are still limited by the
   * format (denormal low limb), as for exp. */
  int __e;
  const fp32mp2_low __L = ::cuda::__internal_fpmp2_log_core(__abs_a_hi, __abs_a_lo, 0.0f, &__e);
  const float __e_f     = ::cuda::__fpmp_int2fp_rn<float>(__e);

  constexpr float __inv_ln2(0x1.715476p+0f);
  constexpr float __ln2_h(0x1.62e430p-1f);
  constexpr float __ln2_l(-0x1.05c610p-29f);
  constexpr float __ln2_hi(0x1.62e4000000000p-1f); /* 16-bit head: n * ln2_hi exact for |n| < 256 */
  constexpr float __ln2_mid(0x1.7f7d1c0000000p-20f);
  constexpr float __ln2_lo(0x1.ef357a0000000p-45f);

  /* b e as two exact products; N1 = round(be_h) leaves the exact fraction
   * d = be_h - N1 */
  float __be_l, __bl_l;
  const float __be_h = ::cuda::__fpmp_two_mult_fma(__b_hi, __e_f, &__be_l);
  const float __bl_h = ::cuda::__fpmp_two_mult_fma(__b_lo, __e_f, &__bl_l);

  /* b L as an exact product with exact cross terms */
  float __p_l, __q_l, __w_l;
  const float __p_h = ::cuda::__fpmp_two_mult_fma(__b_hi, __L.hi(), &__p_l);
  const float __q_h = ::cuda::__fpmp_two_mult_fma(__b_hi, __L.lo(), &__q_l);
  const float __w_h = ::cuda::__fpmp_two_mult_fma(__b_lo, __L.hi(), &__w_l);

  /* t estimate for the thresholds and N (float accuracy suffices: an N
   * off by one only widens |r| by ln 2 * 2^-24 |t|) */
  const float __k   = __be_h + (__be_l + __bl_h); /* ~ b e */
  const float __t_f = ::cuda::__fpmp_fma_rn(__k, __ln2_h, __p_h);
  if (__t_f > 0x1.62e430p+6f) /* ln(FLT_MAX): overflow */
  {
    *__res_hi = (__a_is_neg && __b_is_odd_int) ? -::cuda::std::__fp_inf<float>() : ::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }
  if (__t_f < -0x1.9fe3680000000p+6f) /* ln(2^-150): below half the smallest subnormal */
  {
    *__res_hi = (__a_is_neg && __b_is_odd_int) ? -0.0f : 0.0f;
    *__res_lo = 0.0f;
    return;
  }
  const int32_t __n  = ::cuda::__fpmp_fp2int_rn(::cuda::__fpmp_fma_rn(__p_h, __inv_ln2, __k));
  const int32_t __n1 = ::cuda::__fpmp_fp2int_rn(__be_h);
  const float __d    = __be_h - ::cuda::__fpmp_int2fp_rn<float>(__n1); /* exact, |d| <= 1/2 */
  const float __n2   = ::cuda::__fpmp_int2fp_rn<float>(__n - __n1); /* |n2| ~ |b L| / ln 2 */

  /* A = (d + be_l + b_lo e) ln 2: the small fraction of b e, an exact pair
   * times the ln 2 pair (|A| <= 0.35, so the constant's 2^-52.8 is below
   * u^2 of r) */
  float __s1, __s2, __A_l;
  float __S         = ::cuda::__fpmp_two_sum(__d, __be_l, &__s1);
  __S               = ::cuda::__fpmp_two_sum(__S, __bl_h, &__s2);
  const float __S_l = (__s1 + __s2) + __bl_l;
  const float __A   = ::cuda::__fpmp_two_mult_fma(__S, __ln2_h, &__A_l);
  __A_l             = __A_l + ::cuda::__fpmp_fma_rn(__S, __ln2_l, ::cuda::__fpmp_mul_rn(__S_l, __ln2_h));

  /* B = n2 ln 2 with the three-piece constant: n2 ln2_hi is exact (|n2| <
   * 256), n2 ln2_mid an exact pair, the rest float; accurate to ~2^-60 */
  float __bm_l;
  const float __B   = ::cuda::__fpmp_mul_rn(__n2, __ln2_hi);
  const float __bm  = ::cuda::__fpmp_two_mult_fma(__n2, __ln2_mid, &__bm_l);
  const float __B_l = ::cuda::__fpmp_fma_rn(__n2, __ln2_lo, __bm_l);

  /* r = A - B + C with C = b L; B and C cancel to |r| <= ln 2 / 2.  Every
   * piece above 2^-48 |b L| -- the high words, and the three products'
   * low words, which are 2^-24 |b L| and so far larger than r's ulp when
   * |b L| is large -- goes through the error-free chain; only the sum
   * errors of that chain (each below half an ulp of a partial sum, i.e.
   * of r) and the third-order pieces are gathered in float.  What
   * remains is the error of L itself, scaled by |b L|. */
  float __h1, __h2, __h3, __h4, __h5, __h6;
  float __r_h       = ::cuda::__fpmp_two_sum(__p_h, -__B, &__h1);
  __r_h             = ::cuda::__fpmp_two_sum(__r_h, -__bm, &__h2);
  __r_h             = ::cuda::__fpmp_two_sum(__r_h, __A, &__h3);
  __r_h             = ::cuda::__fpmp_two_sum(__r_h, __p_l, &__h4);
  __r_h             = ::cuda::__fpmp_two_sum(__r_h, __q_h, &__h5);
  __r_h             = ::cuda::__fpmp_two_sum(__r_h, __w_h, &__h6);
  const float __r_l = (((__h1 + __h2) + (__h3 + __h4)) + (__h5 + __h6)) + ((__A_l - __B_l) + (__q_l + __w_l));

  float __fscale, __fscale_lo;
  ::cuda::__internal_fpmp2_exp_scales(__n + 254, &__fscale, &__fscale_lo);
  float __t_hi, __t_lo;
  ::cuda::__internal_fpmp2_exp_core(fp32mp2_low(__r_h, __r_l), __fscale, __fscale_lo, &__t_hi, &__t_lo);

  /* ---- sign fixup for a < 0 with odd integer b ---- */
  if (__a_is_neg && __b_is_odd_int)
  {
    __t_hi = -__t_hi;
    __t_lo = -__t_lo;
  }

  *__res_hi = __t_hi;
  *__res_lo = __t_lo;
} // __internal_fpmp2_pow

/*
 * --------------------------------------------------------------------
 * Power function pow(x, y) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_pow(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH_2A(pow, _CCCL_FPMP_POWQ, __x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_2A(pow)

/*
 * ====================================================================
 * cbrt(x) - cube root
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Cube root cbrt(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *   1. Special cases: pass +-0, +-Inf, NaN through unchanged
 *      (cbrt(x) == x for these inputs).
 *   2. Operate on |x|; cbrt is odd, sign is restored at the end.
 *   3. Pre-scale denormal inputs by 2^24 (= 2^(3*8)) so the exponent
 *      extraction sees a normal float; the +8 bias in the result
 *      exponent is undone in the final scaling step.
 *   4. Argument reduction:  ax = r * 2^(3*nexpo)  with
 *      nexpo = round((expo - 126) / 3).  The reduced r sits in
 *      roughly [2^-1, 2^1), which gives the SFU lg2/ex2 pair a tight
 *      enough range that one Halley step recovers full fp32mp2
 *      precision.
 *   5. Seed s0 = fast_exp2(third * fast_log2(r_hi)) from the SFU pair
 *      (~2^-22), polished by one Newton step in float with an exact
 *      s0^2, so s1 is within an ulp of cbrt(r) whatever the seed did.
 *   6. One Halley step on s1 with everything that matters exact:
 *         cbrt(r) = s1 + s1 (r - s1^3) / (2 s1^3 + r) + O(eps^3)
 *      r - s1^3 as an error-free pair (exact cube pieces, Sterbenz,
 *      one two_sum), the denominator as a pair, the quotient corrected
 *      once from its remainder; the correction then carries no
 *      rounding of its own above 2^-47.  Host model against binary128:
 *      1.5 u^2 worst, 0.16 mean, independent of the seed.  (The former
 *      fp32mp2 Halley step rounded its 2^-22 correction through the
 *      1-ulp rcp.approx: 7 u^2, 45 bits on the device.)
 *   7. Multiply the result back by 2^(nexpo - denorm_div3) using
 *      a power-of-two scale factor (exact, no rounding).
 *   8. Restore the sign.
 *
 * No fp64 operations.  Negative inputs are supported (cbrt(-x) = -cbrt(x)).
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cbrt(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  // 1/3 in single precision (round-to-nearest); the exact 1/3 is not
  // representable in any binary float, but ulp(1/3) is well below the
  // accuracy of the SFU lg2/ex2 pair so this is sufficient.
  constexpr float __third_f = 0x1.555556p-2f;

  const uint32_t __xbits   = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const uint32_t __absbits = __xbits & 0x7FFFFFFFu;
  const uint32_t __signbit = __xbits & 0x80000000u;

  if (__absbits == 0u || __absbits >= 0x7F800000u)
  {
    *__res_hi = __x_hi;
    *__res_lo = (__absbits >= 0x7F800000u) ? 0.0f : __x_lo;
    return;
  }

  /* Operate on |x|; sign of x_lo follows the sign of x_hi. */
  float __ax_hi = ::cuda::std::bit_cast<float>(__absbits);
  float __ax_lo = (__signbit != 0u) ? -__x_lo : __x_lo;

  /* Denormal pre-scaling: multiply by 2^24 (chosen so the offset is
   * divisible by 3 -> denorm_div3 = 8 unscales the result later). */
  int __denorm_div3         = 0;
  uint32_t __scaled_absbits = __absbits;
  if ((__absbits >> 23) == 0u)
  {
    constexpr float __scale_up = 0x1.0p24f;
    __ax_hi *= __scale_up;
    __ax_lo *= __scale_up;
    __denorm_div3    = 8;
    __scaled_absbits = ::cuda::std::bit_cast<uint32_t>(__ax_hi);
  }

  /* Reduce: ax = r * 2^(3 * nexpo), with nexpo chosen so r ~= 1. */
  const int __expo  = static_cast<int>(__scaled_absbits >> 23);
  const int __nexpo = ::cuda::__fpmp_fp2int_rn(__third_f * static_cast<float>(__expo - 126));

  /* r_hi = ax_hi * 2^(-3*nexpo): exact, by exponent-field subtraction.
   * (The mantissa is untouched; only the biased exponent shifts.)
   * Use multiplication by 2^23 instead of left-shift to avoid UB
   * when (3 * nexpo) is negative. */
  constexpr int __exp_shift = 1 << 23;
  const int __delta_exp     = 3 * __nexpo;
  const int __new_bits      = static_cast<int>(__scaled_absbits) - __delta_exp * __exp_shift;
  const float __r_hi        = ::cuda::std::bit_cast<float>(static_cast<uint32_t>(__new_bits));

  /* r_lo: scale by the same power of two via float multiply.  Split
   * 2^(-3*nexpo) into two normal-range factors: for x near max float
   * |3*nexpo| can reach ~129, which would give an invalid biased
   * exponent of -2 if applied as a single bit-cast.  Splitting keeps
   * each factor's biased exponent in the normal range (about
   * [62, 190]); the product stays exact for all valid inputs. */
  const int __half_pow  = -__delta_exp / 2;
  const int __rest_pow  = -__delta_exp - __half_pow;
  const float __scale_a = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __half_pow) * __exp_shift));
  const float __scale_b = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __rest_pow) * __exp_shift));
  const float __r_lo    = (__ax_lo * __scale_a) * __scale_b;

  /* Seed: the SFU lg2/ex2 pair, about 2^-22 relative. */
  const float __s0 = ::cuda::__fpmp_fast_exp2(__third_f * ::cuda::__fpmp_fast_log2(__r_hi));

  /* Float polish, one Newton step s1 = s0 - (s0^3 - r) s0 / (3 r) with s0^2 as
   * an exact pair, so that s1 lands within an ulp of cbrt(r) whatever the seed
   * did; the raw approximate reciprocal of r only scales a 2^-22 correction. */
  float __q_lo;
  const float __q_hi = ::cuda::__fpmp_two_mult_fma(__s0, __s0, &__q_lo);
  const float __e0   = ::cuda::__fpmp_fma_rn(__q_hi, __s0, -__r_hi) + __q_lo * __s0;
  const float __g0   = (__third_f * ::cuda::__fpmp_rcp_rn(__r_hi)) * __s0;
  const float __s1   = ::cuda::__fpmp_fma_rn(-__e0, __g0, __s0);

  /* Residual r - s1^3 as an error-free pair: s1^2 and s1^2 s1 are exact
   * products, r_hi minus the cube's high word is exact (Sterbenz), the
   * 2^-24-sized remainders are gathered in float and the two parts are
   * joined by a two_sum (their order is not known). */
  float __l2, __l3, __n_e;
  const float __h2    = ::cuda::__fpmp_two_mult_fma(__s1, __s1, &__l2);
  const float __h3    = ::cuda::__fpmp_two_mult_fma(__h2, __s1, &__l3);
  const float __m3    = __l2 * __s1;
  const float __d     = __r_hi - __h3;
  const float __small = ::cuda::__fpmp_fma_rn(-__l2, __s1, __r_lo - __l3);
  const float __n_h   = ::cuda::__fpmp_two_sum(__d, __small, &__n_e);

  /* Halley: cbrt(r) = s1 + s1 (r - s1^3) / (2 s1^3 + r) + O(eps^3).  The
   * denominator is a pair (2 h3 + r_hi by a Fast2Sum, the rest in float, which
   * only has to be present) and the quotient is corrected once from its
   * remainder, so the correction carries no rounding of its own above 2^-47;
   * a 2^-23-sized piece left for the low word would cost 2^-47 in the final
   * addition, hence the two_sum above. */
  float __dn_e, __ns_l, __t_l, __t_lo;
  const float __dn_h = ::cuda::__fpmp_fast_two_sum(__h3 + __h3, __r_hi, &__dn_e);
  const float __dn_l = ::cuda::__fpmp_fma_rn(2.0f, __l3, ::cuda::__fpmp_fma_rn(2.0f, __m3, __dn_e + __r_lo));
  const float __inv  = ::cuda::__fpmp_rcp_rn(__dn_h);
  const float __ns_h = ::cuda::__fpmp_two_mult_fma(__n_h, __s1, &__ns_l);
  __ns_l             = ::cuda::__fpmp_fma_rn(__n_e, __s1, __ns_l);
  const float __q    = __ns_h * __inv;
  float __rem        = ::cuda::__fpmp_fma_rn(-__dn_h, __q, __ns_h);
  __rem              = ::cuda::__fpmp_fma_rn(-__dn_l, __q, __rem) + __ns_l;
  const float __q_l  = __rem * __inv;
  const float __t_h  = ::cuda::__fpmp_fast_two_sum(__s1, __q, &__t_l);
  const float __t_hi = ::cuda::__fpmp_fast_two_sum(__t_h, __t_l + __q_l, &__t_lo);

  /* Scale back by 2^(nexpo - denorm_div3) via an exact power-of-two
   * float multiply.  back_shift stays in a range that keeps the
   * scale factor a normal float for all valid float inputs
   * (biased exponent is always in [77, 170]). */
  const int __back_shift   = __nexpo - __denorm_div3;
  const float __scale_back = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __back_shift) * __exp_shift));
  float __t_hi_back        = __t_hi * __scale_back;
  float __t_lo_back        = __t_lo * __scale_back;

  /* Restore sign (cbrt is an odd function). */
  if (__signbit != 0u)
  {
    __t_hi_back = -__t_hi_back;
    __t_lo_back = -__t_lo_back;
  }

  *__res_hi = __t_hi_back;
  *__res_lo = __t_lo_back;
} // __internal_fpmp2_cbrt

/*
 * --------------------------------------------------------------------
 * Cube root cbrt(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cbrt(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
#  if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1)
  __fpmp_fp128 __res = _CCCL_FPMP_CBRTQ(::cuda::__fpmp2_to_quad(__x_hi, __x_lo));
  ::cuda::__fpmp2_from_quad(__res, __res_hi, __res_lo);
#  else
  double __res = ::cuda::std::cbrt(::cuda::__fpmp2_to_double(__x_hi, __x_lo));
  ::cuda::__fpmp2_from_double(__res, __res_hi, __res_lo);
#  endif
}

_CCCL_FPMP_MATH_DISPATCH_1A(cbrt)

/*
 * ====================================================================
 * rcbrt(x) - reciprocal cube root
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Reciprocal cube root rcbrt(x) = 1/cbrt(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *   1. Special cases:
 *        cbrt(+-0)   = +-Inf  (result inherits the sign of x)
 *        cbrt(+-Inf) = +-0
 *        cbrt(NaN)  = NaN
 *   2. Operate on |x|; rcbrt is odd, sign is restored at the end.
 *   3. Pre-scale denormal inputs by 2^24 (= 2^(3*8)) so the exponent
 *      extraction sees a normal float; the +8 offset ends up as a
 *      +8 shift on the back-scale (denorm_div3 = 8).
 *   4. Argument reduction:  ax = r * 2^(3*nexpo) with
 *      nexpo = round((expo - 126) / 3).  Reduced r sits in roughly
 *      [2^-2, 2^2), giving the SFU lg2/ex2 pair a tight enough range
 *      that one Halley step recovers full fp32mp2 precision.
 *   5. Seed s0 = fast_exp2(-third * fast_log2(r_hi)) from the SFU pair
 *      (~2^-22), polished by one Newton step in float (no reciprocal
 *      needed), so s1 is within an ulp of 1/cbrt(r) whatever the seed did.
 *   6. One Halley step on s1 with the residual exact:
 *          u     = 1 - r s1^3        (error-free pair: exact cube pieces,
 *                                     1 - (r s1^3)_hi exact, the four
 *                                     2^-24-sized remainders by two two_sums)
 *          t_new = s1 + u s1 (1/3 + (2/9) u)
 *      u s1 and (u s1)/3 as exact products with 1/3 as a pair; the Halley
 *      term and the remainders go to the low word.  Host model against
 *      binary128: 1.4 u^2 worst, 0.18 mean, independent of the seed
 *      (3 u^2 before, with the correction rounded in fp32mp2 arithmetic).
 *   7. Multiply the result back by 2^(-nexpo + denorm_div3) using a
 *      power-of-two scale factor (exact, no rounding).
 *   8. Restore the sign.
 *
 * No fp64 operations.  No final reciprocal: the algorithm targets 1/cbrt
 * directly, which is why it is faster than cbrt(x) followed by a divide.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_rcbrt(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  // 1/3 and 2/9 in single precision (round-to-nearest); ulp at this
  // scale is far below the SFU lg2/ex2 estimate's accuracy.
  constexpr float __third_f      = 0x1.555556p-2f; // ~= 1/3
  constexpr float __two_ninths_f = 0x1.c71c72p-3f; // ~= 2/9

  const uint32_t __xbits   = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const uint32_t __absbits = __xbits & 0x7FFFFFFFu;
  const uint32_t __signbit = __xbits & 0x80000000u;

  /* Special inputs:
   *   +-0   -> +-Inf
   *   +-Inf -> +-0
   *   NaN  -> NaN  (propagated via x_hi)
   */
  if (__absbits == 0u || __absbits >= 0x7F800000u)
  {
    if (__absbits == 0u)
    {
      *__res_hi = ::cuda::std::bit_cast<float>(__signbit | 0x7F800000u);
    }
    else if (__absbits == 0x7F800000u)
    {
      *__res_hi = ::cuda::std::bit_cast<float>(__signbit);
    }
    else
    {
      *__res_hi = __x_hi; // NaN
    }
    *__res_lo = 0.0f;
    return;
  }

  /* Operate on |x|; sign of x_lo follows the sign of x_hi. */
  float __ax_hi = ::cuda::std::bit_cast<float>(__absbits);
  float __ax_lo = (__signbit != 0u) ? -__x_lo : __x_lo;

  /* Denormal pre-scaling: multiply by 2^24. */
  int __denorm_div3         = 0;
  uint32_t __scaled_absbits = __absbits;
  if ((__absbits >> 23) == 0u)
  {
    constexpr float __scale_up = 0x1.0p24f;
    __ax_hi *= __scale_up;
    __ax_lo *= __scale_up;
    __denorm_div3    = 8;
    __scaled_absbits = ::cuda::std::bit_cast<uint32_t>(__ax_hi);
  }

  /* Reduce: ax = r * 2^(3 * nexpo), with nexpo chosen so r ~= 1. */
  const int __expo  = static_cast<int>(__scaled_absbits >> 23);
  const int __nexpo = ::cuda::__fpmp_fp2int_rn(__third_f * static_cast<float>(__expo - 126));

  /* r_hi = ax_hi * 2^(-3*nexpo): exact, by exponent-field subtraction.
   * Use multiplication by 2^23 instead of left-shift to avoid UB
   * when (3 * nexpo) is negative. */
  constexpr int __exp_shift = 1 << 23;
  const int __delta_exp     = 3 * __nexpo;
  const int __new_bits      = static_cast<int>(__scaled_absbits) - __delta_exp * __exp_shift;
  const float __r_hi        = ::cuda::std::bit_cast<float>(static_cast<uint32_t>(__new_bits));

  /* r_lo: scale by 2^(-3*nexpo) via float multiply.  Split into two
   * normal-range factors to keep each biased exponent in roughly
   * [62, 190] for all valid float inputs. */
  const int __half_pow  = -__delta_exp / 2;
  const int __rest_pow  = -__delta_exp - __half_pow;
  const float __scale_a = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __half_pow) * __exp_shift));
  const float __scale_b = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __rest_pow) * __exp_shift));
  const float __r_lo    = (__ax_lo * __scale_a) * __scale_b;

  /* Seed: the SFU lg2/ex2 pair, about 2^-22 relative. */
  const float __s0 = ::cuda::__fpmp_fast_exp2(-__third_f * ::cuda::__fpmp_fast_log2(__r_hi));

  /* Float polish, one Newton step s1 = s0 (1 + (1 - r s0^3) / 3): no reciprocal
   * needed, and s1 lands within an ulp of 1/cbrt(r) whatever the seed did. */
  const float __c0 = (__s0 * __s0) * __s0;
  const float __e0 = ::cuda::__fpmp_fma_rn(-__r_hi, __c0, 1.0f);
  const float __s1 = ::cuda::__fpmp_fma_rn(__e0 * __third_f, __s0, __s0);

  /* Residual u = 1 - r s1^3 as an error-free pair: s1^2, s1^2 s1 and
   * r_hi (s1^3)_hi are exact products, 1 minus the last is exact, and the four
   * 2^-24-sized remainders enter the HIGH word through two two_sums (left for
   * the low word they would cost 2^-47 in the final addition). */
  float __l2, __l3, __p_l, __u_l, __u_l2;
  const float __h2  = ::cuda::__fpmp_two_mult_fma(__s1, __s1, &__l2);
  const float __h3  = ::cuda::__fpmp_two_mult_fma(__h2, __s1, &__l3);
  const float __m3  = __l2 * __s1;
  const float __p_h = ::cuda::__fpmp_two_mult_fma(__r_hi, __h3, &__p_l);
  const float __d   = 1.0f - __p_h;
  const float __w1  = ::cuda::__fpmp_fma_rn(__r_hi, __l3, __p_l);
  const float __w2  = ::cuda::__fpmp_fma_rn(__r_lo, __h3, __r_hi * __m3);
  float __u         = ::cuda::__fpmp_two_sum(__d, -__w1, &__u_l);
  __u               = ::cuda::__fpmp_two_sum(__u, -__w2, &__u_l2);
  __u_l += __u_l2;

  /* Halley: 1/cbrt(r) = s1 + u s1 (1/3 + (2/9) u) + O(u^3).  u s1 and (u s1)/3
   * are exact products with 1/3 as a pair; the Halley term and all remainders
   * go to the low word. */
  constexpr float __third_lo_f = -0x1.555556p-27f; // 1/3 - third_f
  float __us_l, __pp_e, __t_l, __t_lo;
  const float __us   = ::cuda::__fpmp_two_mult_fma(__u, __s1, &__us_l);
  __us_l             = ::cuda::__fpmp_fma_rn(__u_l, __s1, __us_l);
  const float __pp   = ::cuda::__fpmp_two_mult_fma(__us, __third_f, &__pp_e);
  float __pp_l       = ::cuda::__fpmp_fma_rn(__us, __third_lo_f, __pp_e);
  __pp_l             = ::cuda::__fpmp_fma_rn(__us_l, __third_f, __pp_l);
  __pp_l             = ::cuda::__fpmp_fma_rn(__us * __u, __two_ninths_f, __pp_l);
  const float __t_h  = ::cuda::__fpmp_fast_two_sum(__s1, __pp, &__t_l);
  const float __t_hi = ::cuda::__fpmp_fast_two_sum(__t_h, __t_l + __pp_l, &__t_lo);

  /* Scale back by 2^(-nexpo + denorm_div3) via an exact power-of-two
   * float multiply.  back_shift stays in [-43, +49] for all valid
   * float inputs, so the biased exponent is always in [84, 176]. */
  const int __back_shift   = -__nexpo + __denorm_div3;
  const float __scale_back = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __back_shift) * __exp_shift));
  float __t_hi_back        = __t_hi * __scale_back;
  float __t_lo_back        = __t_lo * __scale_back;

  /* Restore sign (rcbrt is an odd function). */
  if (__signbit != 0u)
  {
    __t_hi_back = -__t_hi_back;
    __t_lo_back = -__t_lo_back;
  }

  *__res_hi = __t_hi_back;
  *__res_lo = __t_lo_back;
} // __internal_fpmp2_rcbrt

/*
 * --------------------------------------------------------------------
 * Reciprocal cube root rcbrt(x) = 1/cbrt(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_rcbrt(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  double __xd = ::cuda::__fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (::cuda::__fpmp2_from_double(::rcbrt(__xd), __res_hi, __res_lo);),
                    (::cuda::__fpmp2_from_double(1.0 / ::cuda::std::cbrt(__xd), __res_hi, __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_1A(rcbrt)

/*
 * ====================================================================
 * hypot(x, y) - sqrt(x^2 + y^2) without spurious overflow
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * x^2 + y^2 for the fp32mp2 hypot / rhypot kernels
 * --------------------------------------------------------------------
 * Both inputs are scaled by one power of two 2^-k chosen from the
 * larger magnitude (k in [-126, 126]: the squares can neither overflow
 * nor lose their low limbs to underflow, and the scaling is exact), and
 * the sum is formed as a normalized pair: the two squares as error-free
 * products with their cross terms 2 x_hi x_lo folded into the low words
 * by one fma each, the larger square plus the smaller one by a Fast2Sum,
 * the two low words folded into the high word by two more Fast2Sums in
 * magnitude order, the remainders and x_lo^2 + y_lo^2 summed in float.
 * Host model against binary128, through the sqrt core: 2.6 u^2 worst,
 * 0.3 mean (the core alone is 2.5).  Returns k; the caller scales its
 * root back by 2^k (the squares were scaled by 2^-2k), exactly.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API int __internal_fpmp2_hypot_sum(
  const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __a_hi, float* __a_lo) noexcept
{
  constexpr uint32_t __abs_mask = 0x7FFFFFFFu;
  const float __ax_hi           = ::cuda::std::bit_cast<float>(::cuda::std::bit_cast<uint32_t>(__x_hi) & __abs_mask);
  const float __ay_hi           = ::cuda::std::bit_cast<float>(::cuda::std::bit_cast<uint32_t>(__y_hi) & __abs_mask);
  const float __ax_lo           = (__x_hi < 0.0f) ? -__x_lo : __x_lo;
  const float __ay_lo           = (__y_hi < 0.0f) ? -__y_lo : __y_lo;
  /* The larger magnitude first: the Fast2Sums below depend on it. */
  const bool __swap = __ax_hi < __ay_hi;
  float __bx_hi     = __swap ? __ay_hi : __ax_hi;
  float __bx_lo     = __swap ? __ay_lo : __ax_lo;
  float __by_hi     = __swap ? __ax_hi : __ay_hi;
  float __by_lo     = __swap ? __ax_lo : __ay_lo;
  /* k: the exponent of the larger magnitude, clamped so that 2^-k and 2^k
   * are normal floats (denormals and zero scale by 2^126). */
  const int __e       = static_cast<int>(::cuda::std::bit_cast<uint32_t>(__bx_hi) >> 23);
  const int __k       = ((__e < 1) ? 1 : ((__e > 253) ? 253 : __e)) - 127;
  const float __scale = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 - __k) << 23));
  __bx_hi *= __scale;
  __bx_lo *= __scale;
  __by_hi *= __scale;
  __by_lo *= __scale;
  /* Squares with their cross terms, then the pair. */
  float __xx_l, __yy_l, __e0, __e1, __e2;
  const float __xx_h  = ::cuda::__fpmp_two_mult_fma(__bx_hi, __bx_hi, &__xx_l);
  const float __yy_h  = ::cuda::__fpmp_two_mult_fma(__by_hi, __by_hi, &__yy_l);
  const float __x_low = ::cuda::__fpmp_fma_rn(__bx_lo + __bx_lo, __bx_hi, __xx_l);
  const float __y_low = ::cuda::__fpmp_fma_rn(__by_lo + __by_lo, __by_hi, __yy_l);
  float __s           = ::cuda::__fpmp_fast_two_sum(__xx_h, __yy_h, &__e0);
  __s                 = ::cuda::__fpmp_fast_two_sum(__s, __x_low, &__e1);
  __s                 = ::cuda::__fpmp_fast_two_sum(__s, __y_low, &__e2);
  const float __tiny  = ::cuda::__fpmp_fma_rn(__by_lo, __by_lo, ::cuda::__fpmp_fma_rn(__bx_lo, __bx_lo, __e0));
  *__a_hi             = ::cuda::__fpmp_fast_two_sum(__s, (__e1 + __e2) + __tiny, __a_lo);
  return __k;
}

/*
 * --------------------------------------------------------------------
 * Hypotenuse hypot(x, y) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * sqrt of the scaled pair above through the fp32mp2 sqrt core, scaled
 * back by 2^k.  An infinite argument gives +inf even when the other
 * one is NaN (C99 F.10.4.3); otherwise a NaN propagates.  No fp64.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_hypot(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  const uint32_t __ax = ::cuda::std::bit_cast<uint32_t>(__x_hi) & 0x7FFFFFFFu;
  const uint32_t __ay = ::cuda::std::bit_cast<uint32_t>(__y_hi) & 0x7FFFFFFFu;
  if (__ax >= 0x7F800000u || __ay >= 0x7F800000u)
  {
    const bool __inf = (__ax == 0x7F800000u) || (__ay == 0x7F800000u);
    *__res_hi        = __inf ? ::cuda::std::bit_cast<float>(0x7F800000u) : (__x_hi + __y_hi);
    *__res_lo        = 0.0f;
    return;
  }
  float __a_hi, __a_lo, __h_hi, __h_lo;
  const int __k = ::cuda::__internal_fpmp2_hypot_sum(__x_hi, __x_lo, __y_hi, __y_lo, &__a_hi, &__a_lo);
  ::cuda::__fpmp2_sqrt(__a_hi, __a_lo, &__h_hi, &__h_lo);
  const float __back = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __k) << 23));
  *__res_hi          = __h_hi * __back;
  *__res_lo          = __h_lo * __back;
}

/*
 * --------------------------------------------------------------------
 * Hypotenuse hypot(x, y) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_hypot(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  ::cuda::__fpmp2_from_double(
    ::cuda::std::hypot(::cuda::__fpmp2_to_double(__x_hi, __x_lo), ::cuda::__fpmp2_to_double(__y_hi, __y_lo)),
    __res_hi,
    __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_2A(hypot)

/*
 * ====================================================================
 * norm3d(a, b, c) - Euclidean norm of a 3-vector
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Vector norm norm3d(a, b, c) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_norm3d(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __ad   = static_cast<double>(__mp2_t(__a_hi, __a_lo));
  double __bd   = static_cast<double>(__mp2_t(__b_hi, __b_lo));
  double __cd   = static_cast<double>(__mp2_t(__c_hi, __c_lo));
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (__r = ::norm3d(__ad, __bd, __cd);),
                    (__r = ::cuda::std::sqrt(__ad * __ad + __bd * __bd + __cd * __cd);))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Vector norm norm3d(a, b, c) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_norm3d(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  double __ad = ::cuda::__fpmp2_to_double(__a_hi, __a_lo), __bd = ::cuda::__fpmp2_to_double(__b_hi, __b_lo),
         __cd = ::cuda::__fpmp2_to_double(__c_hi, __c_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::norm3d(__ad, __bd, __cd), __res_hi, __res_lo);),
    (::cuda::__fpmp2_from_double(::cuda::std::sqrt(__ad * __ad + __bd * __bd + __cd * __cd), __res_hi, __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_3A(norm3d)

/*
 * ====================================================================
 * norm4d(a, b, c, d) - Euclidean norm of a 4-vector
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Vector norm norm4d(a, b, c, d) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_norm4d(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  const float __d_hi,
  const float __d_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __ad   = static_cast<double>(__mp2_t(__a_hi, __a_lo));
  double __bd   = static_cast<double>(__mp2_t(__b_hi, __b_lo));
  double __cd   = static_cast<double>(__mp2_t(__c_hi, __c_lo));
  double __dd   = static_cast<double>(__mp2_t(__d_hi, __d_lo));
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (__r = ::norm4d(__ad, __bd, __cd, __dd);),
                    (__r = ::cuda::std::sqrt(__ad * __ad + __bd * __bd + __cd * __cd + __dd * __dd);))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Vector norm norm4d(a, b, c, d) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_norm4d(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  const double __d_hi,
  const double __d_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  double __ad = ::cuda::__fpmp2_to_double(__a_hi, __a_lo), __bd = ::cuda::__fpmp2_to_double(__b_hi, __b_lo),
         __cd = ::cuda::__fpmp2_to_double(__c_hi, __c_lo), __dd = ::cuda::__fpmp2_to_double(__d_hi, __d_lo);
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (::cuda::__fpmp2_from_double(::norm4d(__ad, __bd, __cd, __dd), __res_hi, __res_lo);),
                    (::cuda::__fpmp2_from_double(
                       ::cuda::std::sqrt(__ad * __ad + __bd * __bd + __cd * __cd + __dd * __dd), __res_hi, __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_4A(norm4d)

/*
 * ====================================================================
 * rnorm3d(a, b, c) - reciprocal Euclidean norm of a 3-vector
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Reciprocal vector norm rnorm3d(a, b, c) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_rnorm3d(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __ad   = static_cast<double>(__mp2_t(__a_hi, __a_lo));
  double __bd   = static_cast<double>(__mp2_t(__b_hi, __b_lo));
  double __cd   = static_cast<double>(__mp2_t(__c_hi, __c_lo));
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (__r = ::rnorm3d(__ad, __bd, __cd);),
                    (__r = 1.0 / ::cuda::std::sqrt(__ad * __ad + __bd * __bd + __cd * __cd);))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Reciprocal vector norm rnorm3d(a, b, c) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_rnorm3d(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  double __ad = ::cuda::__fpmp2_to_double(__a_hi, __a_lo), __bd = ::cuda::__fpmp2_to_double(__b_hi, __b_lo),
         __cd = ::cuda::__fpmp2_to_double(__c_hi, __c_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::rnorm3d(__ad, __bd, __cd), __res_hi, __res_lo);),
    (::cuda::__fpmp2_from_double(1.0 / ::cuda::std::sqrt(__ad * __ad + __bd * __bd + __cd * __cd), __res_hi, __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_3A(rnorm3d)

/*
 * ====================================================================
 * rnorm4d(a, b, c, d) - reciprocal Euclidean norm of a 4-vector
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Reciprocal vector norm rnorm4d(a, b, c, d) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_rnorm4d(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  const float __d_hi,
  const float __d_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __ad   = static_cast<double>(__mp2_t(__a_hi, __a_lo));
  double __bd   = static_cast<double>(__mp2_t(__b_hi, __b_lo));
  double __cd   = static_cast<double>(__mp2_t(__c_hi, __c_lo));
  double __dd   = static_cast<double>(__mp2_t(__d_hi, __d_lo));
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (__r = ::rnorm4d(__ad, __bd, __cd, __dd);),
                    (__r = 1.0 / ::cuda::std::sqrt(__ad * __ad + __bd * __bd + __cd * __cd + __dd * __dd);))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Reciprocal vector norm rnorm4d(a, b, c, d) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_rnorm4d(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  const double __d_hi,
  const double __d_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  double __ad = ::cuda::__fpmp2_to_double(__a_hi, __a_lo), __bd = ::cuda::__fpmp2_to_double(__b_hi, __b_lo),
         __cd = ::cuda::__fpmp2_to_double(__c_hi, __c_lo), __dd = ::cuda::__fpmp2_to_double(__d_hi, __d_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::rnorm4d(__ad, __bd, __cd, __dd), __res_hi, __res_lo);),
    (::cuda::__fpmp2_from_double(
       1.0 / ::cuda::std::sqrt(__ad * __ad + __bd * __bd + __cd * __cd + __dd * __dd), __res_hi, __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_4A(rnorm4d)

/*
 * ====================================================================
 * rhypot(x, y) - 1 / hypot(x, y)
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Reciprocal hypotenuse rhypot(x, y) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * The scaled pair of __internal_fpmp2_hypot_sum through the fp32mp2
 * rsqrt core, scaled back by 2^-k.  An infinite argument gives +0
 * even when the other one is NaN, rhypot(0, 0) = +inf.  No fp64.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_rhypot(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  const uint32_t __ax = ::cuda::std::bit_cast<uint32_t>(__x_hi) & 0x7FFFFFFFu;
  const uint32_t __ay = ::cuda::std::bit_cast<uint32_t>(__y_hi) & 0x7FFFFFFFu;
  if (__ax >= 0x7F800000u || __ay >= 0x7F800000u)
  {
    const bool __inf = (__ax == 0x7F800000u) || (__ay == 0x7F800000u);
    *__res_hi        = __inf ? 0.0f : (__x_hi + __y_hi);
    *__res_lo        = 0.0f;
    return;
  }
  if ((__ax | __ay) == 0u)
  {
    *__res_hi = ::cuda::std::bit_cast<float>(0x7F800000u);
    *__res_lo = 0.0f;
    return;
  }
  float __a_hi, __a_lo, __h_hi, __h_lo;
  const int __k = ::cuda::__internal_fpmp2_hypot_sum(__x_hi, __x_lo, __y_hi, __y_lo, &__a_hi, &__a_lo);
  ::cuda::__fpmp2_rsqrt(__a_hi, __a_lo, &__h_hi, &__h_lo);
  const float __back = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 - __k) << 23));
  *__res_hi          = __h_hi * __back;
  *__res_lo          = __h_lo * __back;
}

/*
 * --------------------------------------------------------------------
 * Reciprocal hypotenuse rhypot(x, y) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_rhypot(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(
       ::rhypot(::cuda::__fpmp2_to_double(__x_hi, __x_lo), ::cuda::__fpmp2_to_double(__y_hi, __y_lo)),
       __res_hi,
       __res_lo);),
    (::cuda::__fpmp2_from_double(
       1.0 / ::cuda::std::hypot(::cuda::__fpmp2_to_double(__x_hi, __x_lo), ::cuda::__fpmp2_to_double(__y_hi, __y_lo)),
       __res_hi,
       __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_2A(rhypot)

#endif // _CCCL_FPMP_USE_LIB
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_POW_H
