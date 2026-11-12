//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_EXP_H
#define _CUDA___FP_FPMP_MATH_IMPL_EXP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_exp.h - fpmp2 exponential and logarithmic functions (exp, log, exp2/10, expm1, log1p/2/10)
    ==================================================================================================
*/

#include <cuda/__fp/fpmp_math_impl.h>
#include <cuda/std/__floating_point/constants.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * --------------------------------------------------------------------
 * Internal helper: 2^r kernel for fp32mp2, |r| <= 0.5
 * --------------------------------------------------------------------
 * Evaluates  2^r  directly with a 13-term Taylor polynomial whose
 * coefficients absorb the ln(2) scaling:
 *
 *     2^r = exp(r * ln 2) = Sum_{k>=0}  a_k * r^k,   a_k = (ln 2)^k / k!
 *
 * The polynomial argument is the *base-2* reduced argument, so no
 * intermediate r * ln(2) multiplication sits between the reduction
 * and the polynomial.  Compared to evaluating exp(r * ln 2) via the
 * natural-log Taylor inside __fpmp2_exp, this kernel saves:
 *   - the y = r * ln 2 fp32mp2 product (1 ULP),
 *   - the trivial-reduction housekeeping inside __fpmp2_exp
 *     (the y/ln 2 round, the n*ln 2 subtraction, the 2^0 split-scale
 *     dance) -- none of which mathematically contribute when |y| is
 *     already inside the post-reduction window, but each adds 1-2
 *     ULP of rounding noise to the lo limb.
 *
 * Coefficient layout for the mixed-precision Horner dispatcher:
 *   - a1 is folded outside `poly_eval` (it's the wrong end of the
 *     polynomial for the M-split optimisation) and kept as __ffloat:
 *     a single fp32-rounded a1 would lose ~2 ULPs absolute, which
 *     directly pollutes the result lo since the final fold step is
 *     `p * r + a1` with `p * r` already at the magnitude of a1.
 *   - a2..a7 are the 5 lowest-degree coeffs evaluated in ff
 *     arithmetic by `poly_eval`'s ff tail; their .lo() bits matter
 *     because a_k * r^k stays above the fp32mp2 noise floor through
 *     k <= 7 at |r| = 0.5.
 *   - a8..a13 are the M = 6 highest-degree coeffs evaluated in
 *     float-only Horner steps (the dispatcher's "horner_mixed"
 *     inner loop); their fp32 round-off contributes <2^-46 to the
 *     final polynomial value at |r| <= 0.5.
 *
 * Truncation error at |r| = 0.5:
 *   |a_13 * r^13| <= (ln 2 / 2)^13 / 13! ~= 1.7 * 10^-16, well below
 *   the fp32mp2 ulp floor.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API fp32mp2_low __internal_fpmp2_exp2_kernel(fp32mp2_low __r) noexcept
{
  using __ffloat = fp32mp2_low;

  /* a1..a13 = (ln 2)^k / k!,  ordered low -> high degree.
   * 7 low-degree ff entries (carry .lo()) + 6 high-degree float
   * entries (M = 6) consumed by poly_eval's float-only inner loop. */
  constexpr __ffloat __exp2_c[13] = {
    __ffloat(0x1.62e42fefa39efp-1), /* [ 0] a1 = ln 2  */
    __ffloat(0x1.ebfbdff82c58ep-3), /* [ 1] a2  */
    __ffloat(0x1.c6b08d704a0bfp-5), /* [ 2] a3  */
    __ffloat(0x1.3b2ab6fba4e77p-7), /* [ 3] a4  */
    __ffloat(0x1.5d87fe78a6730p-10), /* [ 4] a5  */
    __ffloat(0x1.430912f86c786p-13), /* [ 5] a6  */
    __ffloat(0x1.ffcbfc588b0c5p-17), /* [ 6] a7  */
    /* high-degree M = 6, zero .lo() by construction */
    __ffloat(0x1.62c022p-20f), /* [ 7] a8  */
    __ffloat(0x1.b5253ep-24f), /* [ 8] a9  */
    __ffloat(0x1.e4cf52p-28f), /* [ 9] a10 */
    __ffloat(0x1.e8cac8p-32f), /* [10] a11 */
    __ffloat(0x1.c3bd66p-36f), /* [11] a12 */
    __ffloat(0x1.816194p-40f) /* [12] a13 */
  };

  /* G(r) = a1 + a2*r + a3*r^2 + ... + a13*r^12 ~= (2^r - 1)/r */
  __ffloat __p = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 6>(__r, __exp2_c);

  /* Close with the implicit a0 = 1 constant:
   *   2^r = 1 + r * G(r) */
  __p = __p * __r + 1.0f;
  return __p;
}

/*
 * --------------------------------------------------------------------
 * Internal helper: 10^r kernel for fp32mp2, |r| <= log10(2)/2 ~= 0.1505
 * --------------------------------------------------------------------
 * Evaluates 10^r directly with a 13-term Taylor polynomial whose
 * coefficients absorb the ln(10) scaling:
 *
 *     10^r = exp(r * ln 10) = Sum_{k>=0}  b_k * r^k,   b_k = (ln 10)^k / k!
 *
 * This is the natural companion to __internal_fpmp2_exp2_kernel
 * "exp10(x) = 2^n * 10^(x - n*log10(2))" reduction.
 * Because |r| <= log10(2)/2 ~= 0.151 (vs. 0.5 for the base-2 kernel),
 * the Horner chain accumulates noticeably less rounding noise even
 * though the b_k coefficients are larger (peak ratio |b_k r^k|
 * matches |a_k (0.5)^k| ~= exp_k since (ln 10 * 0.151)^k = (ln 2 *
 * 0.5)^k).
 *
 * Coefficient layout:
 *   - b1..b7 (the 7 lowest-degree terms) kept as ff: their
 *     fp32-rounding error at |r| = log10(2)/2 lifts the polynomial
 *     value above the fp32mp2 noise floor and must carry .lo() bits
 *     (with b7 in plain float the result lost two bits: one ulp of
 *     b7 ~= 2^-27 times r^7 reaches 2^-46.7 of the result).
 *   - b8..b13 (the M = 6 highest) are plain float (zero .lo());
 *     their fp32 round-off contributes < 2^-50 to the result.
 *   - Implicit b0 = 1 is folded via the final  p * r + 1  step.
 *
 * Truncation error at |r| = log10(2)/2:
 *   |b_13 * r^13| = (ln 10)^13 (log10(2)/2)^13 / 13! ~= 1.7 * 10^-16,
 *   below the fp32mp2 ulp floor.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API fp32mp2_low __internal_fpmp2_exp10_kernel(fp32mp2_low __r) noexcept
{
  using __ffloat = fp32mp2_low;

  /* b_k = (ln 10)^k / k!,  ordered low -> high degree.
   * 7 low-degree ff entries (carry .lo()) + 6 high-degree float
   * entries (M = 6) consumed by poly_eval's float-only inner loop.
   * b7 must stay float-float: at |r| = 0.1505 the float phase's
   * own rounding on b7 would reach the result at 2^-46.7. */
  constexpr __ffloat __exp10_c[13] = {
    __ffloat(0x1.26bb1bbb55516p+1), /* [ 0] b1 = ln 10  */
    __ffloat(0x1.53524c73cea6ap+1), /* [ 1] b2          */
    __ffloat(0x1.0470591de2ca6p+1), /* [ 2] b3          */
    __ffloat(0x1.2bd7609fd98c6p+0), /* [ 3] b4          */
    __ffloat(0x1.1429ffd1d4d79p-1), /* [ 4] b5          */
    __ffloat(0x1.a7ed70847c8bap-3), /* [ 5] b6          */
    __ffloat(0x1.16e4dfc333a87p-4), /* [ 6] b7          */
    /* high-degree M = 6, zero .lo() by construction */
    __ffloat(0x1.4116bp-6f), /* [ 7] b8          */
    __ffloat(0x1.4897c4p-8f), /* [ 8] b9          */
    __ffloat(0x1.2ea52cp-10f), /* [ 9] b10         */
    __ffloat(0x1.facfd6p-13f), /* [10] b11         */
    __ffloat(0x1.84fe12p-15f), /* [11] b12         */
    __ffloat(0x1.1398aep-17f) /* [12] b13         */
  };

  /* G(r) = b1 + b2*r + b3*r^2 + ... + b13*r^12 ~= (10^r - 1)/r */
  __ffloat __p = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 6>(__r, __exp10_c);

  /* Close with the implicit b0 = 1:  10^r = 1 + r * G(r) */
  __p = __p * __r + 1.0f;
  return __p;
}

/*
 * --------------------------------------------------------------------
 * Internal helper: 2^n scaling for fp32mp2 with split exponent
 * --------------------------------------------------------------------
 * Computes  result = p * 2^n  in fp32mp2 for any integer n that
 * keeps the final result in the fp32 representable range.  The 2^n
 * factor is split into two halves (2^(n/2) * 2^(n - n/2)) so neither
 * intermediate multiplier overflows or denormalizes
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API fp32mp2_low __internal_fpmp2_ldexp2(fp32mp2_low __p, int __n) noexcept
{
  const int __k = __n >> 1; /* floor-div-by-2; signed shift on negative n */
  int __ek1     = 127 + __k;
  int __ek2     = 127 + (__n - __k);
  /* Clamp split exponents into representable normal-float biased
   * range [1, 254].  When |n| is large, one half can saturate to
   * the denormal floor / overflow ceiling -- handled by the chained
   * multiply, which then sees a fully-collapsed factor at the
   * other end. */
  if (__ek1 < 1)
  {
    __ek1 = 1;
  }
  if (__ek2 < 1)
  {
    __ek2 = 1;
  }
  if (__ek1 > 254)
  {
    __ek1 = 254;
  }
  if (__ek2 > 254)
  {
    __ek2 = 254;
  }
  const float __scale_a = ::cuda::std::bit_cast<float>(static_cast<unsigned>(__ek1) << 23);
  const float __scale_b = ::cuda::std::bit_cast<float>(static_cast<unsigned>(__ek2) << 23);
  return __p * __scale_a * __scale_b;
}

/*
 * ====================================================================
 * exp(x) - natural exponential
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Internal helper: exp(r) * 2^n for fp32mp2 (the kernel of exp and pow)
 * --------------------------------------------------------------------
 * r is the reduced argument (|r| <= ln(2)/2 plus the slack of a float
 * integer estimate; the pair need not be renormalized) and the scale 2^n
 * comes split in two halves, fscale * fscale_lo, so that neither half
 * overflows or denormalizes on its own and a result in the subnormal
 * range is rounded once (gradual underflow).  exp derives the halves
 * from its shift trick, pow from its integer n; see
 * __internal_fpmp2_exp_scales.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_exp_core(
  const fp32mp2_low __r, const float __fscale, const float __fscale_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  constexpr float __c1(0x1.0p+0);
  constexpr float __c2(0x1.0p-1);

  constexpr __ffloat __exp_c[11] = {
    __ffloat(0x1.5555555555555p-3), // [ 0] (= c3,  constant of q)
    __ffloat(0x1.5555555555555p-5), // [ 1] (= c4)
    __ffloat(0x1.1111111111111p-7), // [ 2] (= c5)
    __ffloat(0x1.6c16c16c16c17p-10), // [ 3] (= c6)
    __ffloat(0x1.a01a01a01a01ap-13), // [ 4] (= c7,  last ff term)
    /* high-order M = 6 entries: .lo() == 0 by construction */
    __ffloat(0x1.a01a0p-16f), // [ 5] (= c8)
    __ffloat(0x1.71de4p-19f), // [ 6] (= c9)
    __ffloat(0x1.27e50p-22f), // [ 7] (= c10)
    __ffloat(0x1.ae646p-26f), // [ 8] (= c11)
    __ffloat(0x1.1eedap-29f), // [ 9] (= c12)
    __ffloat(0x1.6125p-33f) // [10] (= c13, leading)
  };

  // Scale the reduced argument by the low part of the scale
  __ffloat __r_scale = __r * __fscale_lo;

  // Evaluate q(r) = c3 + c4*r + c5*r^2 + ... + c13*r^10 via the
  // mixed-precision dispatcher (6 high-order terms in plain float,
  // remaining 5 in ff).
  //
  // Note: the dispatcher's transition step uses float*float + ff
  // (matching the erfc-style layout of `poly_horner_mixed`),
  // whereas the previous hand-rolled chain used float*ff + ff at
  // the c8->c7 boundary. The numerical difference is below 1 ULP
  // at the polynomial value, well inside the Taylor truncation
  // noise floor.
  __ffloat __p = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 6>(__r, __exp_c);

  // Fold in the low-degree float coefficients c1, c2 outside the
  // dispatcher (they live at the wrong end of the polynomial for
  // the M-split optimisation).
  __p = __p * __r + __c2;
  __p = __p * __r + __c1;

  __p = __p * __r_scale + __fscale_lo;

  // Scale the result by the high part of the scale
  __p               = __p * __fscale;
  __ffloat __result = renormalize(__p);

  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
} // __internal_fpmp2_exp_core

/* The two halves of 2^n as floats, from the biased exponent field
 * k = n + 254 (the sum of the two biased exponents). */
_CCCL_FPMP_CORE_API void __internal_fpmp2_exp_scales(const int32_t __k, float* __fscale, float* __fscale_lo) noexcept
{
  int32_t __scale     = __k << 23;
  uint32_t __scale_lo = static_cast<uint32_t>(__scale >> 1);
  __scale_lo &= 0x7F800000u;
  __scale -= static_cast<int32_t>(__scale_lo);
  *__fscale    = ::cuda::std::bit_cast<float>(__scale);
  *__fscale_lo = ::cuda::std::bit_cast<float>(__scale_lo);
}

/*
 * --------------------------------------------------------------------
 * Exponential function (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Compute exp(x) for float-float (fp32mp2) precision
 *
 * Algorithm:
 *   1. Argument reduction: x = n*ln(2) + r, where |r| < ln(2)/2, with ln(2) as a
 *      Cody-Waite triple so that the reduction is accurate to ~2^-62 absolute for
 *      every n (a single float-float ln(2) left an |n| * 2^-48 error that capped the
 *      result at 41 bits for |x| > 30).
 *   2. Compute exp(r) using 14-term Taylor series with Horner's method
 *   3. Scale result by 2^n using IEEE-754 bit manipulation
 *
 * Range reduction ensures that the Taylor series converges quickly since |r| < ln(2)/2 ~= 0.35.
 * With 14 terms and float-float arithmetic this reaches ~1e-14 relative error (46 of 48 bits)
 * over the whole range where the result's low limb is a normal float, i.e. exp(x) >= 2^-102.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_exp(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  // Constants as C99 hex floating-point literals - split via constexpr constructor
  constexpr float __inv_ln2(0x1.715476p+0f); // 1/ln(2)
  constexpr float __shift_bias(12582912.0f + 127.0f * 2.0f); // 127.0f*2.0f is the bias for the exponent
  // ln(2) as a Cody-Waite triple: the head carries 16 significant bits, so n * ln2_hi is
  // exact for |n| <= 128 (the whole finite range), and the reduction below is accurate to
  // about 2^-62 absolute instead of the |n| * 2^-48 a single float-float constant allows.
  constexpr float __ln2_hi(0x1.62e4000000000p-1f);
  constexpr float __ln2_mid(0x1.7f7d1c0000000p-20f);
  constexpr float __ln2_lo(0x1.ef357a0000000p-45f);

  /* Taylor series coefficients 1/k! for k = 1..13.
   *
   * The polynomial evaluated below is
   *   p(r) = c1 + c2*r + c3*r^2 + ... + c13*r^12
   * with p(r) = (exp(r) - 1) / r, so exp(r) = r*p(r) + 1.
   *
   * Layout for the mixed-precision Horner dispatcher:
   *   - c1, c2 are plain `float` and live at the LOW-degree end
   *     of p(r), which is the wrong end for the M-split.  We
   *     keep them as scalar constants and fold them in via two
   *     trailing Horner steps OUTSIDE `poly_eval`.
   *   - c3..c13 form an 11-coefficient table consumed by
   *     `poly_eval<horner_mixed, 6>`: 6 high-degree entries
   *     (c8..c13) are plain float literals, the remaining
   *     5 (c3..c7) carry an ff `.lo()` part.
   */
  // Overflow threshold for single precision: ln(FLT_MAX) ~= 88.7228 = 0x1.62e430p+6
  if (__x_hi > 0x1.62e430p+6f)
  {
    *__res_hi = ::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }

  // Below ln(2^-150) ~= -103.97 the result is under half the smallest subnormal and is zero.
  // Between there and ln(FLT_MIN) ~= -87.34 it underflows gradually through the scaling at the
  // end, which rounds the high limb onto the subnormal grid in one step while the low limb
  // vanishes -- the same behaviour as expf and as the fp64mp2 exp.
  if (__x_hi < -0x1.9fe3680000000p+6f)
  {
    *__res_hi = 0.0f;
    *__res_lo = 0.0f;
    return;
  }

  // Step 1: Argument reduction: x = n*ln(2) + r, where |r| < ln(2)/2
  float __t = __x_hi * __inv_ln2 + __shift_bias;

  // Shift the exponent by 23 bits to get the scale as fp32 value
  int32_t __scale = ::cuda::std::bit_cast<int32_t>(__t);
  __scale <<= 23;

  // Split the scale into high and low parts
  uint32_t __scale_lo = __scale >> 1;
  __scale_lo &= 0x7F800000u;
  __scale -= __scale_lo;

  // Cast the scales to fp32 values
  float __fscale    = ::cuda::std::bit_cast<float>(__scale);
  float __fscale_lo = ::cuda::std::bit_cast<float>(__scale_lo);

  // Compute the reduced argument r = x - n*ln(2) as a pair: x_hi - n*ln2_hi is exact (the
  // product is, and Sterbenz covers the difference); n*ln2_mid is split exactly and, like
  // x_lo, folded in with an error-free sum; the two sum errors and the low product are far
  // enough below r to be summed in plain float. The pair is not renormalized: its low word
  // is below half an ulp of the high one except when x lies within ~4e-6 of a multiple of
  // ln 2, and an overlapping r costs the float-float kernel below nothing.
  float __tt       = __t - __shift_bias;
  const float __r1 = ::cuda::__fpmp_fma_rn(-__tt, __ln2_hi, __x_hi);
  float __pm_lo;
  const float __pm_hi = ::cuda::__fpmp_two_mult_fma(__tt, __ln2_mid, &__pm_lo);
  float __e1, __e2;
  const float __s1   = ::cuda::__fpmp_two_sum(__r1, -__pm_hi, &__e1);
  const float __s2   = ::cuda::__fpmp_two_sum(__s1, __x_lo, &__e2);
  const float __tail = (__e1 + __e2) - ::cuda::__fpmp_fma_rn(__tt, __ln2_lo, __pm_lo);
  __ffloat __r(__s2, __tail);

  ::cuda::__internal_fpmp2_exp_core(__r, __fscale, __fscale_lo, __res_hi, __res_lo);
} // __internal_fpmp2_exp

/*
 * --------------------------------------------------------------------
 * Exponential function (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_exp(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(exp, _CCCL_FPMP_EXPQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(exp)

/*
 * ====================================================================
 * log(x) - natural logarithm
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Internal helper: log(m) and e for x = m * 2^e (fp32mp2)
 * --------------------------------------------------------------------
 * Shared by log, log1p, log2 and log10.  Splits the argument
 * x = x_hi + x_lo + x_tail into m * 2^e with m in [sqrt(1/2), sqrt(2)]
 * and returns
 *     log(m) = 2 atanh(u) = u + u * v * q(v),   u = 2 (m - 1) / (m + 1),
 *     v = u^2
 * as a float-float (not renormalized) plus the integer e.  The callers
 * fold in e * ln 2 (log, log1p), e (log2) or e * log10(2) (log10), so
 * the integer part enters exactly and the base constant only
 * multiplies the small log(m).
 *
 * x_tail is a third limb of the argument, below the rounding of x_lo.
 * log1p passes the rounding error of 1 + x there -- the information
 * whose loss limited log1p just above |x| = 1/16 to 43 bits, since
 * log1p(x) ~= x is 16x smaller than 1 + x.  The other callers pass 0
 * and the compiler folds it away.
 *
 * The division u = f / g is where the kernel used to lose two bits
 * near e = 0 (x in [sqrt(1/2), sqrt(2)], where no e * ln 2 term
 * dilutes it): the generic fast division derives its correction from
 * a quotient of the high words only, so the single float carrying the
 * correction is up to 3 * 2^-24 q in size; its own rounding, plus the
 * 1 ulp of the device's rcp.approx, reach 10-25 u^2.  Here the
 * quotient is first corrected with the FULL numerator and denominator
 * (so it is the rounding of the whole quotient), and only then the
 * exact remainder of that corrected quotient is taken: the correction
 * is below one ulp of q, and its rounding, the once-refined
 * reciprocal and the single rounding on g_lo add up to < 5 u^2 (host
 * model against binary128: 4.8 u^2 max / 0.35 u^2 mean, against
 * 26 u^2 / 2.2 u^2 for the generic division with a 1-ulp reciprocal).
 * Cost: 12 flops plus the reciprocal.
 *
 * All arithmetic in fp32mp2_low; no fp64 operations.  Denormal x is
 * pre-scaled by 2^24.  Does not handle NaN, +-0, negative or infinite
 * inputs.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API fp32mp2_low
__internal_fpmp2_log_core(const float __x_hi, const float __x_lo, const float __x_tail, int* __e_out) noexcept
{
  using __ffloat = fp32mp2_low;

  /* Minimax polynomial coefficients for the atanh series:
   * log(m) = u * (1 + v*(c1 + v*(c2 + ... + v*c8)))
   * where u = 2*(m-1)/(m+1), v = u^2
   * Coefficients: ~= 1/((2k+1)*4^k) for k = 1..8
   *
   * Packed for the mixed-precision dispatcher in ascending degree
   * (atanh_c[0] = constant c1, atanh_c[7] = leading c8).  The
   * 5 highest-degree entries (atanh_c[3..7] = c4..c8) are plain
   * float literals (.lo() == 0 by construction), so
   * poly_eval<horner_mixed, 5> evaluates them in float and
   * transitions to ff arithmetic at atanh_c[2] = c3.
   */
  constexpr __ffloat __atanh_c[8] = {
    __ffloat(0x1.5555555555554p-4), // [0] (= c1, constant of q)
    __ffloat(0x1.999999999a3c4p-7), // [1] (= c2)
    __ffloat(0x1.24924923be72dp-9), // [2] (= c3, last ff term)
    /* high-order M = 5 entries: .lo() == 0 by construction */
    __ffloat(0x1.c71c72p-12f), // [3] (= c4)
    __ffloat(0x1.745cbap-14f), // [4] (= c5)
    __ffloat(0x1.3b266ap-16f), // [5] (= c6)
    __ffloat(0x1.0ee258p-18f), // [6] (= c7)
    __ffloat(0x1.1380b4p-20f) // [7] (= c8, leading)
  };

  /* Range reduction: x = m * 2^e, m in [1, 2) first */
  float __a_hi   = __x_hi;
  float __a_lo   = __x_lo;
  float __a_tail = __x_tail;
  int __e_adj    = 0;

  /* Normalize denormals: scale by 2^24 to make the exponent field nonzero */
  uint32_t __xbits = ::cuda::std::bit_cast<uint32_t>(__a_hi);
  if ((__xbits & 0x7F800000u) == 0u)
  {
    __a_hi   = __a_hi * 0x1.0p24f;
    __a_lo   = __a_lo * 0x1.0p24f;
    __a_tail = __a_tail * 0x1.0p24f;
    __e_adj  = -24;
    __xbits  = ::cuda::std::bit_cast<uint32_t>(__a_hi);
  }

  int __e = static_cast<int>((__xbits >> 23) & 0xFFu) - 127 + __e_adj;

  /* m_hi in [1, 2) by replacing exponent field with bias 127 */
  float __m_hi = ::cuda::std::bit_cast<float>((__xbits & 0x007FFFFFu) | 0x3F800000u);

  /* Scale a_lo and a_tail by 2^(-e_orig) where e_orig = e - e_adj,
   * using split factors to stay in normal float range */
  const int __e_orig = __e - __e_adj;
  const int __e2     = __e_orig / 2;
  const float __s1   = ::cuda::std::bit_cast<float>(static_cast<uint32_t>(127 - __e2) << 23);
  const float __s2   = ::cuda::std::bit_cast<float>(static_cast<uint32_t>(127 - (__e_orig - __e2)) << 23);
  float __m_lo       = __a_lo * __s1 * __s2;
  float __m_tail     = __a_tail * __s1 * __s2;

  /* Guard against an unnormalized input pair (fp32mp2_low operands);
   * a no-op for the type's normal invariant. */
  __m_hi = ::cuda::__fpmp_fast_two_sum(__m_hi, __m_lo, &__m_lo);

  /* If m > sqrt(2), halve m and increment e (exact) */
  if (__m_hi > 0x1.6a09e6p+0f)
  {
    __m_hi   = __m_hi * 0.5f;
    __m_lo   = __m_lo * 0.5f;
    __m_tail = __m_tail * 0.5f;
    __e      = __e + 1;
  }

  /* f = m - 1 as a normalized pair: m_hi - 1 is exact (Sterbenz), and
   * Fast2Sum folds m_lo back in (|m_hi - 1| >= ulp(m_hi) >= 2 |m_lo|
   * whenever it is nonzero).  The tail joins the low word. */
  float __f_lo;
  const float __f_hi = ::cuda::__fpmp_fast_two_sum(::cuda::__fpmp_sub_rn(__m_hi, 1.0f), __m_lo, &__f_lo);
  __f_lo             = __f_lo + __m_tail;

  /* g = m + 1: the high word's rounding error is exact (g_hi - 1 is
   * representable for g_hi in [1.7, 2.42) and Sterbenz covers the
   * remaining difference), then a single rounding on the low word,
   * below u^2 relative to g. */
  const float __g_hi = ::cuda::__fpmp_add_rn(__m_hi, 1.0f);
  const float __g_lo = (__m_hi - (__g_hi - 1.0f)) + (__m_lo + __m_tail);

  /* u = 2 f / g with the refined division (see the header comment and
   * __fpmp_math_div_refined). */
  float __q1, __lo;
  ::cuda::__fpmp_math_div_refined(__f_hi, __f_lo, __g_hi, __g_lo, &__q1, &__lo);

  /* The doubling is exact; Fast2Sum yields the normalized u that the
   * kernel's float-only Horner phase relies on. */
  float __u_lo;
  const float __u_hi = ::cuda::__fpmp_fast_two_sum(__q1 + __q1, __lo + __lo, &__u_lo);
  const __ffloat __u(__u_hi, __u_lo);
  const __ffloat __v = __u * __u;

  /* Horner evaluation: q(v) = c1 + c2*v + c3*v^2 + ... + c8*v^7
   * via the mixed-precision dispatcher (5 high-order terms in
   * plain float, remaining 3 in ff). */
  __ffloat __q = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 5>(__v, __atanh_c);

  /* log(m) = u + u*v*q(v) */
  __q      = __q * __v;
  *__e_out = __e;
  return __q * __u + __u;
} // __internal_fpmp2_log_core

/*
 * --------------------------------------------------------------------
 * Internal helper: log(m) + e * ln 2 (fp32mp2), the tail of log / log1p
 * --------------------------------------------------------------------
 * e * ln 2 is formed as an exact high product plus one rounding on the
 * low part (the ln 2 pair is accurate to 2^-52.8), then added to log(m)
 * and renormalized.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_log_finish(const fp32mp2_low __log_m, const int __e, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  constexpr float __ln2_hi(0x1.62e4300000000p-1f);
  constexpr float __ln2_lo(-0x1.05c6100000000p-29f);

  const float __e_f = ::cuda::__fpmp_int2fp_rn<float>(__e);
  float __p_lo;
  const float __p_hi = ::cuda::__fpmp_two_mult_fma(__e_f, __ln2_hi, &__p_lo);
  __p_lo             = ::cuda::__fpmp_fma_rn(__e_f, __ln2_lo, __p_lo);

  const __ffloat __result = renormalize(__log_m + __ffloat(__p_hi, __p_lo));
  *__res_hi               = __result.hi();
  *__res_lo               = __result.lo();
} // __internal_fpmp2_log_finish

/*
 * --------------------------------------------------------------------
 * Natural logarithm log(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * log(x) = log(m) + e * ln 2 over the shared core above.
 * All arithmetic in fp32mp2_low; no fp64 operations.
 * Handles denormals via pre-scaling by 2^24.
 * Does not handle NaN, +-0, or negative inputs.
 * --------------------------------------------------------------------
 */
/*
 * log, log2 and log10 outside (0, +inf): +-0 -> -inf, +inf -> +inf, a negative argument or NaN -> NaN
 * (C99 Annex F). The core assumes a positive finite input and is left alone; the callers run it
 * unconditionally and select the special value afterwards, which costs an integer range test and
 * two selects, no branch. __is_special is true for +-0, negative, +-inf and NaN high limbs (a
 * subnormal positive limb is not special: the core rescales it).
 */
_CCCL_FPMP_CORE_API bool __internal_fpmp2_log_is_special(const float __x_hi) noexcept
{
  const uint32_t __xb = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  return (__xb - 1u) >= 0x7F7FFFFFu;
}
_CCCL_FPMP_CORE_API float __internal_fpmp2_log_special_value(const float __x_hi) noexcept
{
  const uint32_t __xb = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const uint32_t __rb = ((__xb & 0x7FFFFFFFu) == 0u) ? 0xFF800000u /* -inf */
                      : (__xb == 0x7F800000u) ? 0x7F800000u /* +inf */
                                               : 0x7FC00000u; /* NaN: negative, -inf or NaN */
  return ::cuda::std::bit_cast<float>(__rb);
}
_CCCL_FPMP_CORE_API void __internal_fpmp2_log_select(
  const float __x_hi, const float __r_hi, const float __r_lo, float* __res_hi, float* __res_lo) noexcept
{
  const bool __special = ::cuda::__internal_fpmp2_log_is_special(__x_hi);
  const float __sv     = ::cuda::__internal_fpmp2_log_special_value(__x_hi);
  *__res_hi            = __special ? __sv : __r_hi;
  *__res_lo            = __special ? ((__sv != __sv) ? __sv : 0.0f) : __r_lo;
}

_CCCL_FPMP_CORE_API void
__internal_fpmp2_log(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  int __e;
  const fp32mp2_low __log_m = ::cuda::__internal_fpmp2_log_core(__x_hi, __x_lo, 0.0f, &__e);
  float __r_hi, __r_lo;
  ::cuda::__internal_fpmp2_log_finish(__log_m, __e, &__r_hi, &__r_lo);
  ::cuda::__internal_fpmp2_log_select(__x_hi, __r_hi, __r_lo, __res_hi, __res_lo);
} // __internal_fpmp2_log

/*
 * --------------------------------------------------------------------
 * Natural logarithm log(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_log(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(log, _CCCL_FPMP_LOGQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(log)

/*
 * ====================================================================
 * log1p(x) - natural logarithm of 1 + x
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Natural logarithm of (1 + x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Strategy:
 *   - Small |x_hi| (< 1/16):  direct Taylor series in (x_hi, x_lo),
 *     keeping the full fp32mp2 input intact.
 *   - Otherwise:              1 + x as an exact triple s_hi + s_lo + d
 *     (two error-free sums), handed to the shared log core with d as
 *     the third limb, then log(m) + e * ln 2.
 *
 * Why a small-|x| branch is necessary:
 *   The log path runs the input through the m - 1, f / g and Horner
 *   steps, each of which leaves ~1 ulp of relative error in the *lo*
 *   limb of the intermediate.  These ulps are insignificant when the
 *   final |result| is order |x|, but as |x| -> 0 the absolute lo
 *   error stays roughly constant while |result| ~= |x| shrinks, so
 *   the relative error blows up: at |x| ~ 3e-8 the chain leaves
 *   ~24 bits of accuracy (rel_err ~ 5e-8) -- barely fp32 quality.
 *
 *   The Taylor series
 *       log1p(x) = x * (1 - x/2 + x^2/3 - x^3/4 + ...)
 *                = x + x^2 * T(x),  T(x) = -1/2 + x/3 - x^2/4 + ...
 *   never inflates rel error: x is preserved verbatim and the
 *   correction `x^2 * T` is order x^2, so its ulps cost rel_err of
 *   order x * ulp ~= negligible against |x|.  This restores full
 *   fp32mp2 precision (~46 bits) for arbitrarily small |x|.
 *
 * Why the third limb matters on the log path:
 *   an fp32mp2 pair holds 1 + x only to ~2^-48 absolute, while just
 *   above the branch point log1p(x) ~= x is 16x smaller than 1 + x;
 *   dropping the residue d of 1 + x therefore cost three bits there
 *   (43 bits).  The two error-free sums keep d exactly and the core
 *   folds it into m - 1, where it is cheap.
 *
 * Branch point 1/16 = 2^-4 keeps the polynomial narrow (covers ~6%
 * of the typical work range) so most threads stay on the log path,
 * limiting warp divergence; at |x| = 1/16 the omitted x^12 term
 * contributes 0.0625^12/14 ~= 2.6*10^-16, well below fp32mp2 ulp at
 * log1p(1/16) ~= 0.061.
 *
 * Special-case handling (mirrors libm log1p semantics):
 *   - NaN propagation (any NaN component -> NaN result).
 *   - +inf      -> +inf.
 *   - x = -1    -> -inf (1 + x = 0 exactly).
 *   - x < -1    -> NaN  (1 + x < 0).
 *   - -inf      -> NaN  (1 + (-inf) = -inf, log of negative).
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_log1p(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* NaN propagation: any NaN component -> NaN result. */
  if (__x_hi != __x_hi || __x_lo != __x_lo)
  {
    const float __nan_val = __x_hi + __x_lo;
    *__res_hi             = __nan_val;
    *__res_lo             = __nan_val;
    return;
  }

  /* +inf input -> +inf. */
  if (__x_hi == ::cuda::std::__fp_inf<float>())
  {
    *__res_hi = ::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }

  /* -inf input -> log(-inf) = NaN. */
  if (__x_hi == -::cuda::std::__fp_inf<float>())
  {
    *__res_hi = ::cuda::std::__fp_nan<float>();
    *__res_lo = ::cuda::std::__fp_nan<float>();
    return;
  }

  /* Small-|x| polynomial branch.  See header comment for the
   * rationale: bypasses the (1+x) -> log() pipeline whose
   * accumulated lo-ulp errors dominate as |x| -> 0.  Domain check
   * for x = -1 / x < -1 still applies (covered by the |x|<1/16
   * threshold trivially: any x in this range is well above -1). */
  const float __abs_hi                 = (__x_hi < 0.0f) ? -__x_hi : __x_hi;
  constexpr float __log1p_branch_point = 0.0625f; /* 1/16 = 2^-4 */
  if (__abs_hi < __log1p_branch_point)
  {
    /* T(x) = sum_{k>=0} (-1)^k * x^k / (k+2),
     *   T[0] = -1/2, T[1] = +1/3, ..., T[11] = +1/13.
     * Layout for poly_eval<horner_mixed, M=4>: bottom 8 entries
     * are full ff (their contributions stay above fp32mp2 ulp at
     * the branch point), top 4 entries are plain float (.lo == 0
     * by construction; their contributions sit below 0.5 ulp). */
    constexpr __ffloat __log1p_poly_c[12] = {
      __ffloat(-5.0e-1), /* [ 0] -1/2 (constant) */
      __ffloat(3.3333333333333333e-1), /* [ 1] +1/3 */
      __ffloat(-2.5e-1), /* [ 2] -1/4 */
      __ffloat(2.0e-1), /* [ 3] +1/5 */
      __ffloat(-1.6666666666666666e-1), /* [ 4] -1/6 */
      __ffloat(1.4285714285714285e-1), /* [ 5] +1/7 */
      __ffloat(-1.25e-1), /* [ 6] -1/8 */
      __ffloat(1.1111111111111111e-1), /* [ 7] +1/9  (last ff term) */
      /* high-order M = 4 entries: .lo() == 0 by construction */
      __ffloat(-1.0e-1f), /* [ 8] -1/10 */
      __ffloat(9.0909094e-2f), /* [ 9] +1/11 */
      __ffloat(-8.3333336e-2f), /* [10] -1/12 */
      __ffloat(7.6923080e-2f), /* [11] +1/13 (leading) */
    };

    __ffloat __x(__x_hi, __x_lo);
    __ffloat __x2     = __x * __x;
    __ffloat __t      = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 4>(__x, __log1p_poly_c);
    __ffloat __result = renormalize(__x + __x2 * __t);
    *__res_hi         = __result.hi();
    *__res_lo         = __result.lo();
    return;
  }

  /* 1 + x = s_hi + s_lo + d exactly: the high sum and the fold of x_lo
   * are error-free sums, so the residue d that an fp32mp2 pair cannot
   * hold is kept and handed to the log core as a third limb.  For
   * x_hi = -1 the first sum is zero and Fast2Sum promotes x_lo to the
   * high word, so the core always sees a normalized argument. */
  float __e_s, __d, __s_lo;
  const float __s    = ::cuda::__fpmp_two_sum(1.0f, __x_hi, &__e_s);
  const float __l    = ::cuda::__fpmp_two_sum(__e_s, __x_lo, &__d);
  const float __s_hi = ::cuda::__fpmp_fast_two_sum(__s, __l, &__s_lo);

  /* (1 + x) < 0  -> NaN. */
  if (__s_hi < 0.0f)
  {
    *__res_hi = ::cuda::std::__fp_nan<float>();
    *__res_lo = ::cuda::std::__fp_nan<float>();
    return;
  }

  /* (1 + x) == 0 (i.e., x == -1) -> log(0) = -inf.  A zero high word
   * here means the whole triple is zero. */
  if (__s_hi == 0.0f)
  {
    *__res_hi = -::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }

  int __e;
  const __ffloat __log_m = ::cuda::__internal_fpmp2_log_core(__s_hi, __s_lo, __d, &__e);
  ::cuda::__internal_fpmp2_log_finish(__log_m, __e, __res_hi, __res_lo);
} // __internal_fpmp2_log1p

/*
 * --------------------------------------------------------------------
 * Natural logarithm of (1 + x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_log1p(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(log1p, _CCCL_FPMP_LOG1PQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(log1p)

/*
 * ====================================================================
 * log2(x) - base-2 logarithm
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Base-2 logarithm log2(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Over the shared log core:
 *     log2(x) = log(m) * (1/ln 2) + e
 * with (1/ln 2) carried as an fp32mp2 constant.  The integer e enters
 * exactly, so the constant multiplies only |log(m)| <= 0.35 and its
 * ~u^2-level rounding is diluted by |e| >= 1 everywhere outside
 * [sqrt(1/2), sqrt(2)].  (The previous composition log(x) * (1/ln 2)
 * rounded the full-size product and cost a bit on every dataset.)
 *
 * Special inputs (x <= 0, NaN, +-inf) are not handled, as in
 * __internal_fpmp2_log.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_log2(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* 1/ln(2) ~= 1.4426950408889634073599... */
  constexpr __ffloat __inv_ln2(0x1.71547652b82fep+0);

  int __e;
  const __ffloat __log_m  = ::cuda::__internal_fpmp2_log_core(__x_hi, __x_lo, 0.0f, &__e);
  const __ffloat __result = renormalize(__log_m * __inv_ln2 + ::cuda::__fpmp_int2fp_rn<float>(__e));
  ::cuda::__internal_fpmp2_log_select(__x_hi, __result.hi(), __result.lo(), __res_hi, __res_lo);
} // __internal_fpmp2_log2

/*
 * --------------------------------------------------------------------
 * Base-2 logarithm log2(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_log2(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(log2, _CCCL_FPMP_LOG2Q, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(log2)

/*
 * ====================================================================
 * log10(x) - base-10 logarithm
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Base-10 logarithm log10(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Over the shared log core:
 *     log10(x) = log(m) * (1/ln 10) + e * log10(2)
 * with (1/ln 10) as an fp32mp2 constant and e * log10(2) formed as an
 * exact high product plus one rounding on the low part.  Same
 * trade-off as log2; see its header comment.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_log10(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* 1/ln(10) ~= 0.4342944819032518276511289... */
  constexpr __ffloat __inv_ln10(0x1.bcb7b1526e50ep-2);
  /* log10(2) ~= 0.30102999566398119521 as a float pair (2^-52.4) */
  constexpr float __log10_2_hi(0x1.3441360000000p-2f);
  constexpr float __log10_2_lo(-0x1.ec10c00000000p-27f);

  int __e;
  const __ffloat __log_m = ::cuda::__internal_fpmp2_log_core(__x_hi, __x_lo, 0.0f, &__e);

  const float __e_f = ::cuda::__fpmp_int2fp_rn<float>(__e);
  float __p_lo;
  const float __p_hi = ::cuda::__fpmp_two_mult_fma(__e_f, __log10_2_hi, &__p_lo);
  __p_lo             = ::cuda::__fpmp_fma_rn(__e_f, __log10_2_lo, __p_lo);

  const __ffloat __result = renormalize(__log_m * __inv_ln10 + __ffloat(__p_hi, __p_lo));
  ::cuda::__internal_fpmp2_log_select(__x_hi, __result.hi(), __result.lo(), __res_hi, __res_lo);
} // __internal_fpmp2_log10

/*
 * --------------------------------------------------------------------
 * Base-10 logarithm log10(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_log10(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(log10, _CCCL_FPMP_LOG10Q, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(log10)

/*
 * ====================================================================
 * exp2(x) - base-2 exponential
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Base-2 exponential exp2(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Do a single integer/fractional split in *base-2* units so the
 * integer power 2^n drops out exactly and the polynomial only sees a
 * small reduced argument.
 *
 *   n = round(x_hi)              [exact integer, __ffloat sub is exact]
 *   r = x - n                    [|r_hi| <= 0.5, r_lo from x_lo preserved]
 *   y = r * ln(2)                [|y| <= ln(2)/2 ~= 0.347]
 *   exp(y)                       [via dedicated fp32mp2 exp; its
 *                                 internal reduction yields n_internal
 *                                 = 0 because |y| < ln(2)/2, so the
 *                                 call collapses to a clean Taylor
 *                                 evaluation with no further reduction
 *                                 loss]
 *   result = 2^n * exp(y)        [scaled via the split-exponent helper]
 *
 * Why this beats the previous `exp(x * ln 2)` composition:
 *   That path computed y_outer = x * ln 2 (large; |y_outer| ~= 62 for
 *   |x| ~= 90) and then re-derived n_internal = round(y_outer/ln 2)
 *   inside __fpmp2_exp.  Two stacked reductions accumulate:
 *     (a) one ulp on the __ffloat multiplication x * ln 2, then
 *     (b) the cancellation in y_outer - n_internal*ln 2 magnifies
 *         that ulp because both operands are O(|y_outer|).
 *   By doing the round() directly on x_hi we get an exact integer n
 *   in one shot, the residual r = x - n is exact in the __ffloat
 *   sense, and only one (small) multiplication r * ln(2) remains
 *   before the polynomial.  Empirically the dedicated path gains
 *   2-3 bits on the `work` dataset versus the composed form.
 *
 * Overflow boundary: 2^x overflows float for x >= 128 (FLT_MAX has
 *   biased exponent 254, so 2^128 = +inf in fp32).
 * Underflow boundary: 2^-149 is the smallest positive denormal; for
 *   x <= -150 the result rounds to 0 in fp32.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_exp2(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* NaN propagation: short-circuit so the lo limb propagates too. */
  if (__x_hi != __x_hi)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_hi;
    return;
  }

  /* Overflow / underflow shortcuts. */
  if (__x_hi >= 128.0f)
  {
    *__res_hi = ::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }
  if (__x_hi <= -150.0f)
  {
    *__res_hi = 0.0f;
    *__res_lo = 0.0f;
    return;
  }

  /* Step 1: integer/fractional split directly in base-2 units. */
  const int __n     = ::cuda::__fpmp_fp2int_rn(__x_hi);
  const float __n_f = ::cuda::__fpmp_int2fp_rn<float>(__n);

  /* Step 2: r = x - n as a *normalized* pair.  x_hi - n_f is exact
   * (both are multiples of ulp(x_hi) and |x_hi - n| <= 1/2), but
   * x_lo, up to half an ulp of x_hi, can be far larger than the ulp
   * of that difference once |x| is large (2^-18 against 2^-26 at
   * |x| ~ 84).  The kernel's float-only Horner phase reads r.hi()
   * alone, so an unnormalized (x_hi - n, x_lo) pair silently drops
   * x_lo from the high-degree terms (5 bits lost above |x| ~ 32).
   * Fast2Sum is valid here: a nonzero x_hi - n_f is at least
   * ulp(x_hi) >= 2 |x_lo|; when it is zero, (0, x_lo) is exact. */
  const float __d = ::cuda::__fpmp_sub_rn(__x_hi, __n_f);
  float __r_lo;
  const float __r_hi = ::cuda::__fpmp_fast_two_sum(__d, __x_lo, &__r_lo);
  const __ffloat __r(__r_hi, __r_lo);

  /* Step 3: 2^r via the dedicated base-2 Taylor kernel (no r * ln 2
   * detour, no internal natural-log reduction). */
  const __ffloat __u = ::cuda::__internal_fpmp2_exp2_kernel(__r);

  /* Step 4: multiply by 2^n via the split-exponent helper. */
  const __ffloat __result = ::cuda::__internal_fpmp2_ldexp2(__u, __n);

  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
} // __internal_fpmp2_exp2

/*
 * --------------------------------------------------------------------
 * Base-2 exponential exp2(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_exp2(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(exp2, _CCCL_FPMP_EXP2Q, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(exp2)

/*
 * ====================================================================
 * exp10(x) - base-10 exponential
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Base-10 exponential exp10(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *   n = round(x * log2 10)          [plain float: an n off by one only
 *                                    widens |r| by log10(2)]
 *   r = x - n * log10(2)            [Cody-Waite triple, error-free sums]
 *   10^r via the base-10 Taylor kernel, |r| <= log10(2)/2 ~= 0.1505
 *   result = 2^n * 10^r             [split-exponent helper]
 *
 * The reduction mirrors the fp32mp2 exp: log10(2) = c_hi + c_mid + c_lo
 * where c_hi carries 16 significant bits, so n * c_hi is exact for
 * |n| < 256 (the finite range needs |n| <= 153) and x_hi - n * c_hi is
 * exact as well (both are multiples of min(ulp(x_hi), 2^-17) and the
 * difference is below 2^-2).  n * c_mid is split exactly, then folded
 * in with x_lo through two error-free sums; the sum errors and
 * n * c_lo are small enough for plain float.  The resulting pair is
 * accurate to ~2^-59 absolute versus the ~2^-48 a single float-float
 * constant allows, and it costs ~20 flops instead of the ~60 of an
 * fp32mp2_high accumulation.  It is not renormalized: the low word
 * exceeds half an ulp of the high one only when x lies within ~1e-6
 * of a multiple of log10(2), which costs the kernel nothing.
 *
 * Overflow boundary: 10^x overflows float at x ~= log10(FLT_MAX) =
 *   38.5318394... -- round generously to 39; the 2^n scaling
 *   overflows naturally in between.
 * Underflow boundary: 10^x < 2^-150 for x < -45.154...; results in
 *   [2^-150, FLT_MIN) underflow gradually through the two-step
 *   scaling.  Below -46 the scaling would collapse to zero anyway.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_exp10(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  if (__x_hi != __x_hi)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_hi;
    return;
  }
  if (__x_hi >= 39.0f)
  {
    *__res_hi = ::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }
  if (__x_hi <= -46.0f)
  {
    *__res_hi = 0.0f;
    *__res_lo = 0.0f;
    return;
  }

  /* log2(10) ~= 3.3219280948873623 as a single float: only the integer
   * part of x * log2(10) is used. */
  constexpr float __log2_10(0x1.a934f0p+1f);
  /* log10(2) ~= 0.30102999566398119521 as a Cody-Waite triple: the head
   * carries 16 significant bits (see the header comment). */
  constexpr float __c_hi(0x1.3440000000000p-2f);
  constexpr float __c_mid(0x1.3509f80000000p-18f);
  constexpr float __c_lo(-0x1.80433c0000000p-44f);

  /* Step 1: n = round(x * log2 10). */
  const int __n     = ::cuda::__fpmp_fp2int_rn(::cuda::__fpmp_mul_rn(__x_hi, __log2_10));
  const float __n_f = ::cuda::__fpmp_int2fp_rn<float>(__n);

  /* Step 2: r = x - n * log10(2) as a pair (see the header comment). */
  const float __r1 = ::cuda::__fpmp_fma_rn(-__n_f, __c_hi, __x_hi);
  float __pm_lo;
  const float __pm_hi = ::cuda::__fpmp_two_mult_fma(__n_f, __c_mid, &__pm_lo);
  float __e1, __e2;
  const float __s1   = ::cuda::__fpmp_two_sum(__r1, -__pm_hi, &__e1);
  const float __s2   = ::cuda::__fpmp_two_sum(__s1, __x_lo, &__e2);
  const float __tail = (__e1 + __e2) - ::cuda::__fpmp_fma_rn(__n_f, __c_lo, __pm_lo);
  const __ffloat __r(__s2, __tail);

  /* Step 3: 10^r via the dedicated base-10 Taylor kernel. */
  const __ffloat __u = ::cuda::__internal_fpmp2_exp10_kernel(__r);

  /* Step 4: scale by 2^n via the split-exponent helper. */
  const __ffloat __result = ::cuda::__internal_fpmp2_ldexp2(__u, __n);

  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
} // __internal_fpmp2_exp10

/*
 * --------------------------------------------------------------------
 * Base-10 exponential exp10(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_exp10(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
#  if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1)
  /* fp128 path: _CCCL_FPMP_EXP10Q handles every backend (libquadmath
   * powq, CUDA double widen, host long-double powl). */
  _CCCL_FPMP_CALL_FP64MP2_MATH(exp10, _CCCL_FPMP_EXP10Q, __x_hi, __x_lo, __res_hi, __res_lo);
#  else
  /* fp64 fallback: libm has no portable `exp10`; synthesize via
   * pow(10, x).  CUDA device has the intrinsic, prefer it. */
  double __xd = ::cuda::__fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (::cuda::__fpmp2_from_double(::exp10(__xd), __res_hi, __res_lo);),
                    (::cuda::__fpmp2_from_double(::cuda::std::pow(10.0, __xd), __res_hi, __res_lo);))
#  endif
}

_CCCL_FPMP_MATH_DISPATCH_1A(exp10)

/*
 * ====================================================================
 * expm1(x) - exp(x) - 1
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * exp(x) - 1, i.e. expm1(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Strategy:
 *   - Small |x_hi| (< 1/2):  direct Taylor series
 *         expm1(x) = x + x^2 * P(x),
 *         P(x) = 1/2 + x/6 + x^2/24 + ... + x^11/(13!)
 *     keeping the full fp32mp2 input intact.
 *   - Otherwise:              compute exp(x) and subtract 1 with
 *     fp32mp2 accurate sub.
 *
 * Why a small-|x| branch is necessary:
 *   The exp(x) - 1 path produces exp(x) ~= 1 + O(x) for tiny x, so
 *   the leading bits cancel in the subtraction -- the relative
 *   accuracy of the result drops to log2(1/|x|) ulps.  At |x| ~ 1
 *   the loss is negligible; below |x| ~ 1/2 we lose ~1 bit; below
 *   |x| ~ 2^-46 the result collapses to zero entirely.  The Taylor
 *   form has no cancellation: x is preserved verbatim and the x^2*P
 *   correction sits an order of magnitude below x, so its lo-ulp
 *   noise costs only ~ |x| * ulp ~= negligible relative error.
 *
 * Branch point 1/2 chosen so that the omitted x^13 term contributes
 *   0.5^13 / 13! ~= 1.96*10^-14, comfortably below fp32mp2 ulp at
 *   expm1(1/2) ~= 0.6487, while keeping the polynomial narrow enough
 *   that the warp divergence cost stays modest (the symmetric width
 *   covers ~25% of a normal-around-0 input distribution).
 *
 * Special cases:
 *   - NaN propagation (any NaN -> NaN).
 *   - +inf      -> +inf.
 *   - -inf      -> -1.
 *   - x = 0     -> 0 (Taylor branch returns it exactly).
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_expm1(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* NaN propagation: any NaN component -> NaN result. */
  if (__x_hi != __x_hi || __x_lo != __x_lo)
  {
    const float __nan_val = __x_hi + __x_lo;
    *__res_hi             = __nan_val;
    *__res_lo             = __nan_val;
    return;
  }

  /* +inf input -> +inf. */
  if (__x_hi == ::cuda::std::__fp_inf<float>())
  {
    *__res_hi = ::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }

  /* -inf input -> -1 exactly. */
  if (__x_hi == -::cuda::std::__fp_inf<float>())
  {
    *__res_hi = -1.0f;
    *__res_lo = 0.0f;
    return;
  }

  const float __abs_hi                 = (__x_hi < 0.0f) ? -__x_hi : __x_hi;
  constexpr float __expm1_branch_point = 0.5f;
  if (__abs_hi < __expm1_branch_point)
  {
    /* P(x) = sum_{k>=0} x^k / (k+2)!,
     *   P[0] = 1/2!, P[1] = 1/3!, ..., P[11] = 1/13!.
     * Layout for poly_eval<horner_mixed, M=4>: bottom 8 entries
     * are full ff (their contributions stay above fp32mp2 ulp at
     * the branch point), top 4 entries are plain float (.lo == 0
     * by construction; their contributions sit below 0.5 ulp). */
    constexpr __ffloat __expm1_poly_c[12] = {
      __ffloat(5.0000000000000000e-1), /* [ 0] 1/2!  = 1/2 */
      __ffloat(1.6666666666666666e-1), /* [ 1] 1/3!  = 1/6 */
      __ffloat(4.1666666666666664e-2), /* [ 2] 1/4!  = 1/24 */
      __ffloat(8.3333333333333332e-3), /* [ 3] 1/5!  = 1/120 */
      __ffloat(1.3888888888888889e-3), /* [ 4] 1/6!  = 1/720 */
      __ffloat(1.9841269841269841e-4), /* [ 5] 1/7!  = 1/5040 */
      __ffloat(2.4801587301587302e-5), /* [ 6] 1/8!  = 1/40320 */
      __ffloat(2.7557319223985893e-6), /* [ 7] 1/9!  (last ff term) */
      /* high-order M = 4 entries: .lo() == 0 by construction */
      __ffloat(2.7557320e-7f), /* [ 8] 1/10! */
      __ffloat(2.5052108e-8f), /* [ 9] 1/11! */
      __ffloat(2.0876756e-9f), /* [10] 1/12! */
      __ffloat(1.6059044e-10f), /* [11] 1/13! (leading) */
    };

    __ffloat __x(__x_hi, __x_lo);
    __ffloat __x2     = __x * __x;
    __ffloat __pval   = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 4>(__x, __expm1_poly_c);
    __ffloat __result = renormalize(__x + __x2 * __pval);
    *__res_hi         = __result.hi();
    *__res_lo         = __result.lo();
    return;
  }

  /* Large-|x| branch: compute exp(x) and subtract 1.0 with the
   * accurate sub variant so the lo-limb captures the cancellation
   * residual that a plain fast 2-sum would quantise away.  For
   * |x| >= 1/2 the leading term exp(x) is at least 0.6 away from 1
   * (positive side) or 0.6 below 1 (negative side), so the
   * subtraction never loses more than ~1 bit. */
  float __e_hi;
  float __e_lo;
  ::cuda::__fpmp2_exp<float>(__x_hi, __x_lo, &__e_hi, &__e_lo);

  /* exp() may already produce +inf for very large x; pass that
   * through without quietly turning it into NaN via inf - 1. */
  if (__e_hi == ::cuda::std::__fp_inf<float>())
  {
    *__res_hi = ::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }

  __ffloat __result = sub<fpmp2_accuracy::high>(__ffloat(__e_hi, __e_lo), 1.0f);
  *__res_hi         = __result.hi();
  *__res_lo         = __result.lo();
} // __internal_fpmp2_expm1

/*
 * --------------------------------------------------------------------
 * exp(x) - 1, i.e. expm1(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_expm1(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(expm1, _CCCL_FPMP_EXPM1Q, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(expm1)

#endif // _CCCL_FPMP_USE_LIB
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_EXP_H
