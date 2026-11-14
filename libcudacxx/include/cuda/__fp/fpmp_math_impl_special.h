//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_SPECIAL_H
#define _CUDA___FP_FPMP_MATH_IMPL_SPECIAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_special.h - fpmp2 error, gamma, Bessel and probability functions (erf*, gamma, j/y*, normcdf*)
    ==================================================================================================
*/

#include <cuda/__fp/fpmp_math_impl.h>
#include <cuda/std/numbers>
// Sibling family whose log kernel this family calls.
#include <cuda/__fp/fpmp_math_impl_exp.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * --------------------------------------------------------------------
 * Error function erf(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Error function erf(x) = -expm1(-|x|*P(|x|)) where P is a Remez
 * polynomial in |x|, and expm1 is computed via argument reduction
 * and a mixed-precision polynomial, all in fp32mp2.
 *
 * Two polynomial variants are provided, selected at compile time
 * by the _CCCL_FPMP_USE_FAST_ERF macro:
 *
 *   undefined           : uniform degree-23 Remez polynomial over
 *                         [0, 5.92], evaluated with full compensated
 *                         Horner.  Smallest SASS footprint.
 *   defined             : split-domain Remez at x* = 2.1134011
 *                         (LEFT degree 17 over [0, 2.1134011),
 *                          RIGHT degree 16 over [2.1134011, 5.92]),
 *                         each branch evaluated with compensated
 *                         Horner.  Both fits hit the same 46-bit
 *                         precision floor as the default; the
 *                         shorter polynomials trade ~+29% SASS
 *                         lines for ~+20% throughput / ~-8%
 *                         latency on coherent input distributions.
 *                         Warps straddling x* serialize both
 *                         branches, so scattered inputs reach the
 *                         default cost as an upper bound.
 * --------------------------------------------------------------------
 */
#  ifndef _CCCL_FPMP_USE_FAST_ERF
#    define _CCCL_FPMP_USE_FAST_ERF 1
#  endif

/*
 * --------------------------------------------------------------------
 * Zeroth-order Boys function F_0(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *   F_0(x) = 0.5 * sqrt(pi/x) * erf(sqrt(x))     for x > 0
 *   F_0(0) = 1
 *
 * Minimax-quality polynomials (Chebyshev interpolation, binary128
 * references, fit error below 2^-51 each), six ranges:
 *   x < 2            : degree 13 in (x - 1)
 *   2 <= x < 4       : degree 13 in (x - 3)
 *   4 <= x < 7.3     : degree 14 in (x - 5.65)
 *   7.3 <= x < 11.46 : degree 14 in (x - 9.38)
 *   11.46 <= x <= 34.38: rsqrt(x) * G(1/x), G of degree 16 in (1/x - 0.0582)
 *                      with the factor sqrt(pi)/2 absorbed
 *   x > 34.38        : asymptotic  (sqrt(pi)/2) * rsqrt(x)
 * The four polynomials in x run through the mixed Horner with four float
 * leading coefficients (their share is below 2^-30 of the value), G through
 * the compensated one; a Horner chain's error grows with its length and
 * |x|^k: the former degree-16 fit over [0, 4) in (3 - x) read 5 u^2 at
 * x = 0.18, the two halves stay under 1.6.  The former form read 43 bits
 * everywhere: one degree-19 fit over [4, 11.46] whose terms summed to six
 * times the value (a conditioning floor of 6 u^2 for pair coefficients) and
 * a float head in the first range; the fits themselves were good to
 * 0.02 u^2.  The last two ranges end in the exact-cross-term product.
 *
 * No erf, no sqrt.  Only rsqrt in the last two ranges.
 * All arithmetic is in fp32mp2 (no double-precision operations).
 * --------------------------------------------------------------------
 */
/*
// Note: _CCCL_HOST_DEVICE_API inline (non-static) in inline mode for performance.
// With -dc compilation, nvcc caps registers at ~37 for static
// __forceinline__ functions, ignoring __launch_bounds__ on the caller.
// This limits the boys kernel to 5.8x speedup instead of 12x.
// Without static, the compiler respects __launch_bounds__ and allocates
// 156 registers, enabling full ILP across the polynomial chains.
//
// In library build mode (_CCCL_FPMP_BUILD_LIB), static (internal linkage) is
// required to avoid ODR violations with the explicit float specialization in the
// _CCCL_FPMP_USE_LIB block, which would otherwise fuse into infinite recursion and
// be eliminated under device LTO. See the _CCCL_FPMP_CORE_API note in fpmp_impl.h.
*/
#  if defined(_CCCL_FPMP_BUILD_LIB)
#    define _CCCL_FPMP_INTERNAL_CUSTOM_DECL _CCCL_FPMP_CORE_API
#  else
#    define _CCCL_FPMP_INTERNAL_CUSTOM_DECL _CCCL_HOST_DEVICE_API inline
#  endif

/*
 * ====================================================================
 * erf(x) - error function
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Error function erf(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_erf(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* expm1(r) polynomial: u(r) = m2 + m3*r + m4*r^2 + ... + m11*r^9,
   * packed in ascending degree (m_c[0] = m2 = constant, m_c[9] =
   * m11 = leading).  The 4 highest-degree entries (m_c[6..9] =
   * original m8..m11) are plain float literals and run in the
   * dispatcher's FpType phase; the remaining 6 carry an ff `.lo()`.
   *
   * Note: the dispatcher transition uses `uf * r.hi() + m7`
   * (float*float + ff), whereas the previous hand-rolled step
   * was `uf * r + m7` (float * full-ff + ff).  This is the same
   * sub-ULP shift the exp refactor produced -- well below the
   * polynomial truncation noise floor. */
  constexpr __ffloat __m_c[10] = {
    __ffloat(0.50000000000000056), // [0] (= m2, constant)
    __ffloat(0.16666666666666607), // [1] (= m3)
    __ffloat(4.1666666666573884e-2), // [2] (= m4)
    __ffloat(8.3333333333771645e-3), // [3] (= m5)
    __ffloat(1.3888888932264757e-3), // [4] (= m6)
    __ffloat(1.9841269746984988e-4), // [5] (= m7, last ff term)
    /* high-order M = 4 entries: .lo() == 0 by construction */
    __ffloat(2.4801505e-5f), // [6] (= m8)
    __ffloat(2.7557382e-6f), // [7] (= m9)
    __ffloat(2.7626265e-7f), // [8] (= m10)
    __ffloat(2.5062102e-8f) // [9] (= m11, leading)
  };

  constexpr __ffloat __L2E(1.4426950408889634);
  constexpr __ffloat __ln2_hi(0.6931471805599453);

  __ffloat __x     = renormalize(__ffloat(__x_hi, __x_lo));
  bool __is_neg    = __x.hi() < 0.f;
  uint32_t __xhi   = ::cuda::std::bit_cast<uint32_t>(__x.hi()) & 0x7fffffffU;
  __ffloat __abs_a = __is_neg ? -__x : __x;

  /* |x| >= saturation_bound (~5.92) or Inf -> erf = +-1 */
  if (__xhi >= 0x40bd7da4U && __xhi <= 0x7f800000U)
  {
    *__res_hi = __is_neg ? -1.f : 1.f;
    *__res_lo = 0.f;
    return;
  }

#  if _CCCL_FPMP_USE_FAST_ERF == 1
  /* Fast variant: split-domain Remez at x* = 2.1134011.
   *   LEFT  : degree 17 (18 coeffs) over [0, 2.1134011)
   *   RIGHT : degree 16 (17 coeffs) over [2.1134011, 5.92]
   * Both fits sit at the same precision floor as the default
   * degree-23 polynomial.  Each branch keeps its own clean ILP
   * dataflow (a branchless coefficient-select variant was
   * measured to put `selp.f32` on the critical path of every
   * Horner step, cancelling the latency win).  Trades ~+29%
   * SASS for ~+20% throughput / ~-8% latency on coherent
   * workloads.
   */
  constexpr float __x_star = 2.1134011f;

  constexpr __ffloat __dc_left[18] = {
    __ffloat(1.2837916709551273e-01), // [ 0] constant
    __ffloat(6.3661977236753761e-01), // [ 1]
    __ffloat(1.0277260330382626e-01), // [ 2]
    __ffloat(-1.9128447038837399e-02), // [ 3]
    __ffloat(-2.0919443027514459e-04), // [ 4]
    __ffloat(1.6962054283924491e-03), // [ 5]
    __ffloat(-5.9012551064862781e-04), // [ 6]
    __ffloat(2.5894044204962638e-05), // [ 7]
    __ffloat(6.4414111344269855e-05), // [ 8]
    __ffloat(-2.9502940222999094e-05), // [ 9]
    __ffloat(2.9772044480981463e-06), // [10]
    __ffloat(3.4470407727555699e-06), // [11]
    __ffloat(-2.3997080766216321e-06), // [12]
    __ffloat(8.8126532430964285e-07), // [13]
    __ffloat(-2.1347246296037766e-07), // [14]
    __ffloat(3.4395369235060941e-08), // [15]
    __ffloat(-3.3767065506818252e-09), // [16]
    __ffloat(1.5374576174679341e-10) // [17] leading
  };

  constexpr __ffloat __dc_right[17] = {
    __ffloat(1.2838182329753376e-01), // [ 0] constant
    __ffloat(6.3664135493147287e-01), // [ 1]
    __ffloat(1.0262001147255973e-01), // [ 2]
    __ffloat(-1.8718159485718171e-02), // [ 3]
    __ffloat(-8.6902967978178309e-04), // [ 4]
    __ffloat(2.4246233937155400e-03), // [ 5]
    __ffloat(-1.1769573995400237e-03), // [ 6]
    __ffloat(3.7914816346061311e-04), // [ 7]
    __ffloat(-9.2599432840590657e-05), // [ 8]
    __ffloat(1.7765977912059822e-05), // [ 9]
    __ffloat(-2.6957054726382021e-06), // [10]
    __ffloat(3.2096938307796375e-07), // [11]
    __ffloat(-2.9400887323643838e-08), // [12]
    __ffloat(2.0010667763467461e-09), // [13]
    __ffloat(-9.5320838658187351e-11), // [14]
    __ffloat(2.8357961123952052e-12), // [15]
    __ffloat(-3.9648740890296208e-14) // [16] leading
  };

  __ffloat __poly;
  if (__abs_a.hi() < __x_star)
  {
    __poly = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__abs_a, __dc_left);
  }
  else
  {
    __poly = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__abs_a, __dc_right);
  }
#  else // _CCCL_FPMP_USE_FAST_ERF == 0
  /* Default: uniform degree-23 Remez polynomial over [0, 5.92],
   * evaluated with full compensated Horner (P(0) = d1).
   * Well-conditioned uniform Horner is the case where
   * compensated evaluation wins on both accuracy and SASS
   * footprint at this polynomial length. */
  constexpr __ffloat d1(0.12837916709551259);
  constexpr __ffloat d2(0.6366197723675876);
  constexpr __ffloat d3(0.10277260330144233);
  constexpr __ffloat d4(-1.9128446995328407e-2);
  constexpr __ffloat d5(-2.0919483164788562e-4);
  constexpr __ffloat d6(1.696207528729842e-3);
  constexpr __ffloat d7(-5.901318195328236e-4);
  constexpr __ffloat d8(2.5902605702646151e-5);
  constexpr __ffloat d9(6.4424832324704525e-5);
  constexpr __ffloat d10(-2.9583306728241582e-5);
  constexpr __ffloat d11(3.1800461703546548e-6);
  constexpr __ffloat d12(3.1218939658311085e-6);
  constexpr __ffloat d13(-2.0278249778025215e-6);
  constexpr __ffloat d14(5.643145203798444e-7);
  constexpr __ffloat d15(-8.299332548682465e-9);
  constexpr __ffloat d16(-6.7203270800518394e-8);
  constexpr __ffloat d17(3.5089011868220468e-8);
  constexpr __ffloat d18(-1.0909760903049583e-8);
  constexpr __ffloat d19(2.389211325400646e-9);
  constexpr __ffloat d20(-3.806599039253438e-10);
  constexpr __ffloat d21(4.3555974045566826e-11);
  constexpr __ffloat d22(-3.4079297100747907e-12);
  constexpr __ffloat d23(1.6366247078834561e-13);
  constexpr __ffloat d24(-3.642577040697121e-15);

  constexpr __ffloat dc[24] = {
    d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24};
  __ffloat __poly = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__abs_a, dc);
#  endif // _CCCL_FPMP_USE_FAST_ERF == 0

  /* arg = |x| * P(|x|) + |x| (replaces polyHi/polyLo splitting) */
  __ffloat __arg = renormalize(__poly * __abs_a + __abs_a);

  /* Compute -expm1(-arg): argument reduction */
  __ffloat __neg_arg  = -__arg;
  float __neg_arg_l2e = (__neg_arg * __L2E).hi();
  int __n             = ::cuda::__fpmp_fp2int_rn(__neg_arg_l2e);
  const float __fn    = ::cuda::__fpmp_int2fp_rn<float>(__n);
  __ffloat __r        = __neg_arg - __ln2_hi * __fn;

  /* Evaluate u(r) = m2 + m3*r + ... + m11*r^9 via the mixed-precision
   * dispatcher (4 high-order float coeffs m8..m11, 6 low-order ff
   * coeffs m2..m7).  Cheaper than a unified compensated Horner
   * because the high terms contribute below the noise floor and
   * don't need error tracking.
   */
  __ffloat __u = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 4>(__r, __m_c);

  /* expm1(r) = u*r^2 + r (no separate alo needed, r carries full precision) */
  __u = __u * __r;
  __u = __u * __r;
  __u = __u + __r;

  /* scale = 2^n, scalem1 = 1 - 2^n */
  int __en = 127 + __n;
  if (__en < 1)
  {
    __en = 1;
  }
  if (__en > 254)
  {
    __en = 254;
  }
  float __scale = ::cuda::std::bit_cast<float>(static_cast<unsigned>(__en) << 23);
  /* Exact: n <= 0, so scale <= 1 and Fast2Sum applies. */
  float __scalem1_lo;
  const float __scalem1_hi = ::cuda::__fpmp_fast_two_sum(1.f, -__scale, &__scalem1_lo);
  const __ffloat __scalem1(__scalem1_hi, __scalem1_lo);

  /* result = -expm1(-arg) = -u*scale + scalem1 */
  __ffloat __result = renormalize(-__u * __scale + __scalem1);

  /* Apply sign */
  if (__is_neg)
  {
    __result = -__result;
  }

  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
} // __internal_fpmp2_erf

/*
 * --------------------------------------------------------------------
 * Error function erf(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 * erf/erfc: binary64 unless the active fp128 backend is known to provide them, which
 * no CUDA release does yet -- see _CCCL_FPMP_FP128_QUAD_ERF in fpmp_math_impl.h. The
 * double round trip is spelled out rather than left to _CCCL_FPMP_CALL_FP64MP2_MATH so
 * that it also applies when the fp128 math fallback is on.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_erf(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
#  if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1) && _CCCL_FPMP_FP128_QUAD_ERF
  _CCCL_FPMP_CALL_FP64MP2_MATH(erf, _CCCL_FPMP_ERFQ, __x_hi, __x_lo, __res_hi, __res_lo);
#  else
  ::cuda::__fpmp2_from_double(::cuda::std::erf(::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
#  endif
}

_CCCL_FPMP_MATH_DISPATCH_1A(erf)

/*
 * ====================================================================
 * erfc(x) - complementary error function
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Pair product with exact cross terms (fp32mp2)
 * --------------------------------------------------------------------
 * The low multiply rounds its two cross terms in float: 4.0 u^2 worst
 * (3e6 random normalized pairs on the host).  With the cross terms as
 * exact products, two_sum accumulation and the small sum carried as a
 * pair into the final renormalization, the worst case is 0.5 u^2 -- the
 * rounding of the result's own low limb -- at ten more flops.  Used for
 * the product that forms a result in this family, where it sits on top
 * of a kernel's and an exp's error, and for the squares whose error a
 * Newton step would otherwise inherit.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API fp32mp2_low __internal_fpmp2_mul_acc(const fp32mp2_low __a, const fp32mp2_low __b) noexcept
{
  float __p_lo, __c1_lo, __c2_lo, __e1, __e2, __s2, __e;
  const float __p_hi = ::cuda::__fpmp_two_mult_fma(__a.hi(), __b.hi(), &__p_lo);
  const float __c1   = ::cuda::__fpmp_two_mult_fma(__a.hi(), __b.lo(), &__c1_lo);
  const float __c2   = ::cuda::__fpmp_two_mult_fma(__a.lo(), __b.hi(), &__c2_lo);
  float __s          = ::cuda::__fpmp_two_sum(__p_lo, __c1, &__e1);
  __s                = ::cuda::__fpmp_two_sum(__s, __c2, &__e2);
  __s                = ::cuda::__fpmp_two_sum(__s, ((__e1 + __e2) + (__c1_lo + __c2_lo)) + __a.lo() * __b.lo(), &__s2);
  const float __r_hi = ::cuda::__fpmp_fast_two_sum(__p_hi, __s, &__e);
  return fp32mp2_low(__r_hi, __e + __s2);
}

/*
 * --------------------------------------------------------------------
 * erf(y) for |y| < 1/2 (fp32mp2): y Q(y^2)
 * --------------------------------------------------------------------
 * Chebyshev fit of erf(x)/x in s = x^2 on [0, 1/4], degree 8 (2^-57.8,
 * erf_small_fit.py), two float leading coefficients, the mixed Horner, the
 * exact-cross-term product for the leading term and an exact final sum.
 * The small-argument paths of erfc, erfcx, normcdf and the inverse error
 * functions use it in place of the dedicated erf (a degree-17 polynomial
 * followed by expm1), which they would otherwise inline whole.  Host
 * check: 2.2 u^2 worst, 0.29 mean (the dedicated erf: 3.2 / 0.33).
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API fp32mp2_low __internal_fpmp2_erf_small(const fp32mp2_low __y) noexcept
{
  using __ffloat = fp32mp2_low;
  /* Q(s) = c0 + s R(s), c0 = 2/sqrt pi; the leading term is added exactly at
   * the end, so R's Horner roundings enter scaled by |s R| <= 0.09 */
  constexpr __ffloat __c0(0x1.20dd750429b6dp+0);
  constexpr __ffloat __r[8] = {
    __ffloat(-0x1.812746b0379b5p-2),
    __ffloat(0x1.ce2f21a03d814p-4),
    __ffloat(-0x1.b82ce30f2b28fp-6),
    __ffloat(0x1.565bcbf8e0365p-8),
    __ffloat(-0x1.c02d4f6e4921ap-11),
    __ffloat(0x1.f98db34e8872ep-14),
    __ffloat(-0x1.f224ep-17f),
    __ffloat(0x1.8b4b6p-20f)};
  const __ffloat __s = renormalize(__y * __y);
  const __ffloat __t = __y * (__s * ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 2>(__s, __r));
  return add<fpmp2_accuracy::high>(::cuda::__internal_fpmp2_mul_acc(__y, __c0), __t);
}

/*
 * --------------------------------------------------------------------
 * exp(s x^2) for a pair x, s in {-1, +1, -1/2} (fp32mp2) - exact argument
 * --------------------------------------------------------------------
 * erfc, erfcx, normcdf and the inverse error functions need exp(-x^2),
 * exp(x^2) or exp(-x^2/2) of a pair x at the kernel's full accuracy.  A
 * rounded pair s x^2 carries an absolute error of 2^-48 |s x^2|, which
 * becomes the result's relative error -- 88 u^2 at x = 9 (the old erfc's
 * 40 bits).  As in pow, the argument is never rounded: x_hi^2 and
 * 2 x_hi x_lo are exact products, x_lo^2 a float, and every piece above
 * 2^-48 |s x^2| enters the reduced argument r = s x^2 - n ln2 through
 * error-free sums (n ln2 as the three-piece constant of exp's own
 * reduction) before the shared exp kernel.  Host model against
 * binary128: 2.8 u^2 worst for |x| <= 9.3, independent of s.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_exp_sq(
  const float __x_hi, const float __x_lo, const float __s, float* __res_hi, float* __res_lo) noexcept
{
  constexpr float __inv_ln2(0x1.715476p+0f), __ln2_hi(0x1.62e4p-1f), __ln2_mid(0x1.7f7d1cp-20f),
    __ln2_lo(0x1.ef357ap-45f);
  float __p_l, __c_l;
  const float __p_h = ::cuda::__fpmp_two_mult_fma(__x_hi, __x_hi, &__p_l);
  const float __c_h = ::cuda::__fpmp_two_mult_fma(__x_hi, __x_lo + __x_lo, &__c_l);
  const float __t3  = __x_lo * __x_lo;
  const float __P_h = __p_h * __s, __P_l = __p_l * __s, __C_h = __c_h * __s, __C_l = __c_l * __s, __T3 = __t3 * __s;
  const float __X = __P_h + (__C_h + (__P_l + __T3));
  if (__X > 0x1.62e430p+6f) /* e^88.72 overflows float */
  {
    *__res_hi = ::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }
  if (__X < -0x1.9fe368p+6f) /* below 2^-150: zero */
  {
    *__res_hi = 0.0f;
    *__res_lo = 0.0f;
    return;
  }
  const int __n    = ::cuda::__fpmp_fp2int_rn(__X * __inv_ln2);
  const float __nf = static_cast<float>(__n);
  float __e1, __e2, __e3, __m_l;
  float __r         = ::cuda::__fpmp_fma_rn(-__nf, __ln2_hi, __P_h); /* exact: 16-bit head, Sterbenz */
  const float __m_h = ::cuda::__fpmp_two_mult_fma(__nf, __ln2_mid, &__m_l);
  __r               = ::cuda::__fpmp_two_sum(__r, -__m_h, &__e1);
  __r               = ::cuda::__fpmp_two_sum(__r, __P_l, &__e2);
  __r               = ::cuda::__fpmp_two_sum(__r, __C_h, &__e3);
  const float __r_l = ((__e1 + __e2) + __e3) + ((__C_l + __T3) - ::cuda::__fpmp_fma_rn(__nf, __ln2_lo, __m_l));
  float __fs, __fs_lo;
  ::cuda::__internal_fpmp2_exp_scales(__n + 254, &__fs, &__fs_lo);
  ::cuda::__internal_fpmp2_exp_core(fp32mp2_low(__r, __r_l), __fs, __fs_lo, __res_hi, __res_lo);
}

/*
 * --------------------------------------------------------------------
 * erfcx kernel for a >= 1/2 (fp32mp2): (1 + 2a) e^{a^2} erfc(a) ~ P(t)
 * --------------------------------------------------------------------
 * CUDA's erfcx form: a degree-22 polynomial P in t = (a - 4)/(a + 4), the
 * map of [0, inf) onto [-1, 1), divided by 1 + 2a.  Both divisions are the
 * refined one with the exact remainder; the generic fast division's 26 u^2
 * worst case was, with the fit's own 13 u^2 at t = -1, the old erfc's floor
 * near a = 0, which is why the callers use 1 - erf below a = 1/2.  The
 * stored fit holds 0.2 u^2 up to a = 27 and 0.5 u^2 at a = 40, where erfcx
 * switches to its asymptotic series.  Host model against binary128: 3.6 u^2
 * worst on [1/2, 40].
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_erfcx_kernel(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* P, ascending degree; the M = 7 leading entries are float literals (their
   * share at |t| <= 0.78 is below 2^-55) and run in the dispatcher's float
   * phase. */
  constexpr __ffloat __cheb[23] = {
    __ffloat(1.2329951186255526E+000), // [ 0] (= c22, constant)
    __ffloat(-1.3962111684056291E-001), // [ 1] (= c21)
    __ffloat(1.5379652102605428E-002), // [ 2] (= c20)
    __ffloat(6.8097054254735140E-002), // [ 3] (= c19)
    __ffloat(-1.0103906603555676E-001), // [ 4] (= c18)
    __ffloat(9.3732834997115544E-002), // [ 5] (= c17)
    __ffloat(-6.6330365827532434E-002), // [ 6] (= c16)
    __ffloat(3.7167515553018733E-002), // [ 7] (= c15)
    __ffloat(-1.6197733895953217E-002), // [ 8] (= c14)
    __ffloat(5.0319698792599572E-003), // [ 9] (= c13)
    __ffloat(-7.5777429182785833E-004), // [10] (= c12)
    __ffloat(-1.9925637684786154E-004), // [11] (= c11)
    __ffloat(1.5062557169571788E-004), // [12] (= c10)
    __ffloat(-2.4399558857200190E-005), // [13] (= c9)
    __ffloat(-1.1231787437600085E-005), // [14] (= c8)
    __ffloat(5.7087871844325649E-006), // [15] (= c7, last ff term)
    /* high-order M = 7 entries: .lo() == 0 by construction */
    __ffloat(3.095641e-7f), // [16] (= c6)
    __ffloat(-8.214741e-7f), // [17] (= c5)
    __ffloat(5.88067e-8f), // [18] (= c4)
    __ffloat(1.0404431e-7f), // [19] (= c3)
    __ffloat(-8.935022e-9f), // [20] (= c2)
    __ffloat(-9.723912e-9f), // [21] (= c1)
    __ffloat(-3.5602695e-10f) // [22] (= c0, leading)
  };

  const __ffloat __a(__a_hi, __a_lo);
  const __ffloat __num = sub<fpmp2_accuracy::high>(__a, 4.0f);
  const __ffloat __den = add<fpmp2_accuracy::high>(__a, 4.0f);
  float __t_hi, __t_lo;
  ::cuda::__fpmp_math_div_refined<true>(__num.hi(), __num.lo(), __den.hi(), __den.lo(), &__t_hi, &__t_lo);
  const __ffloat __t  = renormalize(__ffloat(__t_hi, __t_lo));
  const __ffloat __p  = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 7>(__t, __cheb);
  const __ffloat __d2 = add<fpmp2_accuracy::high>(__a * 2.0f, 1.0f);
  ::cuda::__fpmp_math_div_refined<true>(__p.hi(), __p.lo(), __d2.hi(), __d2.lo(), __res_hi, __res_lo);
}

/* erfc(a) = erfcx(a) exp(-a^2) for a >= 1/2: the kernel, the exact-argument
 * exp and the exact-cross-term product; zero beyond 27.5, where the exp
 * underflows.  The tail of erfc, the Newton residuals of erfcinv and
 * normcdfinv. */
_CCCL_FPMP_CORE_API fp32mp2_low __internal_fpmp2_erfc_tail(const fp32mp2_low __a) noexcept
{
  using __ffloat = fp32mp2_low;
  float __k_hi, __k_lo, __e_hi, __e_lo;
  ::cuda::__internal_fpmp2_erfcx_kernel(__a.hi(), __a.lo(), &__k_hi, &__k_lo);
  ::cuda::__internal_fpmp2_exp_sq(__a.hi(), __a.lo(), -1.0f, &__e_hi, &__e_lo);
  return ::cuda::__internal_fpmp2_mul_acc(__ffloat(__k_hi, __k_lo), __ffloat(__e_hi, __e_lo));
}

/*
 * --------------------------------------------------------------------
 * Complementary error function erfc(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *   |x| < 1/2 : 1 - x Q(x^2) (no cancellation, erfc >= 0.48 there)
 *   x >= 1/2  : erfcx(x) exp(-x^2), the tail above; zero above 27.5
 *   x <= -1/2 : 2 - erfc(|x|)
 * The former single formula read 40-41 bits: 13 u^2 of fit error plus the
 * generic divisions at the kernel's t = -1 end near x = 0, and a reduction
 * of exp(-x^2) that rounded n ln2 (36 u^2 at x = 6).  Host check against
 * binary128: 4.1 u^2 worst on [1/2, 8.3], 2.3 u^2 below.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_erfc(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  if (__x_hi != __x_hi)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_hi;
    return;
  }
  const __ffloat __x = renormalize(__ffloat(__x_hi, __x_lo));
  if (__x.hi() > 27.5f || __x.hi() < -27.5f) /* including +-inf */
  {
    *__res_hi = (__x.hi() > 0.0f) ? 0.0f : 2.0f;
    *__res_lo = 0.0f;
    return;
  }
  __ffloat __r;
  if (__x.hi() < 0.5f && __x.hi() > -0.5f)
  {
    __r = sub<fpmp2_accuracy::high>(1.0f, ::cuda::__internal_fpmp2_erf_small(__x));
  }
  else
  {
    const __ffloat __k = ::cuda::__internal_fpmp2_erfc_tail((__x.hi() < 0.0f) ? -__x : __x);
    __r                = (__x.hi() < 0.0f) ? sub<fpmp2_accuracy::high>(2.0f, __k) : __k;
  }
  *__res_hi = __r.hi();
  *__res_lo = __r.lo();
} // __internal_fpmp2_erfc

/*
 * --------------------------------------------------------------------
 * Inverse error functions (fp32mp2) - one Halley step from Giles' seed
 * --------------------------------------------------------------------
 * erfinv(x) for |x| <= 1/2 and erfcinv(q) for q in (0, 1/2] share one scheme:
 * Giles' single-precision approximation (2^-22 relative up to
 * w = -log(q (2 - q)) = 16, i.e. q >= 1.1e-7; the Abramowitz-Stegun normal
 * quantile 26.2.23, 2^-11, below that), then a Halley step on
 * f(y) = erf(y) - x or erfc(y) - q with the small erf y Q(y^2) (|y| < 1/2)
 * or the erfc tail, and f' = -+ (2/sqrt pi) e^{-y^2} from the exact-argument
 * exp: cubic
 * convergence takes 2^-22 to 2^-66.  The step is d = f/f',
 * y1 = y0 - d (1 - y0 d), Halley's y0 - d/(1 + y0 d) to the order needed.
 * Which function to invert matters: near |x| = 1 the error of erf(y) - x
 * reaches the result scaled by sqrt(pi) e^{y^2}/2, so erfinv(x) for
 * |x| > 1/2 goes through erfcinv(1 - |x|) with 1 - |x| exact, and erfcinv(q)
 * for q in [1/2, 3/2] through erfinv(1 - q).  Two more cases: the 2^-11 seed
 * gets a second, Newton, step; and below q = erfc(8) ~ 1e-29, where erfc's
 * own low limb is denormal, the equation is solved as
 * ln erfcx(y) - y^2 = ln q on the erfcx kernel (two Halley steps with
 * g' = -2/(sqrt pi erfcx), g'' = -2 y g' - g'^2).  Host model against
 * binary128: erfinv 3.6 u^2 worst, erfcinv 3.8 u^2 worst over (0, 2).
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API float __internal_fpmp2_erfinv_giles(const float __w_in) noexcept
{
  /* M. Giles, "Approximating the erfinv function", single precision: the
   * multiplier p(w) of x in erfinv(x) ~ p(w) x, w = -log(1 - x^2). */
  float __w = __w_in, __p;
  if (__w < 5.0f)
  {
    __w = __w - 2.5f;
    __p = 2.81022636e-08f;
    __p = ::cuda::__fpmp_fma_rn(__p, __w, 3.43273939e-07f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, -3.5233877e-06f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, -4.39150654e-06f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, 0.00021858087f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, -0.00125372503f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, -0.00417768164f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, 0.246640727f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, 1.50140941f);
  }
  else
  {
    __w = ::cuda::std::sqrt(__w) - 3.0f;
    __p = -0.000200214257f;
    __p = ::cuda::__fpmp_fma_rn(__p, __w, 0.000100950558f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, 0.00134934322f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, -0.00367342844f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, 0.00573950773f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, -0.0076224613f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, 0.00943887047f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, 1.00167406f);
    __p = ::cuda::__fpmp_fma_rn(__p, __w, 2.83297682f);
  }
  return __p;
}

/* d = f / f' with f' = s (2/sqrt pi) e^{-y0^2}, s = +1 for erf and -1 for erfc; returns y0 - d (1 - y0 d). */
_CCCL_FPMP_CORE_API fp32mp2_low
__internal_fpmp2_erfinv_step(const float __y0, const fp32mp2_low __f, const float __s) noexcept
{
  using __ffloat = fp32mp2_low;
  constexpr __ffloat __two_over_sqrtpi(0x1.20dd750429b6dp+0);
  float __e_hi, __e_lo;
  ::cuda::__internal_fpmp2_exp_sq(__y0, 0.0f, -1.0f, &__e_hi, &__e_lo);
  const __ffloat __gp   = renormalize(__ffloat(__e_hi, __e_lo) * __two_over_sqrtpi * __s);
  const __ffloat __d    = __f / __gp; /* d is a 2^-22 correction: the low division's 26 u^2 land at 2^-65 */
  const __ffloat __corr = __d * (1.0f - __y0 * __d);
  return renormalize(sub<fpmp2_accuracy::high>(__y0, __corr));
}

/* erfinv(x) for |x| <= 1/2. */
_CCCL_FPMP_CORE_API fp32mp2_low __internal_fpmp2_erfinv_core(const fp32mp2_low __x) noexcept
{
  using __ffloat     = fp32mp2_low;
  const float __w    = -::cuda::std::log((1.0f - __x.hi()) * (1.0f + __x.hi()));
  const float __y0   = ::cuda::__internal_fpmp2_erfinv_giles(__w) * __x.hi();
  const __ffloat __f = sub<fpmp2_accuracy::high>(::cuda::__internal_fpmp2_erf_small(__ffloat(__y0)), __x);
  return ::cuda::__internal_fpmp2_erfinv_step(__y0, __f, 1.0f);
}

/* erfcinv(q) for q below erfc(8) ~ 1e-29: ln erfcx(y) - y^2 = ln q on the erfcx kernel. */
_CCCL_FPMP_CORE_API fp32mp2_low
__internal_fpmp2_erfcinv_logspace(const fp32mp2_low __q, fp32mp2_low __y, const int __iters) noexcept
{
  using __ffloat = fp32mp2_low;
  constexpr __ffloat __two_over_sqrtpi(0x1.20dd750429b6dp+0);
  float __l_hi, __l_lo;
  ::cuda::__fpmp2_log(__q.hi(), __q.lo(), &__l_hi, &__l_lo);
  const __ffloat __lnq(__l_hi, __l_lo);
  for (int __it = 0; __it < __iters; ++__it)
  {
    float __k_hi, __k_lo;
    ::cuda::__internal_fpmp2_erfcx_kernel(__y.hi(), __y.lo(), &__k_hi, &__k_lo);
    const __ffloat __e(__k_hi, __k_lo);
    ::cuda::__fpmp2_log(__e.hi(), __e.lo(), &__l_hi, &__l_lo);
    const __ffloat __g = sub<fpmp2_accuracy::high>(
      sub<fpmp2_accuracy::high>(__ffloat(__l_hi, __l_lo), ::cuda::__internal_fpmp2_mul_acc(__y, __y)), __lnq);
    const __ffloat __gp  = -(__two_over_sqrtpi / __e); /* -2/(sqrt pi erfcx) */
    const __ffloat __den = __gp + __g * (__y + __gp * 0.5f); /* g' + g y + g g'/2: Halley */
    __y                  = renormalize(sub<fpmp2_accuracy::high>(__y, __g / __den)); /* low divisions: corrections */
  }
  return __y;
}

/* erfcinv(q) for q in (0, 1/2]: y >= 0. */
_CCCL_FPMP_CORE_API fp32mp2_low __internal_fpmp2_erfcinv_core(const fp32mp2_low __q) noexcept
{
  using __ffloat = fp32mp2_low;
  constexpr __ffloat __two_over_sqrtpi(0x1.20dd750429b6dp+0);
  const bool __far = __q.hi() < 1.1e-7f; /* w > 16: Giles' fit does not reach */
  float __y0;
  if (__far)
  {
    /* Abramowitz-Stegun 26.2.23 on the normal quantile, 2^-11. */
    const float __t = ::cuda::std::sqrt(-2.0f * (::cuda::std::log(__q.hi()) - 0.6931472f));
    const float __z = __t
                    - (2.515517f + 0.802853f * __t + 0.010328f * __t * __t)
                        / (1.0f + 1.432788f * __t + 0.189269f * __t * __t + 0.001308f * __t * __t * __t);
    __y0            = __z * 0x1.6a09e6p-1f;
  }
  else
  {
    const float __w = -::cuda::std::log(__q.hi() * (2.0f - __q.hi()));
    __y0            = ::cuda::__internal_fpmp2_erfinv_giles(__w) * (1.0f - __q.hi());
  }
  if (__y0 < 0.0f)
  {
    __y0 = 0.0f;
  }
  if (__y0 > 8.0f)
  {
    return ::cuda::__internal_fpmp2_erfcinv_logspace(__q, __ffloat(__y0), 2);
  }
  const __ffloat __e =
    (__y0 < 0.5f) ? sub<fpmp2_accuracy::high>(1.0f, ::cuda::__internal_fpmp2_erf_small(__ffloat(__y0)))
                  : ::cuda::__internal_fpmp2_erfc_tail(__ffloat(__y0));
  const __ffloat __f = sub<fpmp2_accuracy::high>(__e, __q);
  __ffloat __y1      = ::cuda::__internal_fpmp2_erfinv_step(__y0, __f, -1.0f);
  if (__far)
  {
    /* from 2^-11 Halley reached ~2^-33: one Newton step more (y1 > 5) */
    const __ffloat __f2 = sub<fpmp2_accuracy::high>(::cuda::__internal_fpmp2_erfc_tail(__y1), __q);
    float __e_hi, __e_lo;
    ::cuda::__internal_fpmp2_exp_sq(__y1.hi(), __y1.lo(), -1.0f, &__e_hi, &__e_lo);
    const __ffloat __gp = renormalize(__ffloat(__e_hi, __e_lo) * __two_over_sqrtpi * -1.0f);
    __y1                = renormalize(sub<fpmp2_accuracy::high>(__y1, __f2 / __gp));
  }
  return __y1;
}

/*
 * --------------------------------------------------------------------
 * Complementary error function erfc(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_erfc(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
#  if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1) && _CCCL_FPMP_FP128_QUAD_ERF
  _CCCL_FPMP_CALL_FP64MP2_MATH(erfc, _CCCL_FPMP_ERFCQ, __x_hi, __x_lo, __res_hi, __res_lo);
#  else
  ::cuda::__fpmp2_from_double(::cuda::std::erfc(::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
#  endif
}

_CCCL_FPMP_MATH_DISPATCH_1A(erfc)

/*
 * ====================================================================
 * boys_f0(x) - Boys function F0(x), the s-type electron repulsion integral
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Boys function boys_f0(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_INTERNAL_CUSTOM_DECL void
__internal_fpmp2_boys_f0(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  const __ffloat __a(__a_hi, __a_lo);

  if (__a_hi < 2.0f)
  {
    if (__a_hi < -94.0f)
    {
      /* F0(a) >= exp(-a) / (2 |a|) for a < 0 passes FLT_MAX near a = -94: +inf from there to -inf. */
      *__res_hi = ::cuda::std::__fp_inf<float>();
      *__res_lo = 0.0f;
      return;
    }
    /* degree 13 in x = a - 1 (fit error 2^-54.9, terms sum to 1.67 x the value) */
    constexpr __ffloat __c[14] = {
      __ffloat(0x1.7e5fbb9495fb9p-1),
      __ffloat(-0x1.840a13c65d039p-3),
      __ffloat(0x1.9ab3748d7912cp-5),
      __ffloat(-0x1.6c727b7ac13edp-7),
      __ffloat(0x1.0f0251d0e2d8bp-9),
      __ffloat(-0x1.57f9b617d1e57p-12),
      __ffloat(0x1.7b6ef679a4cd0p-15),
      __ffloat(-0x1.7171e9cf97d05p-18),
      __ffloat(0x1.41a252d8936a9p-21),
      __ffloat(-0x1.fa1ad30a26789p-25),
      __ffloat(0x1.6b07d80000000p-28f),
      __ffloat(-0x1.de93260000000p-32f),
      __ffloat(0x1.2889600000000p-35f),
      __ffloat(-0x1.4f6dfc0000000p-39f)};
    const __ffloat __x = sub<fpmp2_accuracy::high>(__a, 1.0f);
    const __ffloat __v = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 4>(__x, __c);
    *__res_hi          = __v.hi();
    *__res_lo          = __v.lo();
    return;
  }

  if (__a_hi < 4.0f)
  {
    /* degree 13 in x = a - 3 (fit error 2^-57.1, terms sum to 1.36 x the value) */
    constexpr __ffloat __c[14] = {
      __ffloat(0x1.023951b248d32p-1),
      __ffloat(-0x1.364f8131f82dfp-4),
      __ffloat(0x1.e4ab5374f74dap-7),
      __ffloat(-0x1.65408fedffaffp-9),
      __ffloat(0x1.d70cd2ae27298p-12),
      __ffloat(-0x1.133abad3188ffp-14),
      __ffloat(0x1.1e134e7f7e81cp-17),
      __ffloat(-0x1.0a6cf9faae986p-20),
      __ffloat(0x1.c039c943f9a51p-24),
      __ffloat(-0x1.572c27502b8fap-27),
      __ffloat(0x1.e1916c0000000p-31f),
      __ffloat(-0x1.37af100000000p-34f),
      __ffloat(0x1.7c3de40000000p-38f),
      __ffloat(-0x1.a899680000000p-42f)};
    const __ffloat __x = sub<fpmp2_accuracy::high>(__a, 3.0f);
    const __ffloat __v = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 4>(__x, __c);
    *__res_hi          = __v.hi();
    *__res_lo          = __v.lo();
    return;
  }

  if (__a_hi < 7.3f)
  {
    /* degree 14 in x = a - 5.65 (fit error 2^-54.4, terms sum to 1.34 x the value) */
    constexpr __ffloat __c[15] = {
      __ffloat(0x1.7d7d9d4a2a62ep-2),
      __ffloat(-0x1.0b882799925d8p-5),
      __ffloat(0x1.11e789e5c9b96p-8),
      __ffloat(-0x1.27fd993c15f8cp-11),
      __ffloat(0x1.3850277f62737p-14),
      __ffloat(-0x1.36f34e799d152p-17),
      __ffloat(0x1.1f89c278249abp-20),
      __ffloat(-0x1.ead5fe23e96d7p-24),
      __ffloat(0x1.82486586feeeep-27),
      __ffloat(-0x1.18c5734bee350p-30),
      __ffloat(0x1.7a125a0b55e40p-34),
      __ffloat(-0x1.d8d2320000000p-38f),
      __ffloat(0x1.1419720000000p-41f),
      __ffloat(-0x1.3a39960000000p-45f),
      __ffloat(0x1.414bbc0000000p-49f)};
    const __ffloat __x = sub<fpmp2_accuracy::high>(__a, __ffloat(0x1.699999999999ap+2));
    const __ffloat __v = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 4>(__x, __c);
    *__res_hi          = __v.hi();
    *__res_lo          = __v.lo();
    return;
  }

  if (__a_hi < 0x1.6ebc6ap3f) /* 11.46 */
  {
    /* degree 14 in x = a - 9.38 (fit error 2^-53.9, terms sum to 1.25 x the value) */
    constexpr __ffloat __c[15] = {
      __ffloat(0x1.284dbdbb51de8p-2),
      __ffloat(-0x1.f9464e939dfe4p-7),
      __ffloat(0x1.429d1e2912cd6p-10),
      __ffloat(-0x1.c7706599217c1p-14),
      __ffloat(0x1.4d971c9c331d2p-17),
      __ffloat(-0x1.ebfeaae3279c6p-21),
      __ffloat(0x1.65cef70e7e802p-24),
      __ffloat(-0x1.f9666cdcc2340p-28),
      __ffloat(0x1.56c416edada96p-31),
      __ffloat(-0x1.bb2956e38f651p-35),
      __ffloat(0x1.0fd27f73b0443p-38),
      __ffloat(-0x1.3b21e40000000p-42f),
      __ffloat(0x1.5a5eb60000000p-46f),
      __ffloat(-0x1.7e524a0000000p-50f),
      __ffloat(0x1.7670140000000p-54f)};
    const __ffloat __x = sub<fpmp2_accuracy::high>(__a, __ffloat(0x1.2c28f5c28f5c3p+3));
    const __ffloat __v = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 4>(__x, __c);
    *__res_hi          = __v.hi();
    *__res_lo          = __v.lo();
    return;
  }

  float __r_hi, __r_lo;
  ::cuda::__fpmp2_rsqrt(__a_hi, __a_lo, &__r_hi, &__r_lo);
  const __ffloat __r(__r_hi, __r_lo);

  if (__a_hi > 34.3816f)
  {
    if (__a_hi == ::cuda::std::__fp_inf<float>())
    {
      *__res_hi = 0.0f;
      *__res_lo = 0.0f;
      return;
    }
    constexpr __ffloat __sqrt_pi_4(0x1.c5bf891b4ef6bp-1);
    const __ffloat __result = ::cuda::__internal_fpmp2_mul_acc(__sqrt_pi_4, __r);
    *__res_hi               = __result.hi();
    *__res_lo               = __result.lo();
    return;
  }

  /* 11.46 <= a <= 34.38: F_0 = rsqrt(a) * G(u), u = 1/a = rsqrt(a)^2, G of
   * degree 16 in (u - 0.0582) with sqrt(pi)/2 absorbed (fit error 2^-51.8,
   * terms sum to 1.00 x the value): one product after the polynomial. */
  constexpr __ffloat __c[17] = {
    __ffloat(0x1.c5bf88f8cd958p-1),
    __ffloat(-0x1.476ece6c0d423p-20),
    __ffloat(-0x1.58fa6ef894efdp-13),
    __ffloat(-0x1.a3e477ca0b4d2p-7),
    __ffloat(-0x1.3f3d7b0a0a761p-1),
    __ffloat(-0x1.2c8d6f0df955dp+4),
    __ffloat(-0x1.2f53889de633fp+8),
    __ffloat(-0x1.2e7b6d8f07808p+9),
    __ffloat(0x1.c24ccbf24bfe9p+15),
    __ffloat(0x1.386f58928bd5ap+18),
    __ffloat(-0x1.8e755da18a1f5p+23),
    __ffloat(-0x1.04cd5642daa4cp+25),
    __ffloat(0x1.8682b0eede5b4p+31),
    __ffloat(-0x1.11a17c99337a2p+33),
    __ffloat(-0x1.3321a3c494deap+39),
    __ffloat(0x1.84729dcbd0788p+41),
    __ffloat(0x1.0209fd1540c00p+46)};
  const __ffloat __x      = sub<fpmp2_accuracy::high>(__r * __r, __ffloat(0x1.dc8cf3a93cf93p-5));
  const __ffloat __v      = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__x, __c);
  const __ffloat __result = ::cuda::__internal_fpmp2_mul_acc(__v, __r);
  *__res_hi               = __result.hi();
  *__res_lo               = __result.lo();
} // __internal_fpmp2_boys_f0

/*
 * --------------------------------------------------------------------
 * Boys function boys_f0(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_boys_f0(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  double __x = ::cuda::__fpmp2_to_double(__x_hi, __x_lo);
  double __r;
  if (__x < 1e-15)
  {
    __r = 1.0;
  }
  else
  {
    __r = 0.5 * ::cuda::std::sqrt(::cuda::std::__numbers<double>::__pi() / __x)
        * ::cuda::std::erf(::cuda::std::sqrt(__x));
  }
  ::cuda::__fpmp2_from_double(__r, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(boys_f0)

/*
 * ====================================================================
 * normcdfinv(x) - inverse normal cumulative distribution function
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Inverse normal CDF normcdfinv(p) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Giles' polynomials, all arithmetic in fp32mp2.  With a = 2p - 1 and
 * w = -log(4 p (1 - p)):
 *   central (w < 6.25)      : degree 22 in (w - 3.125)
 *   tail 1  (6.25 <= w < 16): degree 18 in (sqrt(w) - 3.25)
 *   tail 2  (w >= 16)       : degree 24 in (sqrt(w) - 7.25)
 * and z = a R(w), the factor sqrt(2) folded into the coefficients.  Pair
 * coefficients and the mixed Horner with 4 / 4 / 13 leading float terms:
 * the old code kept 9 to 13 float terms everywhere, which |t|^k of the
 * central fit made worth 80 u^2 at w ~ 6 (41 bits).  Sized for a Gaussian
 * sampler: the central region (98.6 % of a uniform p) reads 45-46 bits
 * (4.2 u^2 worst below w = 0.25, under 3.1 elsewhere), tail 1 46 bits,
 * tail 2 (p < 5.6e-8, one sample in 1e7) 41 bits as before -- its stored
 * fit is 10 u^2 at w = 16 and its float head adds the rest; a Newton
 * polish of the tails and a re-centered central polynomial exist in the
 * history (46 bits everywhere) and cost 2.5x the code.
 * The domain test reads both limbs: (1, -tiny) is inside.
 *
 * The fp64mp2 body uses CUDA's erfcinv on device and falls back to this
 * code on host.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_normcdfinv(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* ascending degree; sqrt(2) folded in; the trailing float entries run in
   * the mixed Horner's float phase */
  constexpr __ffloat __rc_c[23] = {
    __ffloat(0x1.2b57ec65a8009p+1),    __ffloat(0x1.5bc94440cc80dp-2),   __ffloat(-0x1.179b25740681dp-7),
    __ffloat(-0x1.12994fbb96ef7p-10),  __ffloat(0x1.14e8fc42dd165p-12),  __ffloat(-0x1.496275f62a21ep-16),
    __ffloat(-0x1.032ee3ea497a2p-19),  __ffloat(0x1.4186bc9a287fcp-21),  __ffloat(-0x1.61259c0cd68f4p-25),
    __ffloat(-0x1.8faeaa54bd6aep-28),  __ffloat(0x1.98a4e5267ae14p-30),  __ffloat(-0x1.50d39916d3907p-34),
    __ffloat(-0x1.42cf4b4d5be04p-36),  __ffloat(0x1.0613d3e006785p-38),  __ffloat(-0x1.0399b722b86c8p-43),
    __ffloat(-0x1.023cc79fc76bdp-44),  __ffloat(0x1.523424826b872p-47),  __ffloat(0x1.1190a1721ebd2p-55),
    __ffloat(-0x1.b2bdbfc6667b1p-53),  __ffloat(0x1.2315280000000p-56f), __ffloat(0x1.0c5b9e0000000p-59f),
    __ffloat(-0x1.1956c40000000p-62f), __ffloat(-0x1.856c6a0000000p-68f)}; /* central, M = 4 */

  constexpr __ffloat __rt_c[19] = {
    __ffloat(0x1.171f182e77876p+2),
    __ffloat(0x1.6bf14fa6fc320p+0),
    __ffloat(0x1.f1c95c5e817cfp-8),
    __ffloat(-0x1.5bab4c39db6c8p-8),
    __ffloat(0x1.cdd2bb90f01a6p-9),
    __ffloat(-0x1.38f1f6d4b9fc3p-9),
    __ffloat(0x1.61692295445aep-10),
    __ffloat(-0x1.073e92079632dp-11),
    __ffloat(0x1.1d167e87dcc75p-15),
    __ffloat(0x1.950a90631312fp-14),
    __ffloat(-0x1.18acbe19ac55fp-14),
    __ffloat(0x1.27ff030ba4289p-16),
    __ffloat(0x1.15740779cb33bp-18),
    __ffloat(-0x1.7cf0f10581326p-18),
    __ffloat(0x1.1d3d2c6318643p-19),
    __ffloat(0x1.bb266e0000000p-26f),
    __ffloat(-0x1.a1da160000000p-22f),
    __ffloat(0x1.13a0b20000000p-23f),
    __ffloat(0x1.ae47a40000000p-29f)}; /* tail 1, M = 4 */

  constexpr __ffloat __rt2_c[25] = {
    __ffloat(0x1.42440f29703b0p+3),   __ffloat(0x1.6d0ef171c7fb7p+0),    __ffloat(-0x1.771b604037765p-11),
    __ffloat(0x1.0eeb6b1feae03p-16),  __ffloat(0x1.97e5f4c1b789fp-18),   __ffloat(-0x1.cc0524c41a4f3p-20),
    __ffloat(0x1.627638b04ffbbp-22),  __ffloat(-0x1.dce792777a139p-25),  __ffloat(0x1.2ac8043d5e657p-27),
    __ffloat(-0x1.6731f01d03e23p-30), __ffloat(0x1.a3161d71a4670p-33),   __ffloat(-0x1.b6d5f3d8909b8p-36),
    __ffloat(0x1.a0c3280000000p-39f), __ffloat(-0x1.13267c0000000p-40f), __ffloat(0x1.37bf440000000p-42f),
    __ffloat(0x1.3d5e2c0000000p-44f), __ffloat(-0x1.504b960000000p-45f), __ffloat(-0x1.2169ac0000000p-47f),
    __ffloat(0x1.473cd60000000p-48f), __ffloat(0x1.a8ea2e0000000p-52f),  __ffloat(-0x1.6e555c0000000p-52f),
    __ffloat(0x1.97858e0000000p-59f), __ffloat(0x1.9a861e0000000p-57f),  __ffloat(-0x1.17d3e80000000p-60f),
    __ffloat(-0x1.2a9f680000000p-65f)}; /* tail 2, M = 13 */

  const __ffloat __p = renormalize(__ffloat(__x_hi, __x_lo));

  /* normcdfinv(0) = -inf, normcdfinv(1) = +inf, both limbs read */
  /* Domain: p in [0, 1]. Outside it, or NaN, the result is NaN (one range test, false for NaN);
   * exactly 0 and 1 give -inf and +inf, a pair just outside them NaN. */
  if (!(__p.hi() >= 0.0f && __p.hi() <= 1.0f))
  {
    const float __nan_val = (__p.hi() != __p.hi()) ? __p.hi() + __p.lo() : ::cuda::std::__fp_nan<float>();
    *__res_hi             = __nan_val;
    *__res_lo             = __nan_val;
    return;
  }
  if (__p.hi() == 0.0f || __p.hi() == 1.0f)
  {
    const bool __at_end  = (__p.hi() == 0.0f) ? (__p.lo() <= 0.0f) : (__p.lo() >= 0.0f);
    const bool __outside = (__p.hi() == 0.0f) ? (__p.lo() < 0.0f) : (__p.lo() > 0.0f);
    if (__at_end)
    {
      const float __inf = ::cuda::std::__fp_inf<float>();
      *__res_hi         = __outside ? ::cuda::std::__fp_nan<float>() : ((__p.hi() == 0.0f) ? -__inf : __inf);
      *__res_lo         = __outside ? *__res_hi : 0.0f;
      return;
    }
  }

  const __ffloat __a   = sub<fpmp2_accuracy::high>(__p * 2.0f, 1.0f);
  const __ffloat __omp = sub<fpmp2_accuracy::high>(1.0f, __p);
  const __ffloat __arg = renormalize(4.0f * __p * __omp);
  float __l_hi, __l_lo;
  ::cuda::__fpmp2_log(__arg.hi(), __arg.lo(), &__l_hi, &__l_lo);
  const __ffloat __w = -__ffloat(__l_hi, __l_lo);

  __ffloat __poly;
  if (__w.hi() < 6.25f)
  {
    __poly = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed_fast, 4>(__w - 3.125f, __rc_c);
  }
  else
  {
    float __s_hi, __s_lo;
    ::cuda::__fpmp2_sqrt(__w.hi(), __w.lo(), &__s_hi, &__s_lo);
    const __ffloat __sw(__s_hi, __s_lo);
    if (__w.hi() < 16.0f)
    {
      __poly = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 4>(__sw - 3.25f, __rt_c);
    }
    else
    {
      __poly = ::cuda::__fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 13>(__sw - 7.25f, __rt2_c);
    }
  }
  const __ffloat __z = ::cuda::__internal_fpmp2_mul_acc(__poly, __a);
  *__res_hi          = __z.hi();
  *__res_lo          = __z.lo();
} // __internal_fpmp2_normcdfinv

/*
 * --------------------------------------------------------------------
 * Inverse normal CDF normcdfinv(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_normcdfinv(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  double __p = ::cuda::__fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    ({
      // Hardcoded value since M_SQRT2 is not guaranteed to be defined on all platforms
      constexpr double __sqrt2_v = 1.41421356237309504880;
      ::cuda::__fpmp2_from_double(-__sqrt2_v * ::erfcinv(2.0 * __p), __res_hi, __res_lo);
    }),
    ({
      // No host erfcinv: the fp32mp2 code on p split into a float pair (46 bits)
      float __f_hi;
      float __f_lo;
      const float __p_hi = static_cast<float>(__p);
      ::cuda::__internal_fpmp2_normcdfinv(__p_hi, static_cast<float>(__p - __p_hi), &__f_hi, &__f_lo);
      *__res_hi = static_cast<double>(__f_hi) + static_cast<double>(__f_lo);
      *__res_lo = 0.0;
    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A(normcdfinv)

/*
 * ====================================================================
 * icdf(x) - Gaussian sample from a uniform integer
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Inverse CDF icdf(uint32_t) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Convert a uniform integer random value to a Gaussian fp32mp2 sample
 * via normcdfinv, with p = (x + 0.5) / 2^32.  Mirrors the input around
 * 0.5 to keep the probability argument in (0, 0.5], preserving precision
 * in the polynomial.
 */
_CCCL_FPMP_CORE_API void __fpmp2_icdf(uint32_t __x, float* __res_hi, float* __res_lo) noexcept
{
  float __sign = 1.0f;
  if (__x > 0x80000000u)
  {
    __x    = 0xFFFFFFFFu - __x;
    __sign = -1.0f;
  }
  /* p = (x + 0.5) / 2^32  in  (0, 0.5]
   * Split x into two 16-bit halves for exact fp32mp2 representation.
   */
  float __hi   = (float) (__x >> 16) * 0x1.0p-16f;
  float __lo   = ((float) (__x & 0xFFFFu) + 0.5f) * 0x1.0p-32f;
  float __p_hi = __hi + __lo;
  float __p_lo = __lo - (__p_hi - __hi);

  ::cuda::__fpmp2_normcdfinv(__p_hi, __p_lo, __res_hi, __res_lo);
  /* Clamp to +-FLT_MAX for safe Gaussian variate generation (no infinities) */
  if (*__res_hi >= 0x1.fffffep+127f)
  {
    *__res_hi = 0x1.fffffep+127f;
    *__res_lo = 0.0f;
  }
  if (*__res_hi <= -0x1.fffffep+127f)
  {
    *__res_hi = -0x1.fffffep+127f;
    *__res_lo = 0.0f;
  }
  *__res_hi *= __sign;
  *__res_lo *= __sign;
} // __fpmp2_icdf

/*
 * --------------------------------------------------------------------
 * Inverse CDF icdf(uint64_t) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * As above, from the top 48 bits: p = (x + 0.5) / 2^48.
 */
_CCCL_FPMP_CORE_API void __fpmp2_icdf(uint64_t __x, float* __res_hi, float* __res_lo) noexcept
{
  float __sign = 1.0f;
  __x >>= 16; /* keep top 48 bits (matches fp32mp2 precision) */
  if (__x > 0x800000000000ULL)
  {
    __x    = 0xFFFFFFFFFFFFULL - __x;
    __sign = -1.0f;
  }
  /* p = (x + 0.5) / 2^48  in  (0, 0.5]
   * Split 48-bit x into two 24-bit halves for exact float representation.
   */
  float __hi   = (float) (uint32_t) (__x >> 24) * 0x1.0p-24f;
  float __lo   = ((float) (uint32_t) (__x & 0xFFFFFFu) + 0.5f) * 0x1.0p-48f;
  float __p_hi = __hi + __lo;
  float __p_lo = __lo - (__p_hi - __hi);

  ::cuda::__fpmp2_normcdfinv(__p_hi, __p_lo, __res_hi, __res_lo);
  /* Clamp to +-FLT_MAX for safe Gaussian variate generation (no infinities) */
  if (*__res_hi >= 0x1.fffffep+127f)
  {
    *__res_hi = 0x1.fffffep+127f;
    *__res_lo = 0.0f;
  }
  if (*__res_hi <= -0x1.fffffep+127f)
  {
    *__res_hi = -0x1.fffffep+127f;
    *__res_lo = 0.0f;
  }
  *__res_hi *= __sign;
  *__res_lo *= __sign;
} // __fpmp2_icdf

/*
 * ====================================================================
 * lgamma(x) - natural log of the absolute value of gamma(x)
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Log gamma lgamma(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_MATH_FALLBACK_1A(lgamma)

/*
 * --------------------------------------------------------------------
 * Log gamma lgamma(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_lgamma(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fpmp2_from_double(::cuda::std::lgamma(::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(lgamma)

/*
 * ====================================================================
 * tgamma(x) - gamma function
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Gamma function tgamma(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_MATH_FALLBACK_1A(tgamma)

/*
 * --------------------------------------------------------------------
 * Gamma function tgamma(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_tgamma(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fpmp2_from_double(::cuda::std::tgamma(::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(tgamma)

/*
 * ====================================================================
 * j0(x) - Bessel function of the first kind, order 0
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Bessel j0(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 * Device only: the CUDA intrinsic carries it, the host build asserts and
 * returns 0.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_j0(
  [[maybe_unused]] const float __x_hi, [[maybe_unused]] const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__r = ::j0(static_cast<double>(__mp2_t(__x_hi, __x_lo)));), ({
                      _CCCL_ASSERT(false, "j0: no host fallback, returning 0");
                    }))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Bessel j0(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_j0(
  [[maybe_unused]] const double __x_hi,
  [[maybe_unused]] const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (::cuda::__fpmp2_from_double(::j0(::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);),
                    ({
                      _CCCL_ASSERT(false, "j0: no host fallback, returning 0");
                      *__res_hi = 0.0;
                      *__res_lo = 0.0;
                    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A(j0)

/*
 * ====================================================================
 * j1(x) - Bessel function of the first kind, order 1
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Bessel j1(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_j1(
  [[maybe_unused]] const float __x_hi, [[maybe_unused]] const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__r = ::j1(static_cast<double>(__mp2_t(__x_hi, __x_lo)));), ({
                      _CCCL_ASSERT(false, "j1: no host fallback, returning 0");
                    }))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Bessel j1(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_j1(
  [[maybe_unused]] const double __x_hi,
  [[maybe_unused]] const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (::cuda::__fpmp2_from_double(::j1(::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);),
                    ({
                      _CCCL_ASSERT(false, "j1: no host fallback, returning 0");
                      *__res_hi = 0.0;
                      *__res_lo = 0.0;
                    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A(j1)

/*
 * ====================================================================
 * y0(x) - Bessel function of the second kind, order 0
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Bessel y0(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_y0(
  [[maybe_unused]] const float __x_hi, [[maybe_unused]] const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__r = ::y0(static_cast<double>(__mp2_t(__x_hi, __x_lo)));), ({
                      _CCCL_ASSERT(false, "y0: no host fallback, returning 0");
                    }))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Bessel y0(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_y0(
  [[maybe_unused]] const double __x_hi,
  [[maybe_unused]] const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (::cuda::__fpmp2_from_double(::y0(::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);),
                    ({
                      _CCCL_ASSERT(false, "y0: no host fallback, returning 0");
                      *__res_hi = 0.0;
                      *__res_lo = 0.0;
                    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A(y0)

/*
 * ====================================================================
 * y1(x) - Bessel function of the second kind, order 1
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Bessel y1(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_y1(
  [[maybe_unused]] const float __x_hi, [[maybe_unused]] const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__r = ::y1(static_cast<double>(__mp2_t(__x_hi, __x_lo)));), ({
                      _CCCL_ASSERT(false, "y1: no host fallback, returning 0");
                    }))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Bessel y1(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_y1(
  [[maybe_unused]] const double __x_hi,
  [[maybe_unused]] const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (::cuda::__fpmp2_from_double(::y1(::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);),
                    ({
                      _CCCL_ASSERT(false, "y1: no host fallback, returning 0");
                      *__res_hi = 0.0;
                      *__res_lo = 0.0;
                    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A(y1)

/*
 * ====================================================================
 * cyl_bessel_i0(x) - modified Bessel function of the first kind, order 0
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Modified Bessel cyl_bessel_i0(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 * Device only: the CUDA intrinsic carries it, the host build asserts and
 * returns 0.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_cyl_bessel_i0(
  [[maybe_unused]] const float __x_hi, [[maybe_unused]] const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__r = ::cyl_bessel_i0(static_cast<double>(__mp2_t(__x_hi, __x_lo)));), ({
                      _CCCL_ASSERT(false, "cyl_bessel_i0: no host fallback, returning 0");
                    }))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Modified Bessel cyl_bessel_i0(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_cyl_bessel_i0(
  [[maybe_unused]] const double __x_hi,
  [[maybe_unused]] const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::cyl_bessel_i0(::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);),
    ({
      _CCCL_ASSERT(false, "cyl_bessel_i0: no host fallback, returning 0");
      *__res_hi = 0.0;
      *__res_lo = 0.0;
    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A(cyl_bessel_i0)

/*
 * ====================================================================
 * cyl_bessel_i1(x) - modified Bessel function of the first kind, order 1
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Modified Bessel cyl_bessel_i1(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_cyl_bessel_i1(
  [[maybe_unused]] const float __x_hi, [[maybe_unused]] const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__r = ::cyl_bessel_i1(static_cast<double>(__mp2_t(__x_hi, __x_lo)));), ({
                      _CCCL_ASSERT(false, "cyl_bessel_i1: no host fallback, returning 0");
                    }))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Modified Bessel cyl_bessel_i1(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_cyl_bessel_i1(
  [[maybe_unused]] const double __x_hi,
  [[maybe_unused]] const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::cyl_bessel_i1(::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);),
    ({
      _CCCL_ASSERT(false, "cyl_bessel_i1: no host fallback, returning 0");
      *__res_hi = 0.0;
      *__res_lo = 0.0;
    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A(cyl_bessel_i1)

/*
 * ====================================================================
 * jn(n, x) - Bessel function of the first kind, order n
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Bessel jn(n, x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 * Device only: the CUDA intrinsic carries it, the host build asserts and
 * returns 0.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_jn(
  [[maybe_unused]] const int __n,
  [[maybe_unused]] const float __x_hi,
  [[maybe_unused]] const float __x_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__r = ::jn(__n, static_cast<double>(__mp2_t(__x_hi, __x_lo)));), ({
                      _CCCL_ASSERT(false, "jn: no host fallback, returning 0");
                    }))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Bessel jn(n, x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_jn(
  [[maybe_unused]] int __n,
  [[maybe_unused]] const double __x_hi,
  [[maybe_unused]] const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::jn(__n, ::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);),
    ({
      _CCCL_ASSERT(false, "jn: no host fallback, returning 0");
      *__res_hi = 0.0;
      *__res_lo = 0.0;
    }))
}

_CCCL_FPMP_MATH_DISPATCH_INT_FP(jn)

/*
 * ====================================================================
 * yn(n, x) - Bessel function of the second kind, order n
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Bessel yn(n, x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_yn(
  [[maybe_unused]] const int __n,
  [[maybe_unused]] const float __x_hi,
  [[maybe_unused]] const float __x_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__r = ::yn(__n, static_cast<double>(__mp2_t(__x_hi, __x_lo)));), ({
                      _CCCL_ASSERT(false, "yn: no host fallback, returning 0");
                    }))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Bessel yn(n, x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_yn(
  [[maybe_unused]] int __n,
  [[maybe_unused]] const double __x_hi,
  [[maybe_unused]] const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::yn(__n, ::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);),
    ({
      _CCCL_ASSERT(false, "yn: no host fallback, returning 0");
      *__res_hi = 0.0;
      *__res_lo = 0.0;
    }))
}

_CCCL_FPMP_MATH_DISPATCH_INT_FP(yn)

/*
 * ====================================================================
 * normcdf(x) - normal cumulative distribution function
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Normal CDF normcdf(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *   normcdf(x) = erfc(-x/sqrt 2)/2, y = |x|/sqrt 2:
 *   y < 1/2     : (1 +- y Q(y^2))/2
 *   otherwise   : h = erfcx(y) exp(-x^2/2)/2 with the erfcx kernel and the
 *                 exact-argument exp applied to x itself (s = -1/2), so the
 *                 rounding of y never reaches the exponent; h for x < 0,
 *                 1 - h for x > 0 (h <= 0.24 there: no cancellation).
 * Zero below x = -14.5 (the result is under 2^-150), one from x = 8.4 on
 * (erfc(y) is under 2^-54 there).  Host model against
 * binary128: 5.3 u^2 worst on [-13, 9].
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_normcdf(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;
  constexpr __ffloat __inv_sqrt2(0x1.6a09e667f3bcdp-1);

  const __ffloat __x = renormalize(__ffloat(__x_hi, __x_lo));
  if (__x.hi() != __x.hi())
  {
    *__res_hi = __x.hi();
    *__res_lo = __x.hi();
    return;
  }
  if (__x.hi() < -14.5f) /* including -inf */
  {
    *__res_hi = 0.0f;
    *__res_lo = 0.0f;
    return;
  }
  if (__x.hi() >= 8.4f) /* erfc(y) < 2^-54; including +inf */
  {
    *__res_hi = 1.0f;
    *__res_lo = 0.0f;
    return;
  }
  const __ffloat __a = (__x.hi() < 0.0f) ? -__x : __x;
  const __ffloat __y = ::cuda::__internal_fpmp2_mul_acc(__a, __inv_sqrt2);
  __ffloat __r;
  if (__y.hi() < 0.5f)
  {
    const __ffloat __e = ::cuda::__internal_fpmp2_erf_small(__y);
    __r = (__x.hi() >= 0.0f) ? add<fpmp2_accuracy::high>(1.0f, __e) : sub<fpmp2_accuracy::high>(1.0f, __e);
    __r = __ffloat(0.5f * __r.hi(), 0.5f * __r.lo());
  }
  else
  {
    /* erfc(y)/2 with the kernel at y and the exp at x itself (s = -1/2): the
     * rounding of y never reaches the exponent */
    float __k_hi, __k_lo, __e_hi, __e_lo;
    ::cuda::__internal_fpmp2_erfcx_kernel(__y.hi(), __y.lo(), &__k_hi, &__k_lo);
    ::cuda::__internal_fpmp2_exp_sq(__a.hi(), __a.lo(), -0.5f, &__e_hi, &__e_lo);
    const __ffloat __h = ::cuda::__internal_fpmp2_mul_acc(__ffloat(__k_hi, __k_lo), __ffloat(__e_hi, __e_lo));
    const __ffloat __half(0.5f * __h.hi(), 0.5f * __h.lo());
    __r = (__x.hi() >= 0.0f) ? sub<fpmp2_accuracy::high>(1.0f, __half) : __half;
  }
  *__res_hi = __r.hi();
  *__res_lo = __r.lo();
}

/*
 * --------------------------------------------------------------------
 * Normal CDF normcdf(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_normcdf(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
#  if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1) && _CCCL_FPMP_FP128_QUAD_ERF
  // N(x) = erfc(-x/sqrt(2))/2 carried entirely in binary128, so the scaling and the
  // halving do not discard the precision the quad erfc just produced. 1/sqrt(2) comes
  // from the backend's own sqrt rather than a literal, which keeps this free of the
  // fp128 literal-suffix spelling differences between backends.
  const __fpmp_fp128 __xq  = ::cuda::__fpmp2_to_quad(__x_hi, __x_lo);
  const __fpmp_fp128 __two = (__fpmp_fp128) 2;
  ::cuda::__fpmp2_from_quad(_CCCL_FPMP_ERFCQ(-__xq / _CCCL_FPMP_SQRTQ(__two)) / __two, __res_hi, __res_lo);
#  else
  double __xd = ::cuda::__fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::normcdf(__xd), __res_hi, __res_lo);),
    (::cuda::__fpmp2_from_double(0.5 * ::cuda::std::erfc(-__xd * 0.70710678118654752440), __res_hi, __res_lo);))
#  endif
}

_CCCL_FPMP_MATH_DISPATCH_1A(normcdf)

// (rcbrt: dedicated fp32mp2 implementation defined above; see __fpmp2_rcbrt.)

/*
 * ====================================================================
 * erfcinv(x) - inverse complementary error function
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Inverse complementary error function erfcinv(q) (fp32mp2) - dedicated
 * --------------------------------------------------------------------
 * q in (0, 2).  q > 1 mirrors to -erfcinv(2 - q); q in [1/2, 1] goes
 * through erfinv(1 - q) (|1 - q| <= 1/2, exact), the rest through the
 * erfcinv core (Giles' seed, one Halley step on erfc; see the inverse
 * error function note above erfinv_giles).  erfcinv(0) = +inf,
 * erfcinv(2) = -inf, NaN outside [0, 2], both limbs read.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_erfcinv(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  __ffloat __q = renormalize(__ffloat(__x_hi, __x_lo));
  if (__q.hi() != __q.hi() || __q.hi() < 0.0f || (__q.hi() == 0.0f && __q.lo() < 0.0f) || __q.hi() > 2.0f
      || (__q.hi() == 2.0f && __q.lo() > 0.0f))
  {
    *__res_hi = ::cuda::std::__fp_nan<float>();
    *__res_lo = *__res_hi;
    return;
  }
  if (__q.hi() == 0.0f || (__q.hi() == 2.0f && __q.lo() == 0.0f)) /* (2, -eps) is a valid pair below 2 */
  {
    *__res_hi = (__q.hi() == 0.0f) ? ::cuda::std::__fp_inf<float>() : -::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }
  const bool __neg = __q.hi() > 1.0f;
  if (__neg)
  {
    __q = sub<fpmp2_accuracy::high>(2.0f, __q);
  }
  __ffloat __r;
  if (__q.hi() >= 0.5f)
  {
    __r = ::cuda::__internal_fpmp2_erfinv_core(sub<fpmp2_accuracy::high>(1.0f, __q));
  }
  else
  {
    __r = ::cuda::__internal_fpmp2_erfcinv_core(__q);
  }
  if (__neg)
  {
    __r = -__r;
  }
  *__res_hi = __r.hi();
  *__res_lo = __r.lo();
}

/*
 * --------------------------------------------------------------------
 * Inverse complementary error function erfcinv(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_erfcinv(
  [[maybe_unused]] const double __x_hi,
  [[maybe_unused]] const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::erfcinv(::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);),
    ({
      _CCCL_ASSERT(false, "erfcinv: no host fallback, returning 0");
      *__res_hi = 0.0;
      *__res_lo = 0.0;
    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A(erfcinv)

/*
 * ====================================================================
 * erfinv(x) - inverse error function
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Inverse error function erfinv(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * |x| <= 1/2 through the erfinv core (Giles' seed, one Halley step on
 * erf), |x| > 1/2 through erfcinv(1 - |x|) with 1 - |x| exact (see the
 * note above erfinv_giles).  erfinv(+-1) = +-inf, NaN beyond, both limbs
 * read.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_erfinv(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  const __ffloat __x = renormalize(__ffloat(__x_hi, __x_lo));
  const bool __neg   = __x.hi() < 0.0f;
  const __ffloat __a = __neg ? -__x : __x;
  if (__a.hi() != __a.hi() || __a.hi() > 1.0f || (__a.hi() == 1.0f && __a.lo() > 0.0f))
  {
    *__res_hi = ::cuda::std::__fp_nan<float>();
    *__res_lo = *__res_hi;
    return;
  }
  if (__a.hi() == 1.0f && __a.lo() == 0.0f)
  {
    *__res_hi = __neg ? -::cuda::std::__fp_inf<float>() : ::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
    return;
  }
  if (__x.hi() == 0.0f) /* +-0 */
  {
    *__res_hi = __x.hi();
    *__res_lo = 0.0f;
    return;
  }
  __ffloat __r;
  if (__a.hi() <= 0.5f)
  {
    __r = ::cuda::__internal_fpmp2_erfinv_core(__x);
  }
  else
  {
    __r = ::cuda::__internal_fpmp2_erfcinv_core(sub<fpmp2_accuracy::high>(1.0f, __a));
    if (__neg)
    {
      __r = -__r;
    }
  }
  *__res_hi = __r.hi();
  *__res_lo = __r.lo();
}

/*
 * --------------------------------------------------------------------
 * Inverse error function erfinv(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_erfinv(
  [[maybe_unused]] const double __x_hi,
  [[maybe_unused]] const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::erfinv(::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);),
    ({
      _CCCL_ASSERT(false, "erfinv: no host fallback, returning 0");
      *__res_hi = 0.0;
      *__res_lo = 0.0;
    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A(erfinv)

/*
 * ====================================================================
 * erfcx(x) - scaled complementary error function exp(x^2) * erfc(x)
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Scaled complementary error function erfcx(x) (fp32mp2) - dedicated
 * --------------------------------------------------------------------
 *   |x| < 1/2  : exp(x^2) (1 - x Q(x^2)), exact-argument exp
 *   x >= 1/2   : the erfcx kernel up to 40, beyond it the asymptotic series
 *                (1/(x sqrt pi)) sum_k (-1)^k (2k-1)!!/(2x^2)^k, 8 terms
 *                (truncation below 2^-65 at 40)
 *   x <= -1/2  : 2 exp(x^2) - erfcx(|x|); +inf below -9.3824, where
 *                2 exp(x^2) passes FLT_MAX
 * erfcx(+inf) = 0.  Host check against binary128: 3.7 u^2 worst up to 40,
 * 4.5 u^2 in the asymptotic tail (one refined division and a product).
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_erfcx(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  const __ffloat __x = renormalize(__ffloat(__x_hi, __x_lo));
  if (__x.hi() != __x.hi())
  {
    *__res_hi = __x.hi();
    *__res_lo = __x.hi();
    return;
  }
  const bool __neg   = __x.hi() < 0.0f;
  const __ffloat __a = __neg ? -__x : __x;
  __ffloat __r;
  if (__a.hi() >= 0.5f)
  {
    if (__neg)
    {
      if (__a.hi() > 9.3824f) /* including -inf */
      {
        *__res_hi = ::cuda::std::__fp_inf<float>();
        *__res_lo = 0.0f;
        return;
      }
      float __k_hi, __k_lo, __e_hi, __e_lo;
      ::cuda::__internal_fpmp2_erfcx_kernel(__a.hi(), __a.lo(), &__k_hi, &__k_lo);
      ::cuda::__internal_fpmp2_exp_sq(__a.hi(), __a.lo(), 1.0f, &__e_hi, &__e_lo);
      __r =
        renormalize(sub<fpmp2_accuracy::high>(__ffloat(__e_hi + __e_hi, __e_lo + __e_lo), __ffloat(__k_hi, __k_lo)));
    }
    else if (__a.hi() > 40.0f)
    {
      if (__a.hi() == ::cuda::std::__fp_inf<float>())
      {
        *__res_hi = 0.0f;
        *__res_lo = 0.0f;
        return;
      }
      /* 1/(x sqrt pi) by one refined division; y = 1/x^2 = pi (1/(x sqrt pi))^2
       * only needs ~36 bits (its terms are below 2^-11 of the sum) */
      constexpr __ffloat __inv_sqrtpi(0x1.20dd750429b6dp-1), __pi(0x1.921fb54442d18p+1);
      constexpr float __ac[8] = {1.0f, -0.5f, 0.75f, -1.875f, 6.5625f, -29.53125f, 162.421875f, -1055.7421875f};
      float __i_hi, __i_lo;
      ::cuda::__fpmp_math_div_refined<true>(__inv_sqrtpi.hi(), __inv_sqrtpi.lo(), __a.hi(), __a.lo(), &__i_hi, &__i_lo);
      const __ffloat __rx(__i_hi, __i_lo);
      const __ffloat __y = renormalize(__rx * __rx) * __pi;
      __ffloat __v(__ac[7]);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __k = 6; __k >= 0; --__k)
      {
        __v = __v * __y + __ffloat(__ac[__k]);
      }
      __r = ::cuda::__internal_fpmp2_mul_acc(__v, __rx);
    }
    else
    {
      float __k_hi, __k_lo;
      ::cuda::__internal_fpmp2_erfcx_kernel(__a.hi(), __a.lo(), &__k_hi, &__k_lo);
      __r = __ffloat(__k_hi, __k_lo);
    }
  }
  else
  {
    float __x_hi2, __x_lo2;
    ::cuda::__internal_fpmp2_exp_sq(__a.hi(), __a.lo(), 1.0f, &__x_hi2, &__x_lo2);
    const __ffloat __one_me = sub<fpmp2_accuracy::high>(1.0f, ::cuda::__internal_fpmp2_erf_small(__x));
    __r                     = ::cuda::__internal_fpmp2_mul_acc(__ffloat(__x_hi2, __x_lo2), __one_me);
  }
  *__res_hi = __r.hi();
  *__res_lo = __r.lo();
}

/*
 * --------------------------------------------------------------------
 * Scaled complementary error function erfcx(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_erfcx(
  [[maybe_unused]] const double __x_hi,
  [[maybe_unused]] const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::erfcx(::cuda::__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);),
    ({
      _CCCL_ASSERT(false, "erfcx: no host fallback, returning 0");
      *__res_hi = 0.0;
      *__res_lo = 0.0;
    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A(erfcx)

#endif // _CCCL_FPMP_USE_LIB
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_SPECIAL_H
