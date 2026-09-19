//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
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
// Sibling families whose kernels this family calls (log from exp, rcbrt from pow, fabs from manip).
#include <cuda/__fp/fpmp_math_impl_exp.h>
#include <cuda/__fp/fpmp_math_impl_manip.h>
#include <cuda/__fp/fpmp_math_impl_pow.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
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
 * Minimax polynomial approximation converted from double precision.
 * Four ranges with transformed-argument Horner polynomials:
 *   x > 34.38    : asymptotic  (sqrt(pi)/2) * rsqrt(x)
 *   x < 4        : 17-term minimax in (3 - x)
 *   4 <= x < 11.46: 20-term minimax in (6.92 - x)
 *   11.46 <= x <= 34.38: 19-term minimax in (rsqrt(x)^2 - 0.058)
 *
 * No erf, no sqrt.  Only rsqrt in ranges 1 and 4.
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
/*
 * Define _CCCL_FPMP_USE_ACCURATE_BOYS_F0 to use the accurate implementation
 * providing 43 bits of accuracy
 */
// #define _CCCL_FPMP_USE_ACCURATE_BOYS_F0

#  if defined(_CCCL_FPMP_BUILD_LIB)
#    define _CCCL_FPMP_INTERNAL_CUSTOM_DECL _CCCL_FPMP_CORE_API
#  else
#    define _CCCL_FPMP_INTERNAL_CUSTOM_DECL _CCCL_HOST_DEVICE_API inline
#  endif

#  if !defined(_CCCL_FPMP_USE_ACCURATE_BOYS_F0)
#    define _CCCL_FPMP_RENORMALIZE(v) v = renormalize(v)
#    define _CCCL_FPMP_SUB(v, x)      sub<fpmp2_accuracy::high>(v, x)
#    define _CCCL_FPMP_METHOD         fpmp2_accuracy::low
#  else
#    define _CCCL_FPMP_RENORMALIZE(v)
#    define _CCCL_FPMP_SUB(v, x) ((v) - (x))
#    define _CCCL_FPMP_METHOD    fpmp2_accuracy::def
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
    __poly = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__abs_a, __dc_left);
  }
  else
  {
    __poly = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__abs_a, __dc_right);
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
  __ffloat __poly = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__abs_a, dc);
#  endif // _CCCL_FPMP_USE_FAST_ERF == 0

  /* arg = |x| * P(|x|) + |x| (replaces polyHi/polyLo splitting) */
  __ffloat __arg = renormalize(__poly * __abs_a + __abs_a);

  /* Compute -expm1(-arg): argument reduction */
  __ffloat __neg_arg  = -__arg;
  float __neg_arg_l2e = (__neg_arg * __L2E).hi();
  int __n             = __fpmp_fp2int_rn(__neg_arg_l2e);
  __ffloat __fn       = __fpmp_int2fp_rn<float>(__n);
  __ffloat __r        = __neg_arg - __fn * __ln2_hi;

  /* Evaluate u(r) = m2 + m3*r + ... + m11*r^9 via the mixed-precision
   * dispatcher (4 high-order float coeffs m8..m11, 6 low-order ff
   * coeffs m2..m7).  Cheaper than a unified compensated Horner
   * because the high terms contribute below the noise floor and
   * don't need error tracking.
   */
  __ffloat __u = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 4>(__r, __m_c);

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
  float __scale      = ::cuda::std::bit_cast<float>(static_cast<unsigned>(__en) << 23);
  __ffloat __scalem1 = __ffloat(1.f, 0.f) - __ffloat(__scale, 0.f);

  /* result = -expm1(-arg) = -u*scale + scalem1 */
  __ffloat __result = renormalize(-__u * __ffloat(__scale, 0.f) + __scalem1);

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
  __fpmp2_from_double(::cuda::std::erf(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
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
 * Complementary error function erfc(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Computes erfc(x) = erfcx(|x|) * exp(-x^2), where erfcx is the
 * scaled complementary error function erfcx(a) = (1+2a)*exp(a^2)*erfc(a).
 *
 * Algorithm:
 *   1. Transform variable: t = (|x| - 4) / (|x| + 4)  (maps [0,inf) -> [-1,1))
 *   2. Evaluate degree-22 Chebyshev polynomial in t to approximate
 *      (1+2|x|)*exp(x^2)*erfc(|x|), then divide by (1+2|x|) to get erfcx.
 *   3. Evaluate exp(-x^2) via argument reduction x^2 = n*ln2 + r and a
 *      degree-11 polynomial for exp(r), with split exponent scaling
 *      for large |x^2| and a fma-style correction for rounding of x^2.
 *   4. Multiply erfcx * exp(-x^2) to obtain erfc(|x|).
 *   5. For negative x, apply erfc(-x) = 2 - erfc(x).
 *
 * Coefficient layout: lower-order Chebyshev/exp terms use single float
 * (negligible contribution), higher-order terms use fp32mp2 (__ffloat).
 * Saturates to 0 or 2 for |x| > 27.5.
 * All arithmetic is in fp32mp2 (no double-precision operations).
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_erfc(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  /*
   * erfc(x) = erfcx(|x|) * exp(-x^2); erfcx = (1+2*x)*exp(x^2)*erfc(x)
   * from Chebyshev approx.
   */
  using __ffloat = fp32mp2_low;

  /* erfcx polynomial (Chebyshev coefficients), ascending degree:
   *   cheb[0]  = constant term (= original c22)
   *   cheb[22] = leading coeff (= original c0)
   * The M = 7 highest-degree entries (cheb[16..22]) are encoded as
   * float literals -- their .lo() parts are zero by construction --
   * so `poly_eval<poly_method::horner_mixed, 7>` evaluates them
   * in plain float and transitions to ff arithmetic at cheb[15]. */
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

  /* exp polynomial coefficients, ascending degree:
   *   exp_c[0]  = constant term (= original ep11)
   *   exp_c[11] = leading coeff (= original ep0)
   * M = 5 highest-degree entries (exp_c[7..11]) run in float. */
  constexpr __ffloat __exp_c[12] = {
    __ffloat(1.0E+000), // [ 0] (= ep11, constant)
    __ffloat(1.0E+000), // [ 1] (= ep10)
    __ffloat(5.0000000000000122E-001), // [ 2] (= ep9)
    __ffloat(1.6666666666666477E-001), // [ 3] (= ep8)
    __ffloat(4.1666666666519754E-002), // [ 4] (= ep7)
    __ffloat(8.3333333334550432E-003), // [ 5] (= ep6)
    __ffloat(1.3888888945916380E-003), // [ 6] (= ep5, last ff term)
    /* high-order M = 5 entries: .lo() == 0 by construction */
    __ffloat(1.984127e-4f), // [ 7] (= ep4)
    __ffloat(2.480149e-5f), // [ 8] (= ep3)
    __ffloat(2.7557515e-6f), // [ 9] (= ep2)
    __ffloat(2.76309e-7f), // [10] (= ep1)
    __ffloat(2.5022323e-8f) // [11] (= ep0, leading)
  };

  constexpr __ffloat __L2E(1.4426950408889634e+0);
  constexpr __ffloat __ln2_hi(6.9314718055994529e-1);
  constexpr __ffloat __LN2_LO(2.3190468138462996e-17);

  __ffloat __x   = renormalize(__ffloat(__x_hi, __x_lo));
  bool __is_neg  = __x.hi() < 0.f;
  uint32_t __xhi = ::cuda::std::bit_cast<uint32_t>(__x.hi()) & 0x7fffffffU;
  __ffloat __a   = (__is_neg) ? -__x : __x;

  // handle x > 27.5 && <= Inf
  if ((__xhi > 0x41dc0000U) && (__xhi <= 0x7f800000U))
  {
    *__res_hi = (__is_neg) ? 2.f : 0.f;
    *__res_lo = 0.f;
    return;
  }

  /* erfcx kernel: (1+2*a)*exp(a^2)*erfc(a) on a = |x|, transform (a-4)/(a+4) */
  __ffloat __t1 = __a - __ffloat(4.0);
  __ffloat __t2 = __a + __ffloat(4.0);
  __t2          = __ffloat(1.0) / __t2;
  __ffloat __t3 = (__t1 * __t2);
  __ffloat __t4 = __t3 + __ffloat(1.0);
  __t1          = (__ffloat(-4.0) * __t4 + __a);
  __t1          = __t1 - __t3 * __a;
  __t2          = (__t2 * __t1 + __t3);

  // Chebyshev polynomial: 7 high-order terms in float, remaining 16 in ff
  __t1 = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 7>(__t2, __cheb);

  /* (1+2*a)*exp(a^2)*erfc(a) / (1+2*a) -> exp(a^2)*erfc(a) = erfcx */
  __t2 = (__ffloat(2.0) * __a + __ffloat(1.0));
  __t2 = __ffloat(1.0) / __t2;
  __t3 = __t1 * __t2;
  __t4 = __a * (__ffloat(-2.0) * __t3) + __t1;
  __t4 = (__t4 - __t3);
  __t1 = (__t4 * __t2 + __t3);

  /* erfc(x) = erfcx * exp(-x^2) */
  __ffloat __xx = renormalize(-__a * __a);

  /* i = round(xx * L2E); t = exp_mantissa(xx); t3 = accurate_scale(t, i) */
  float __prod_hi   = (__xx * __L2E).hi();
  int __i           = __fpmp_fp2int_rn(__prod_hi);
  __ffloat __t_rint = __fpmp_int2fp_rn<float>(__i);
  __ffloat __z      = renormalize(__xx - __t_rint * __ln2_hi - __t_rint * __LN2_LO);

  // exp polynomial: 5 high-order terms in float, remaining 7 in ff
  __ffloat __t = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 5>(__z, __exp_c);

  /* accurate_scale(t, i): t * 2^i in fp32mp2 (split exponent for large |i|)*/
  int __k   = __i / 2;
  int __ek  = 127 + __k;
  int __ek2 = 127 + (__i - __k);
  if (__ek < 1)
  {
    __ek = 1;
  }
  if (__ek2 < 1)
  {
    __ek2 = 1;
  }

  float __scale_lo      = ::cuda::std::bit_cast<float>(static_cast<unsigned>(__ek) << 23);
  float __scale_hi      = ::cuda::std::bit_cast<float>(static_cast<unsigned>(__ek2) << 23);
  __ffloat __exp_scaled = __ffloat(__t.hi() * __scale_lo * __scale_hi, __t.lo() * __scale_lo * __scale_hi);

  /* Correction: exp(-x^2) = exp_scaled * (1 + (-x^2 - xx)) same as double fma(t3, -x*x - xx, t3) */
  __ffloat __remainder = renormalize(-__a * __a - __xx);
  __ffloat __exp_xx    = __exp_scaled * __remainder + __exp_scaled;
  __ffloat __erfc_val  = renormalize(__t1 * __exp_xx);

  if (__is_neg)
  {
    __erfc_val = renormalize(__ffloat(2.0) - __erfc_val);
  }

  *__res_hi = __erfc_val.hi();
  *__res_lo = __erfc_val.lo();
} // __internal_fpmp2_erfc

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
  __fpmp2_from_double(::cuda::std::erfc(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
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
  using __ffloat = fpmp2<float, _CCCL_FPMP_METHOD>;

  __ffloat __a(__a_hi, __a_lo);
  __ffloat __r;

  if (__a_hi >= 0x1.6ebc6ap3f) // a >= 11.46
  {
    __r = rsqrt(__a);
  }

  if (__a_hi > 34.3816f)
  {
    constexpr __ffloat __sqrt_pi_4(0x1.c5bf891b4ef6bp-1);
    __ffloat __result = __sqrt_pi_4 * __r;
    _CCCL_FPMP_RENORMALIZE(__result);
    *__res_hi = __result.hi();
    *__res_lo = __result.lo();
    return;
  }

  if (__a_hi < 0x1p2f)
  {
    /* a < 4: 17-term minimax in x = 3 - a (|x| <= 3, |x| <= 1 typical),
     * evaluated via compensated Horner with an M = 4 plain-FpType
     * head (ascending order: c[0] = constant, c[16] = leading).
     *
     * The top 4 coefficients are c[13]..c[16] with magnitudes
     * 2^-42 .. 2^-52 ~= 3.6e-13 .. 2.3e-16. After 4 plain Horner
     * steps the accumulator magnitude is ~= 7e-12 (worst case |x|=3),
     * so the rounding error those steps drop is < |acc| * 2^-25
     * ~= 1e-19; carried forward through the remaining 13 ff Horner
     * steps and amplified by x^13 ~= 1.6e6 at worst, this still
     * sits well below the ff precision floor (~1.4e-14).  Phase 2a's
     * skipped iterations drop c[13..16].lo() ~= 2^-72 .. 2^-76,
     * totally negligible.  Phase 1 compensation still covers the
     * 13 lower-degree iterations where |acc| grows from 7e-12 to
     * ~0.5 -- that's where the rounding error actually matters.
     *
     * Tried M = 5 (skip compensation on c[12] too, |c[12]| ~= 5.3e-12)
     * and lost 2 bits across all buckets (43->41) with warn % blowing
     * up by 4 orders of magnitude -- c[12]'s rounding error,
     * propagated through 12 ff steps and amplified by x^12 worst-case,
     * just barely surfaces above the precision floor.  M = 4 is the
     * sweet spot.
     */
    __ffloat __x               = _CCCL_FPMP_SUB(__ffloat(0x1.8p1), __a);
    constexpr __ffloat __c[17] = {
      __ffloat(0x1.023951b248d32p-1),
      __ffloat(0x1.364f8131f82eap-4),
      __ffloat(0x1.e4ab5374f7553p-7),
      __ffloat(0x1.65408fedfe46fp-9),
      __ffloat(0x1.d70cd2ae22daap-12),
      __ffloat(0x1.133abad3c99dp-14),
      __ffloat(0x1.1e134e84b9a2ap-17),
      __ffloat(0x1.0a6cf9d0cf714p-20),
      __ffloat(0x1.c039bccbce7dep-24),
      __ffloat(0x1.572c0936d0dcp-27),
      __ffloat(0x1.e19e5b3b8b31bp-31),
      __ffloat(0x1.37ce8ea919fd3p-34),
      __ffloat(0x1.76402f1b7e023p-38),
      __ffloat(0x1.99cd5cbd06043p-42),
      __ffloat(0x1.03356d73ab25fp-45),
      __ffloat(0x1.887a5d0c86047p-52),
      __ffloat(0x1.07f3442d6af1ep-52)};
    __ffloat __v = __fpmp_poly_eval<__fpmp_poly_method::horner_comp, 4>(__x, __c);
    *__res_hi    = __v.hi();
    *__res_lo    = __v.lo();
    return;
  } // if (a_hi < 0x1p2f)

  if (__a_hi < 0x1.6ebc6ap3f)
  {
    /* 4 <= a < 11.46: 20-term minimax in x = 6.92 - a (degree 19).
     * Standard ff-Horner with periodic renormalization -- the
     * compensated variant loses accuracy here because the
     * coefficients span ~22 orders of magnitude. */
    __ffloat __x = _CCCL_FPMP_SUB(__ffloat(0x1.baf1a8p1), __a);
    __ffloat __v = __ffloat(0x1.95402da668f4fp-73);
    __v          = __v * __x + __ffloat(0x1.43744ab1a0e5ap-66);
    __v          = __v * __x + __ffloat(0x1.f70f3953813b1p-61);
    __v          = __v * __x + __ffloat(0x1.00b2c5aae06a1p-55);
    __v          = __v * __x + __ffloat(0x1.87ddc6a10f513p-51);
    __v          = __v * __x + __ffloat(0x1.e450e0340da6fp-47);
    __v          = __v * __x + __ffloat(0x1.ffc73283f2e3dp-43);
    __v          = __v * __x + __ffloat(0x1.dff8a98149ce4p-39);
    __v          = __v * __x + __ffloat(0x1.98aa56613b23p-35);
    _CCCL_FPMP_RENORMALIZE(__v);
    __v = __v * __x + __ffloat(0x1.3f3d23359c3f4p-31);
    __v = __v * __x + __ffloat(0x1.ca89e4f410357p-28);
    __v = __v * __x + __ffloat(0x1.2ddf249b49215p-24);
    _CCCL_FPMP_RENORMALIZE(__v);
    __v = __v * __x + __ffloat(0x1.6a60fc5c32d39p-21);
    __v = __v * __x + __ffloat(0x1.8a0af8927f728p-18);
    __v = __v * __x + __ffloat(0x1.81949bbc35f76p-15);
    _CCCL_FPMP_RENORMALIZE(__v);
    __v = __v * __x + __ffloat(0x1.51d1e0119bf15p-12);
    __v = __v * __x + __ffloat(0x1.090a189fdb05bp-9);
    _CCCL_FPMP_RENORMALIZE(__v);
    __v = __v * __x + __ffloat(0x1.7a16985c09ba2p-7);
    __v = __v * __x + __ffloat(0x1.04f3fb31bb071p-4);
    __v = __v * __x + __ffloat(0x1.e3ae966b0f402p-2);
    _CCCL_FPMP_RENORMALIZE(__v);
    *__res_hi = __v.hi();
    *__res_lo = __v.lo();
    return;
  } // if (a_hi < 0x1.6ebc6ap3f)

  /* 11.46 <= a <= 34.38: 19-term minimax in x = rsqrt(a)^2 - offset
   * (degree 18), evaluated via compensated Horner. Coefficients are
   * in ascending order (c[0] = constant, c[18] = leading). */
  __ffloat __x               = _CCCL_FPMP_SUB(__r * __r, __ffloat(0x1.dc88f0479694p-5));
  constexpr __ffloat __c[19] = {
    __ffloat(0x1.fffffed709646p-1),
    __ffloat(-0x1.71471b65714a8p-20),
    __ffloat(-0x1.85179c0504089p-13),
    __ffloat(-0x1.d99f05bac9192p-7),
    __ffloat(-0x1.681ebc0bfc87p-1),
    __ffloat(-0x1.531388eeb3e37p4),
    __ffloat(-0x1.56423d3c9aee8p8),
    __ffloat(-0x1.55574adbabed4p9),
    __ffloat(0x1.fc17297038ab6p15),
    __ffloat(0x1.5d36617bab8fep18),
    __ffloat(-0x1.c1e691926af02p23),
    __ffloat(-0x1.d06f6451b9b99p24),
    __ffloat(0x1.baa02bef66d96p31),
    __ffloat(-0x1.35636d415d49bp34),
    __ffloat(-0x1.61bc3c687e6ffp39),
    __ffloat(0x1.39af9c72a5c92p43),
    __ffloat(0x1.3f351ae8d044ap46),
    __ffloat(-0x1.ba4d5cfd521a5p50),
    __ffloat(-0x1.6d64bf85e3416p50)};
  __ffloat __v      = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__x, __c);
  __r               = __r * __ffloat(0x1.c5bf8ap-1);
  __ffloat __result = __v * __r;
  _CCCL_FPMP_RENORMALIZE(__result);

  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
} // __internal_fpmp2_boys_f0

/*
 * --------------------------------------------------------------------
 * Boys function boys_f0(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_boys_f0(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  double __x = __fpmp2_to_double(__x_hi, __x_lo);
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
  __fpmp2_from_double(__r, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(boys_f0)

/*
 * ====================================================================
 * normcdfinv(x) - inverse normal cumulative distribution function
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Inverse normal CDF normcdfinv(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Rational approximation (Mike Giles coefficients),
 * fully converted to fp32mp2 arithmetic (no fp64 operations).
 * Three regions selected by w = -log(4p(1-p)), a = 2p - 1:
 *   Central (w < 6.125):      degree-22 polynomial in (w - 3.125)
 *   Tail 1  (6.125 <= w < 16): degree-18 polynomial in (sqrt(w) - 3.25)
 *   Tail 2  (w >= 16):        degree-24 polynomial in (sqrt(w) - 7.25)
 * The central path (~99.9% of inputs) is straight-line code; tail
 * regions are branched off so the common path skips sqrt and tail
 * polynomials entirely.  All regions produce full fp32mp2 precision.
 *
 * The fp64mp2 body uses CUDA's erfcinv on device and falls back to
 * this polynomial on host (no standard erfcinv/normcdfinv).
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_normcdfinv(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  constexpr __ffloat __sqrt2(0x1.6a09e667f3bcdp+0);

  /* Central polynomial: rc(tc) = c22 + c21*tc + ... + c0*tc^22,
   * tc = w - 3.125  (>99.9% of inputs land here).
   * Ascending degree; M = 9 high-order entries (rc_c[14..22] =
   * original c0..c8) are plain float and run in the dispatcher's
   * FpType phase.  The transition step `rcf * tc.hi() + c9`
   * matches the previous hand-rolled float*float + ff step
   * bit-for-bit, so this refactor is numerically identical.
   */
  constexpr __ffloat __rc_c[23] = {
    __ffloat(1.6536545626831027e+00), // [ 0] (= c22, constant)
    __ffloat(2.4015818242558962e-01), // [ 1] (= c21)
    __ffloat(-6.0336708714301491e-03), // [ 2] (= c20)
    __ffloat(-7.4070253416626698e-04), // [ 3] (= c19)
    __ffloat(1.8673420803405714e-04), // [ 4] (= c18)
    __ffloat(-1.3882523362786469e-05), // [ 5] (= c17)
    __ffloat(-1.3654692000834679e-06), // [ 6] (= c16)
    __ffloat(4.2347877827932404e-07), // [ 7] (= c15)
    __ffloat(-2.9070369957882005e-08), // [ 8] (= c14)
    __ffloat(-4.1126339803469837e-09), // [ 9] (= c13)
    __ffloat(1.0512122733215323e-09), // [10] (= c12)
    __ffloat(-5.4154120542946279e-11), // [11] (= c11)
    __ffloat(-1.2975133253453532e-11), // [12] (= c10)
    __ffloat(2.6335093153082323e-12), // [13] (= c9, last ff term)
    /* high-order M = 9 entries: .lo() == 0 by construction */
    __ffloat(-8.1519342e-14f), // [14] (= c8)
    __ffloat(-4.0545663e-14f), // [15] (= c7)
    __ffloat(6.6376381e-15f), // [16] (= c6)
    __ffloat(2.0972768e-17f), // [17] (= c5)
    __ffloat(-1.3331717e-16f), // [18] (= c4)
    __ffloat(1.1157878e-17f), // [19] (= c3)
    __ffloat(1.2858481e-18f), // [20] (= c2)
    __ffloat(-1.6850591e-19f), // [21] (= c1)
    __ffloat(-3.6444121e-21f) // [22] (= c0, leading)
  };

  /* Tail 1 polynomial: rt(tt) = t18 + t17*tt + ... + t0*tt^18,
   * tt = sqrt(w) - 3.25 (w in [6.25, 16],  |z| ~ 2.5 to 5.5 sigma).
   * Ascending degree; M = 9 high-order entries (rt_c[10..18] =
   * original t0..t8) are plain float.  Transition is float*float
   * + ff at rt_c[9] = t9 -- bit-identical to the previous chain.
   */
  constexpr __ffloat __rt_c[19] = {
    __ffloat(3.0838856104922208e+00), // [ 0] (= t18, constant)
    __ffloat(1.0052589676941592e+00), // [ 1] (= t17)
    __ffloat(5.3709145535900636e-03), // [ 2] (= t16)
    __ffloat(-3.7512085075692412e-03), // [ 3] (= t15)
    __ffloat(2.4914420961078508e-03), // [ 4] (= t14)
    __ffloat(-1.6882755560235047e-03), // [ 5] (= t13)
    __ffloat(9.5328937973738050e-04), // [ 6] (= t12)
    __ffloat(-3.5503752036284748e-04), // [ 7] (= t11)
    __ffloat(2.4031110387097894e-05), // [ 8] (= t10)
    __ffloat(6.8284851459573175e-05), // [ 9] (= t9, last ff term)
    /* high-order M = 9 entries: .lo() == 0 by construction */
    __ffloat(-4.7318229e-05f), // [10] (= t8)
    __ffloat(1.2475304e-05f), // [11] (= t7)
    __ffloat(2.9234449e-06f), // [12] (= t6)
    __ffloat(-4.0138675e-06f), // [13] (= t5)
    __ffloat(1.5027404e-06f), // [14] (= t4)
    __ffloat(1.8239629e-08f), // [15] (= t3)
    __ffloat(-2.7517406e-07f), // [16] (= t2)
    __ffloat(9.0756562e-08f), // [17] (= t1)
    __ffloat(2.2137377e-09f) // [18] (= t0, leading)
  };

  /* Tail 2 polynomial: rt2(tt2) = u24 + u23*tt2 + ... + u0*tt2^24,
   * tt2 = sqrt(w) - 7.25  (w >= 16,  |z| > 5.5 sigma).
   * Covers all representable float inputs including denormals;
   * Chebyshev interp at 100-digit precision, relative approx
   * error < 2^{-46} over the fitted range.
   * Ascending degree; M = 13 high-order entries (rt2_c[12..24] =
   * original u0..u12) are plain float.  The transition step
   * `rt2f * tt2.hi() + u13` is one ULP-level different from
   * the previous chain (which lifted to ff one step earlier and
   * thus included tt2.lo() in the transition product); the
   * change is well inside the polynomial truncation noise.
   */
  constexpr __ffloat __rt2_c[25] = {
    __ffloat(7.12113663660053842e+00), // [ 0] (= u24, constant)
    __ffloat(1.00834082079167930e+00), // [ 1] (= u23)
    __ffloat(-5.05906408540271685e-04), // [ 2] (= u22)
    __ffloat(1.14184074807230187e-05), // [ 3] (= u21)
    __ffloat(4.29790660561751423e-06), // [ 4] (= u20)
    __ffloat(-1.21177482126504764e-06), // [ 5] (= u19)
    __ffloat(2.33428873326838655e-07), // [ 6] (= u18)
    __ffloat(-3.92578613880982197e-08), // [ 7] (= u17)
    __ffloat(6.14877480871698432e-09), // [ 8] (= u16)
    __ffloat(-9.24007580865063697e-10), // [ 9] (= u15)
    __ffloat(1.34759296085592452e-10), // [10] (= u14)
    __ffloat(-1.76387252450593334e-11), // [11] (= u13, last ff term)
    /* high-order M = 13 entries: .lo() == 0 by construction */
    __ffloat(2.09393731e-12f), // [12] (= u12)
    __ffloat(-6.91218317e-13f), // [13] (= u11)
    __ffloat(1.95788733e-13f), // [14] (= u10)
    __ffloat(4.98296865e-14f), // [15] (= u9)
    __ffloat(-2.64007334e-14f), // [16] (= u8)
    __ffloat(-5.68006053e-15f), // [17] (= u7)
    __ffloat(3.21120849e-15f), // [18] (= u6)
    __ffloat(2.6060760e-16f), // [19] (= u5)
    __ffloat(-2.2467865e-16f), // [20] (= u4)
    __ffloat(1.9526573e-18f), // [21] (= u3)
    __ffloat(7.8681698e-18f), // [22] (= u2)
    __ffloat(-6.7040324e-19f), // [23] (= u1)
    __ffloat(-2.2357236e-20f) // [24] (= u0, leading)
  };

  __ffloat __p = renormalize(__ffloat(__x_hi, __x_lo));

  /* Standard mathematical convention: normcdfinv(0) = -inf, normcdfinv(1) = +inf */
  if (__p.hi() <= 0.0f)
  {
    *__res_hi = ::cuda::std::bit_cast<float>(0xFF800000U);
    *__res_lo = 0.0f;
    return;
  }
  if (__p.hi() >= 1.0f)
  {
    *__res_hi = ::cuda::std::bit_cast<float>(0x7F800000U);
    *__res_lo = 0.0f;
    return;
  }

  /* a = 2p - 1, accurate subtraction for p ~= 0.5 */
  __ffloat __two_p = __p + __p;
  __ffloat __a     = sub<fpmp2_accuracy::high>(__two_p, 1.0f);

  /* w = -log(1 - a^2) = -log(4p(1-p))
   * Compute 1-p with accurate subtraction to handle p near 0 or 1
   */
  __ffloat __omp = sub<fpmp2_accuracy::high>(1.0f, __p);
  __ffloat __arg = 4.0f * __p * __omp;

  if (__arg.hi() <= 0.0f)
  {
    __arg = __ffloat(0x1.0p-126f);
  }

  float __log_hi;
  float __log_lo;
  __fpmp2_log(__arg.hi(), __arg.lo(), &__log_hi, &__log_lo);
  __ffloat __w = -__ffloat(__log_hi, __log_lo);

  /* Central region (w < 6.125, |z| < ~3.3 sigma, >99.9% of inputs):
   * Horner in tc = w - 3.125 via the mixed-precision dispatcher
   * (9 high-order float coeffs c0..c8, 14 low-order ff coeffs c9..c22).
   */
  __ffloat __tc   = __w - __ffloat(3.125f);
  __ffloat __poly = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 9>(__tc, __rc_c);

  /* Tail regions (w >= 6.125): branched since <0.1% of inputs.
   * sqrt(w) is also deferred into this branch.
   */
  if (__w.hi() >= 6.125f)
  {
    float __sw_hi;
    float __sw_lo;
    __fpmp2_sqrt(__w.hi(), __w.lo(), &__sw_hi, &__sw_lo);
    __ffloat __sw(__sw_hi, __sw_lo);

    /* Tail 1 (6.125 <= w < 16, |z| ~ 3.3 to 5.5 sigma):
     * Horner in tt = sqrt(w) - 3.25 via the dispatcher
     * (9 high-order float coeffs t0..t8, 10 low-order ff coeffs t9..t18).
     */
    __ffloat __tt = __sw - __ffloat(3.25f);
    __poly        = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 9>(__tt, __rt_c);

    /* Tail 2 (w >= 16, |z| > 5.5 sigma):
     * Horner in tt2 = sqrt(w) - 7.25 via the dispatcher
     * (13 high-order float coeffs u0..u12, 12 low-order ff coeffs u13..u24).
     *
     * Note: the dispatcher's transition step uses tt2.hi() only
     * (float*float + ff), whereas the previous hand-rolled chain
     * promoted to ff one step earlier and thus included tt2.lo()
     * in the transition product.  The numerical change is sub-ULP
     * at the polynomial value and well inside the truncation noise.
     */
    if (__w.hi() >= 16.0f)
    {
      __ffloat __tt2 = __sw - __ffloat(7.25f);
      __poly         = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 13>(__tt2, __rt2_c);
    }
  }

  /* Scale: result = poly * a * sqrt(2) */
  __ffloat __result = renormalize(__poly * __a * __sqrt2);

  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
} // __internal_fpmp2_normcdfinv

/*
 * --------------------------------------------------------------------
 * Inverse normal CDF normcdfinv(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_normcdfinv(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  double __p = __fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    ({
      // Hardcoded value since M_SQRT2 is not guaranteed to be defined on all platforms
      constexpr double __sqrt2_v = 1.41421356237309504880;
      __fpmp2_from_double(-__sqrt2_v * ::erfcinv(2.0 * __p), __res_hi, __res_lo);
    }),
    ({
      // Not implemented yet: double precision normcdfinv fallback to float precision
      float __f_hi;
      float __f_lo;
      __internal_fpmp2_normcdfinv(static_cast<float>(__p), 0.0f, &__f_hi, &__f_lo);
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

  __fpmp2_normcdfinv(__p_hi, __p_lo, __res_hi, __res_lo);
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

  __fpmp2_normcdfinv(__p_hi, __p_lo, __res_hi, __res_lo);
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
  __fpmp2_from_double(::cuda::std::lgamma(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
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
  __fpmp2_from_double(::cuda::std::tgamma(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
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
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (__fpmp2_from_double(::j0(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);), ({
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
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (__fpmp2_from_double(::j1(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);), ({
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
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (__fpmp2_from_double(::y0(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);), ({
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
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (__fpmp2_from_double(::y1(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);), ({
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
    NV_IS_DEVICE, (__fpmp2_from_double(::cyl_bessel_i0(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);), ({
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
    NV_IS_DEVICE, (__fpmp2_from_double(::cyl_bessel_i1(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);), ({
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
    NV_IS_DEVICE, (__fpmp2_from_double(::jn(__n, __fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);), ({
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
    NV_IS_DEVICE, (__fpmp2_from_double(::yn(__n, __fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);), ({
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
 * Normal CDF normcdf(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_normcdf(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __xd   = static_cast<double>(__mp2_t(__x_hi, __x_lo));
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (__r = ::normcdf(__xd);), (__r = 0.5 * ::cuda::std::erfc(-__xd * 0.70710678118654752440);))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
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
  const __fpmp_fp128 __xq  = __fpmp2_to_quad(__x_hi, __x_lo);
  const __fpmp_fp128 __two = (__fpmp_fp128) 2;
  __fpmp2_from_quad(_CCCL_FPMP_ERFCQ(-__xq / _CCCL_FPMP_SQRTQ(__two)) / __two, __res_hi, __res_lo);
#  else
  double __xd = __fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (__fpmp2_from_double(::normcdf(__xd), __res_hi, __res_lo);),
                    (__fpmp2_from_double(0.5 * ::cuda::std::erfc(-__xd * 0.70710678118654752440), __res_hi, __res_lo);))
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
 * Inverse complementary error function erfcinv(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 * Device only: the CUDA intrinsic carries it, the host build asserts and
 * returns 0.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_erfcinv(
  [[maybe_unused]] const float __x_hi, [[maybe_unused]] const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__r = ::erfcinv(static_cast<double>(__mp2_t(__x_hi, __x_lo)));), ({
                      _CCCL_ASSERT(false, "erfcinv: no host fallback, returning 0");
                    }))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
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
    NV_IS_DEVICE, (__fpmp2_from_double(::erfcinv(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);), ({
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
 * Inverse error function erfinv(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_erfinv(
  [[maybe_unused]] const float __x_hi, [[maybe_unused]] const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__r = ::erfinv(static_cast<double>(__mp2_t(__x_hi, __x_lo)));), ({
                      _CCCL_ASSERT(false, "erfinv: no host fallback, returning 0");
                    }))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
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
    NV_IS_DEVICE, (__fpmp2_from_double(::erfinv(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);), ({
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
 * Scaled complementary error function erfcx(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_erfcx(
  [[maybe_unused]] const float __x_hi, [[maybe_unused]] const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__r = ::erfcx(static_cast<double>(__mp2_t(__x_hi, __x_lo)));), ({
                      _CCCL_ASSERT(false, "erfcx: no host fallback, returning 0");
                    }))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
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
    NV_IS_DEVICE, (__fpmp2_from_double(::erfcx(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);), ({
      _CCCL_ASSERT(false, "erfcx: no host fallback, returning 0");
      *__res_hi = 0.0;
      *__res_lo = 0.0;
    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A(erfcx)

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_SPECIAL_H
