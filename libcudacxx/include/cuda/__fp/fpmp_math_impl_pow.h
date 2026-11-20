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

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
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
#ifndef _CCCL_FPMP_USE_LIB

/*
 * ====================================================================
 * pow(x, y) - x raised to the power y
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Power function pow(x, y) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * pow(a, b) = exp(b ln|a|), negated for a negative a and an odd integer b.
 *
 * exp turns an absolute error in its argument into the same relative error in its result, and
 * |b ln|a|| reaches 104 before the result underflows, so b ln|a| rounded to the format's own
 * precision would cost up to log2(104) ~= 7 of the 46 bits. ln|a| is therefore formed to about
 * 2^-58 relative, as three floats, and b ln|a| to about 2^-49 absolute. Against binary128, normal
 * results measure within 3u^2 (u = 2^-24), mean 0.3u^2, at every magnitude of b ln|a|.
 *
 * ln|a|: |a| = 2^k z with z in [0.701, 1.402). The bucket holding z_hi, one of 128, selects 1/c, with
 *   at most 9 significant bits, so that r0 = z_hi/c - 1 = fma(z_hi, 1/c, -1) is exact, and
 *   ln(c) = Lc0 + Lc1 + Lc2. The bucket around 1 has c = 1, which keeps ln|a| relatively accurate
 *   as a approaches 1. r = r0 + z_lo/c is exact as three floats, and
 *       ln|a| = k ln2 + ln(c) + r + delta,   delta = ln(1 + r) - r = -r^2/2 + r^3 P(r),
 *   where |r| < 2^-7.5 leaves delta needing only double-float accuracy. Lc0 and the leading part of
 *   ln2 are multiples of 2^-16, so that k ln2_0 + Lc0 is exact.
 * b ln|a|: three exact products, the multiple k_e of ln2/128 nearest them, and the reduced
 *   argument phi = b ln|a| - k_e ln2/128, |phi| < 2^-8.5, as a float pair.
 * exp: e^phi - 1 by a degree-5 polynomial, times 2^(j/128), j = k_e mod 128, from the table, times
 *   2^n, n = k_e div 128, applied in two steps so that a subnormal result is rounded once.
 *
 * Special values follow C99 Annex F (F.10.4.4) on the value hi + lo: b is an integer when both
 * limbs are, and odd when one limb is an odd integer, which for |b| >= 2^24 is a property of b_lo.
 * Overflow and underflow are decided from b_hi * ln|a| before the exact products are formed, so
 * no intermediate is infinite or NaN.
 * --------------------------------------------------------------------
 */

/*
 * ln|x| for finite x = x_hi + x_lo > 0 as __l0 + __l1 + __l2, |__l1| < 2^-22 |__l0|, within
 * about 2^-58 of ln|x| relatively.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_pow_log(float __x_hi, float __x_lo, float* __l0, float* __l1, float* __l2) noexcept
{
  constexpr float __ln2_0    = 0x1.62e4p-1f; // ln2 = __ln2_0 + __ln2_1 + __ln2_2
  constexpr float __ln2_1    = 0x1.7f7d1cp-20f;
  constexpr float __ln2_2    = 0x1.ef357ap-45f;
  constexpr float __third_hi = 0x1.555556p-2f; // 1/3 = __third_hi + __third_lo
  constexpr float __third_lo = -0x1.555556p-27f;

  int __k_adj = 0;
  if (__x_hi < 0x1p-126f) // a subnormal x_hi, whose lo is zero, is scaled into the normal range
  {
    __x_hi  = ::cuda::__fpmp_mul_rn(__x_hi, 0x1p24f);
    __x_lo  = ::cuda::__fpmp_mul_rn(__x_lo, 0x1p24f);
    __k_adj = -24;
  }
  // x_hi = z_hi 2^__ke, and r = r0 + rho_hi + rho_lo = z/c - 1, exactly
  __fpmp2_log_table_entry __t;
  float __z_hi, __r0, __rho_hi, __rho_lo;
  const int __ke = ::cuda::__internal_fpmp2_log_bucket(__x_hi, &__z_hi, &__t);
  ::cuda::__internal_fpmp2_log_residual(__z_hi, __x_lo, __ke, __t, &__r0, &__rho_hi, &__rho_lo);

  // k ln2 + ln(c) = h0 + h1 + h2, h0 exact
  const float __kf = static_cast<float>(__ke + __k_adj);
  const float __h0 = ::cuda::__fpmp_fma_rn(__kf, __ln2_0, __t.__logc0);
  float __kl1_lo, __h1_lo;
  const float __kl1_hi = ::cuda::__fpmp_two_mult_fma(__kf, __ln2_1, &__kl1_lo);
  const float __h1     = ::cuda::__fpmp_two_sum(__kl1_hi, __t.__logc1, &__h1_lo);
  const float __h2 =
    ::cuda::__fpmp_add_rn(::cuda::__fpmp_fma_rn(__kf, __ln2_2, __t.__logc2), ::cuda::__fpmp_add_rn(__kl1_lo, __h1_lo));

  // r as a pair for the nonlinear terms. r0 + rho_hi is a valid fast two-sum unless |r0| < |rho_hi|,
  // where those terms are below 2^-40 of ln|x| and need no more accuracy.
  float __r_lo;
  const float __r_hi = ::cuda::__fpmp_fast_two_sum(__r0, __rho_hi, &__r_lo);
  __r_lo             = ::cuda::__fpmp_add_rn(__r_lo, __rho_lo);

  // delta = -r^2/2 + r^3 P(r), P(r) = 1/3 - r/4 + r^2/5 - r^3/6 + r^4/7 - r^5/8, as a pair
  float __sq_lo;
  const float __sq_hi = ::cuda::__fpmp_two_mult_fma(__r_hi, __r_hi, &__sq_lo);
  __sq_lo             = ::cuda::__fpmp_fma_rn(2.0f * __r_hi, __r_lo, __sq_lo);
  float __cu_lo;
  const float __cu_hi = ::cuda::__fpmp_two_mult_fma(__sq_hi, __r_hi, &__cu_lo);
  __cu_lo             = ::cuda::__fpmp_fma_rn(__sq_lo, __r_hi, ::cuda::__fpmp_fma_rn(__sq_hi, __r_lo, __cu_lo));
  float __p_tail      = ::cuda::__fpmp_fma_rn(__r_hi, -0x1p-3f, 0x1.24924ap-3f);
  __p_tail            = ::cuda::__fpmp_fma_rn(__r_hi, __p_tail, -0x1.555556p-3f);
  __p_tail            = ::cuda::__fpmp_fma_rn(__r_hi, __p_tail, 0x1.99999ap-3f);
  __p_tail            = ::cuda::__fpmp_mul_rn_nocontract(__sq_hi, __p_tail);
  float __p_lo;
  const float __p_hi = ::cuda::__fpmp_fast_two_sum(__third_hi, -0.25f * __r_hi, &__p_lo);
  __p_lo =
    ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__p_lo, __third_lo), ::cuda::__fpmp_fma_rn(-0.25f, __r_lo, __p_tail));
  float __q_lo;
  const float __q_hi = ::cuda::__fpmp_two_mult_fma(__cu_hi, __p_hi, &__q_lo);
  __q_lo             = ::cuda::__fpmp_fma_rn(__cu_lo, __p_hi, ::cuda::__fpmp_fma_rn(__cu_hi, __p_lo, __q_lo));
  float __d_lo;
  const float __d_hi = ::cuda::__fpmp_fast_two_sum(-0.5f * __sq_hi, __q_hi, &__d_lo);
  __d_lo             = ::cuda::__fpmp_add_rn(__d_lo, ::cuda::__fpmp_fma_rn(-0.5f, __sq_lo, __q_lo));

  // ln|x| = h0 + h1 + r0 + rho_hi + d_hi, summed exactly (each fast two-sum adds a term no larger
  // than the running sum, or into a zero sum), plus the remaining terms below 2^-22 of it.
  float __e0, __e1, __e2, __e3;
  float __sum = ::cuda::__fpmp_fast_two_sum(__h0, __h1, &__e0);
  __sum       = ::cuda::__fpmp_fast_two_sum(__sum, __r0, &__e1);
  __sum       = ::cuda::__fpmp_fast_two_sum(__sum, __rho_hi, &__e2);
  __sum       = ::cuda::__fpmp_fast_two_sum(__sum, __d_hi, &__e3);
  float __g1, __g2, __g3;
  float __mid = ::cuda::__fpmp_two_sum(__e0, __e1, &__g1);
  __mid       = ::cuda::__fpmp_two_sum(__mid, __e2, &__g2);
  __mid       = ::cuda::__fpmp_two_sum(__mid, __e3, &__g3);
  *__l0       = __sum;
  *__l1       = __mid;
  *__l2       = ::cuda::__fpmp_add_rn(
    ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__g1, __g2), ::cuda::__fpmp_add_rn(__g3, __h2)),
    ::cuda::__fpmp_add_rn(__rho_lo, __d_lo));
} // __internal_fpmp2_pow_log

/*
 * e^(b L) = (__y_hi + __y_lo) 2^__n for b = b_hi + b_lo and L = l0 + l1 + l2 from
 * __internal_fpmp2_pow_log, given |b_hi l0| <= 105.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_pow_exp(
  const float __b_hi,
  const float __b_lo,
  const float __l0,
  const float __l1,
  const float __l2,
  float* __y_hi,
  float* __y_lo,
  int* __n) noexcept
{
  // f = b L = p0 + (p1 + q0 + r0) + (q1 + r1 + s)
  float __p1, __q1, __r1;
  const float __p0 = ::cuda::__fpmp_two_mult_fma(__b_hi, __l0, &__p1);
  const float __q0 = ::cuda::__fpmp_two_mult_fma(__b_hi, __l1, &__q1);
  const float __r0 = ::cuda::__fpmp_two_mult_fma(__b_lo, __l0, &__r1);
  const float __s  = ::cuda::__fpmp_fma_rn(__b_lo, __l1, ::cuda::__fpmp_mul_rn_nocontract(__b_hi, __l2));

  // k = round(p0 / (ln2/128)); phi = f - k ln2/128 as a pair, with ln2/128 = c0 + c1 + c2.
  // k c0 and p0 - k c0 are exact.
  float __kf;
  const int __k   = ::cuda::__internal_fpmp2_round_product(__p0, __fpmp2_inv_ln2_128, &__kf);
  const float __d = ::cuda::__fpmp_fma_rn(-__kf, __fpmp2_ln2_128_0, __p0);
  float __kc1_lo;
  const float __kc1_hi = ::cuda::__fpmp_two_mult_fma(__kf, __fpmp2_ln2_128_1, &__kc1_lo);
  float __v1, __v2, __v3, __v4;
  const float __u  = ::cuda::__fpmp_two_sum(__d, -__kc1_hi, &__v1);
  float __w        = ::cuda::__fpmp_two_sum(__p1, __q0, &__v2);
  __w              = ::cuda::__fpmp_two_sum(__w, __r0, &__v3);
  const float __ph = ::cuda::__fpmp_two_sum(__u, __w, &__v4);
  const float __pl = ::cuda::__fpmp_add_rn(
    ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__v1, __v2), ::cuda::__fpmp_add_rn(__v3, __v4)),
    ::cuda::__fpmp_sub_rn(::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__q1, __r1), __s),
                          ::cuda::__fpmp_fma_rn(__kf, __fpmp2_ln2_128_2, __kc1_lo)));
  float __phi_lo;
  const float __phi_hi = ::cuda::__fpmp_fast_two_sum(__ph, __pl, &__phi_lo);

  ::cuda::__internal_fpmp2_exp_tail(__phi_hi, __phi_lo, __k, __y_hi, __y_lo, __n);
} // __internal_fpmp2_pow_exp

// Whether the integer __x is odd; every float of magnitude 2^24 or more is even. An integer
// b = b_hi + b_lo is odd when exactly one limb is.
_CCCL_FPMP_CORE_API bool __internal_fpmp2_pow_limb_is_odd(const float __x) noexcept
{
  return ::cuda::__fpmp_internal_fabs(__x) < 0x1p24f && (::cuda::__fpmp_fp2int_rz(__x) & 1) != 0;
}

_CCCL_FPMP_CORE_API void __internal_fpmp2_pow(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  const float __inf      = ::cuda::std::__fp_inf<float>();
  const float __abs_b_hi = ::cuda::__fpmp_internal_fabs(__b_hi);
  // x = |a|, and the sign of the result
  float __x_hi = __a_hi;
  float __x_lo = __a_lo;
  float __sign = 1.0f;

  // Everything but a finite a > 0 with a finite b != 0: NaN, zero and infinite operands, b = 0,
  // and a negative a, which continues with |a| when b is an integer.
  if (!(__a_hi > 0.0f && __a_hi < __inf && __abs_b_hi < __inf && __b_hi != 0.0f))
  {
    // pow(a, +-0) = 1 and pow(+1, b) = 1, even for a NaN operand.
    if (__b_hi == 0.0f || (__a_hi == 1.0f && __a_lo == 0.0f))
    {
      *__res_hi = 1.0f;
      *__res_lo = 0.0f;
      return;
    }
    if (__a_hi != __a_hi || __b_hi != __b_hi)
    {
      *__res_hi = __a_hi + __b_hi;
      *__res_lo = *__res_hi;
      return;
    }
    const bool __a_neg = (::cuda::std::bit_cast<uint32_t>(__a_hi) >> 31) != 0;
    __x_hi             = __a_neg ? -__a_hi : __a_hi;
    __x_lo             = __a_neg ? -__a_lo : __a_lo;
    bool __b_is_int    = false;
    bool __b_is_odd    = false;
    if (__abs_b_hi < __inf)
    {
      __b_is_int = ::cuda::__fpmp_internal_trunc<float>(__b_hi) == __b_hi
                && ::cuda::__fpmp_internal_trunc<float>(__b_lo) == __b_lo;
      __b_is_odd = __b_is_int
                && ::cuda::__internal_fpmp2_pow_limb_is_odd(__b_hi) != ::cuda::__internal_fpmp2_pow_limb_is_odd(__b_lo);
    }
    __sign = (__a_neg && __b_is_odd) ? -1.0f : 1.0f;
    // pow(+-0, b): +-0 for b > 0 and +-inf for b < 0, signed for an odd integer b
    if (__x_hi == 0.0f)
    {
      *__res_hi = __b_hi < 0.0f ? __sign * __inf : __sign * 0.0f;
      *__res_lo = 0.0f;
      return;
    }
    // pow(+-inf, b): +-inf for b > 0 and +-0 for b < 0, signed for an odd integer b
    if (__x_hi == __inf)
    {
      *__res_hi = __b_hi < 0.0f ? __sign * 0.0f : __sign * __inf;
      *__res_lo = 0.0f;
      return;
    }
    // pow(a, +-inf) for a finite a != 0: 1 for |a| = 1, else +inf or +0 by |a| > 1 and the sign of b
    if (__abs_b_hi == __inf)
    {
      const bool __abs_gt_one = __x_hi > 1.0f || (__x_hi == 1.0f && __x_lo > 0.0f);
      const bool __abs_is_one = __x_hi == 1.0f && __x_lo == 0.0f;
      if (__abs_is_one)
      {
        *__res_hi = 1.0f;
      }
      else
      {
        *__res_hi = (__b_hi > 0.0f) == __abs_gt_one ? __inf : 0.0f;
      }
      *__res_lo = 0.0f;
      return;
    }
    // a < 0 finite, b finite: NaN unless b is an integer
    if (!__b_is_int)
    {
      *__res_hi = ::cuda::std::__fp_nan<float>();
      *__res_lo = *__res_hi;
      return;
    }
  }

  float __l0, __l1, __l2;
  ::cuda::__internal_fpmp2_pow_log(__x_hi, __x_lo, &__l0, &__l1, &__l2);
  // b ln|a| above 89.5 overflows and below -105 rounds to zero; deciding it here keeps the
  // products below from overflowing.
  const float __f = ::cuda::__fpmp_mul_rn(__b_hi, __l0);
  if (!(__f <= 89.5f))
  {
    *__res_hi = __sign * __inf;
    *__res_lo = 0.0f;
    return;
  }
  if (__f < -105.0f)
  {
    *__res_hi = __sign * 0.0f;
    *__res_lo = 0.0f;
    return;
  }
  float __y_hi, __y_lo, __r_hi, __r_lo;
  int __n;
  ::cuda::__internal_fpmp2_pow_exp(__b_hi, __b_lo, __l0, __l1, __l2, &__y_hi, &__y_lo, &__n);
  ::cuda::__internal_fpmp2_exp_scale(__y_hi, __y_lo, __n, &__r_hi, &__r_lo);
  *__res_hi = __sign * __r_hi;
  *__res_lo = __sign * __r_lo;
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
 * Single-precision seeds for cbrt(x) and rcbrt(x)
 * ====================================================================
 *
 * r^(1/3) and r^(-1/3) for the reduced argument r in [1/4, 2), to a
 * relative error of about 2^-23; the float polish and the Halley step
 * that follow need no more.
 *
 * The device evaluates ex2(+-lg2(r) / 3) with the SFU lg2/ex2 pair. A
 * host whose fmaf is a hardware instruction refines a minimax polynomial
 * instead, which is faster there than exp2f and log2f. Every multiply-add
 * is an explicit fma, so that result depends neither on the compiler's
 * contraction nor on the C library. Over every float in [1/4, 2) these
 * seeds err by at most 2^-23.98 (cbrt) and 2^-23.29 (rcbrt).
 *
 * Any other host keeps exp2f and log2f: with fmaf a library call, the
 * polynomial seeds make cbrt 9% and rcbrt 14% slower on x86-64.
 *
 * Polynomials: Sollya 8.0,
 *   fpminimax(x^(1/3),  4, [|single...|], [1/4; 2], relative, floating)
 *   fpminimax(x^(-1/3), 3, [|single...|], [1/4; 2], relative, floating)
 * with relative errors of at most 1.87e-3 and 1.69e-2.
 */

// FP_FAST_FMAF is the <cmath> statement that fmaf is hardware; clang does
// not make it on x86, where __FMA__ says the same.
#  if defined(FP_FAST_FMAF) || defined(__FMA__)
#    define _CCCL_FPMP_HOST_SEED_POLY 1
#  else
#    define _CCCL_FPMP_HOST_SEED_POLY 0
#  endif

#  if _CCCL_FPMP_HOST_SEED_POLY

/* Host seed for cbrt: one Halley step s - s (s^3 - r) / (2 s^3 + r) on
 * the quartic, with s^3 - r = fma(s2_hi, s, -r) + s2_lo s for the exact
 * square s2_hi + s2_lo = s^2. A relative error e becomes 2/3 e^3:
 * 4.4e-9, below the rounding of the step. */
_CCCL_FPMP_CORE_API float __internal_fpmp2_cbrt_seed_host(const float __r) noexcept
{
  float __s = ::cuda::__fpmp_fma_rn(-0x1.1d6df4p-4f, __r, 0x1.8bab52p-2f);
  __s       = ::cuda::__fpmp_fma_rn(__s, __r, -0x1.b0554p-1f);
  __s       = ::cuda::__fpmp_fma_rn(__s, __r, 0x1.21c40ap+0f);
  __s       = ::cuda::__fpmp_fma_rn(__s, __r, 0x1.94a7acp-2f);

  float __s2_lo;
  const float __s2_hi = ::cuda::__fpmp_two_mult_fma(__s, __s, &__s2_lo);
  const float __numer = ::cuda::__fpmp_fma_rn(__s2_lo, __s, ::cuda::__fpmp_fma_rn(__s2_hi, __s, -__r));
  const float __denom = ::cuda::__fpmp_fma_rn(__s2_hi + __s2_hi, __s, __r);
  return ::cuda::__fpmp_fma_rn(-__s, __numer / __denom, __s);
}

/* Host seed for rcbrt: two division-free steps s (1 + u/3 + 2u^2/9),
 * u = 1 - r s^3, on the cubic. A relative error e becomes 14/3 e^3:
 * 2.2e-5 after the first step and 5e-14 after the second. */
_CCCL_FPMP_CORE_API float __internal_fpmp2_rcbrt_seed_host(const float __r) noexcept
{
  float __s = ::cuda::__fpmp_fma_rn(-0x1.092b32p-2f, __r, 0x1.2a688p+0f);
  __s       = ::cuda::__fpmp_fma_rn(__s, __r, -0x1.e31ca6p+0f);
  __s       = ::cuda::__fpmp_fma_rn(__s, __r, 0x1.f6b196p+0f);

  for (int __i = 0; __i < 2; ++__i)
  {
    const float __u  = ::cuda::__fpmp_fma_rn(-__r, (__s * __s) * __s, 1.0f);
    const float __hf = ::cuda::__fpmp_fma_rn(0x1.c71c72p-3f, __u, 0x1.555556p-2f);
    __s              = ::cuda::__fpmp_fma_rn(__u * __s, __hf, __s);
  }
  return __s;
}

#  endif // _CCCL_FPMP_HOST_SEED_POLY

_CCCL_FPMP_CORE_API float __internal_fpmp2_cbrt_seed(const float __r) noexcept
{
#  if _CCCL_FPMP_HOST_SEED_POLY
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (return ::cuda::__fpmp_fast_exp2(0x1.555556p-2f * ::cuda::__fpmp_fast_log2(__r));),
                    (return ::cuda::__internal_fpmp2_cbrt_seed_host(__r);))
#  else // ^^^ _CCCL_FPMP_HOST_SEED_POLY ^^^ / vvv !_CCCL_FPMP_HOST_SEED_POLY vvv
  return ::cuda::__fpmp_fast_exp2(0x1.555556p-2f * ::cuda::__fpmp_fast_log2(__r));
#  endif // ^^^ !_CCCL_FPMP_HOST_SEED_POLY ^^^
}

_CCCL_FPMP_CORE_API float __internal_fpmp2_rcbrt_seed(const float __r) noexcept
{
#  if _CCCL_FPMP_HOST_SEED_POLY
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (return ::cuda::__fpmp_fast_exp2(-0x1.555556p-2f * ::cuda::__fpmp_fast_log2(__r));),
                    (return ::cuda::__internal_fpmp2_rcbrt_seed_host(__r);))
#  else // ^^^ _CCCL_FPMP_HOST_SEED_POLY ^^^ / vvv !_CCCL_FPMP_HOST_SEED_POLY vvv
  return ::cuda::__fpmp_fast_exp2(-0x1.555556p-2f * ::cuda::__fpmp_fast_log2(__r));
#  endif // ^^^ !_CCCL_FPMP_HOST_SEED_POLY ^^^
}

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
 *      nexpo = round((expo - 126) / 3).  The reduced r lies in
 *      [1/4, 2): ax = m * 2^(expo - 126) with m in [1/2, 1), and
 *      (expo - 126) - 3 nexpo is -1, 0 or 1.
 *   5. Seed s0 ~ r_hi^(1/3) in single precision
 *      (__internal_fpmp2_cbrt_seed: the SFU lg2/ex2 pair on the device,
 *      ~2^-22; a refined polynomial on a host with hardware fma),
 *      polished by one Newton step in float with an exact s0^2, so s1 is
 *      within an ulp of cbrt(r) whatever the seed did.
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
  // 1/3 in single precision (round-to-nearest). (expo - 126) / 3 is at
  // least 1/6 away from a rounding tie, so the product rounds to the same
  // integer as the exact quotient.
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

  /* Seed: about 2^-22 relative on the device, 2^-24 on a host with hardware fma. */
  const float __s0 = ::cuda::__internal_fpmp2_cbrt_seed(__r_hi);

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
 *      nexpo = round((expo - 126) / 3).  The reduced r lies in
 *      [1/4, 2), as for cbrt.
 *   5. Seed s0 ~ r_hi^(-1/3) in single precision
 *      (__internal_fpmp2_rcbrt_seed: the SFU lg2/ex2 pair on the device,
 *      ~2^-22; a refined polynomial on a host with hardware fma),
 *      polished by one Newton step in float (no reciprocal needed), so
 *      s1 is within an ulp of 1/cbrt(r) whatever the seed did.
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
  // 1/3 and 2/9 in single precision (round-to-nearest). nexpo rounds as
  // in cbrt.
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

  /* Seed: about 2^-22 relative on the device, 2^-23 on a host with hardware fma. */
  const float __s0 = ::cuda::__internal_fpmp2_rcbrt_seed(__r_hi);

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
 * hypot, rhypot, norm3d, norm4d, rnorm3d, rnorm4d - Euclidean norms and their reciprocals
 * ====================================================================
 * For fp32mp2, rhypot and the 3- and 4-operand norms are the pair kernel below. The fp32mp2
 * hypot has a kernel of its own after it, which sums the two squares with one more error-free
 * step (2.9u^2 against 3.1u^2). In fp32mp2_high the norm3d, norm4d, rnorm3d and rnorm4d are
 * computed in double instead (the kernels of #12641, after the pair kernel). fp64mp2 keeps its
 * double fallbacks, which follow the fp32mp2 hypot.
 */

/*
 * --------------------------------------------------------------------
 * Euclidean norm sqrt(x_1^2 + ... + x_n^2) and its reciprocal (fp32mp2)
 * --------------------------------------------------------------------
 * The operands are scaled by a power of two chosen from the largest high limb, so that
 * no square overflows and the square of the largest one, with its rounding error, is a
 * pair of normal numbers; the scaling is exact, except for operands too small next to
 * the largest one to matter. The squares are formed and summed as pairs and the sum goes
 * through the square root kernel, or the reciprocal square root kernel.
 *
 * The roots are the short kernels of the default level (__fpmp2_sqrt, __fpmp2_rsqrt), at
 * every level. Against MPFR the fp32mp2 results are within 5.1 u^2 of the norm and 8.1 u^2 of
 * its reciprocal (13 u^2 on the device, whose reciprocal square root seed is approximate),
 * the errors of those kernels (u = 2^-24; 0.3 and 0.7 u^2 on average), over the whole
 * range, subnormal operands and results included.
 *
 * Special cases, answered on entry (C Annex F for hypot, extended to n operands and to
 * the reciprocal): an infinite operand gives +inf (+0 for the reciprocal), also next to
 * a NaN; otherwise a NaN operand gives NaN; all operands zero give +0 (+inf). A norm
 * that overflows gives +inf, a reciprocal that overflows, +inf.
 */
template <typename _FpType, int _Np, bool _Reciprocal>
_CCCL_FPMP_CORE_API void __internal_fpmp2_norm(
  const _FpType (&__x_hi)[_Np], const _FpType (&__x_lo)[_Np], _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  constexpr _FpType __max = (::cuda::std::numeric_limits<_FpType>::max)();
  constexpr _FpType __inf = ::cuda::std::numeric_limits<_FpType>::infinity();
  constexpr _FpType __nan = ::cuda::std::numeric_limits<_FpType>::quiet_NaN();

  // The largest magnitude; a NaN never enters it.
  _FpType __m = _FpType(0);
  for (int __k = 0; __k < _Np; ++__k)
  {
    const _FpType __a = ::cuda::__fpmp_internal_fabs(__x_hi[__k]);
    __m               = (__a > __m) ? __a : __m;
  }
  if (!(__m > _FpType(0) && __m <= __max))
  {
    // __m is zero or infinite.
    bool __has_nan = false;
    for (int __k = 0; __k < _Np; ++__k)
    {
      __has_nan = __has_nan || ::cuda::__fpmp_internal_isnan(__x_hi[__k]);
    }
    if (__m > __max)
    {
      *__res_hi = _Reciprocal ? _FpType(0) : __inf;
      *__res_lo = _FpType(0);
    }
    else if (__has_nan)
    {
      *__res_hi = *__res_lo = __nan;
    }
    else
    {
      *__res_hi = _Reciprocal ? __inf : _FpType(0);
      *__res_lo = _FpType(0);
    }
    return;
  }

  // The scaled largest operand m lies in [2^-40, 2^60) (fp32mp2; [2^-49, 2^60) from a
  // subnormal) or [2^-474, 2^500) (fp64mp2), where m^2 and the error of that product are
  // normal numbers and four squares cannot overflow.
  constexpr _FpType __big       = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p60f) : _FpType(0x1.0p500);
  constexpr _FpType __small     = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p-40f) : _FpType(0x1.0p-450);
  constexpr _FpType __big_scl   = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p-80f) : _FpType(0x1.0p-600);
  constexpr _FpType __big_inv   = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p80f) : _FpType(0x1.0p600);
  constexpr _FpType __small_scl = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p100f) : _FpType(0x1.0p600);
  constexpr _FpType __small_inv = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p-100f) : _FpType(0x1.0p-600);
  const _FpType __scl           = (__m >= __big) ? __big_scl : (__m < __small) ? __small_scl : _FpType(1);
  const _FpType __inv           = (__m >= __big) ? __big_inv : (__m < __small) ? __small_inv : _FpType(1);

  // s + c = sum of the squares, c holding the rounding errors and the low-limb terms.
  _FpType __s = _FpType(0);
  _FpType __c = _FpType(0);
  for (int __k = 0; __k < _Np; ++__k)
  {
    const _FpType __h = ::cuda::__fpmp_mul_rn(__x_hi[__k], __scl);
    const _FpType __l = ::cuda::__fpmp_mul_rn(__x_lo[__k], __scl);
    const _FpType __p = ::cuda::__fpmp_mul_rn(__h, __h);
    _FpType __e       = ::cuda::__fpmp_fma_rn(__h, __h, -__p);
    __e               = ::cuda::__fpmp_fma_rn(::cuda::__fpmp_add_rn(__h, __h), __l, __e);
    __e               = ::cuda::__fpmp_fma_rn(__l, __l, __e);
    if (__k == 0)
    {
      __s = __p;
      __c = __e;
    }
    else
    {
      _FpType __t;
      __s = ::cuda::__fpmp_two_sum(__s, __p, &__t);
      __c = ::cuda::__fpmp_add_rn(__c, ::cuda::__fpmp_add_rn(__t, __e));
    }
  }
  _FpType __a_lo;
  const _FpType __a_hi = ::cuda::__fpmp_fast_two_sum(__s, __c, &__a_lo);

  _FpType __r_hi, __r_lo;
  if constexpr (_Reciprocal)
  {
    ::cuda::__fpmp2_rsqrt(__a_hi, __a_lo, &__r_hi, &__r_lo);
    __r_hi = ::cuda::__fpmp_mul_rn(__r_hi, __scl);
    __r_lo = ::cuda::__fpmp_mul_rn(__r_lo, __scl);
  }
  else
  {
    ::cuda::__fpmp2_sqrt(__a_hi, __a_lo, &__r_hi, &__r_lo);
    __r_hi = ::cuda::__fpmp_mul_rn(__r_hi, __inv);
    __r_lo = ::cuda::__fpmp_mul_rn(__r_lo, __inv);
  }
  // Overflow, or a NaN operand next to finite ones.
  if (!(__r_hi <= __max))
  {
    __r_lo = ::cuda::__fpmp_internal_isnan(__r_hi) ? __nan : _FpType(0);
  }
  *__res_hi = __r_hi;
  *__res_lo = __r_lo;
}

#  define _CCCL_FPMP_NORM_2A(name, recip)                                               \
    template <typename _FpType>                                                         \
    _CCCL_FPMP_CORE_API void __internal_fpmp2_##name(                                   \
      const _FpType __x_hi,                                                             \
      const _FpType __x_lo,                                                             \
      const _FpType __y_hi,                                                             \
      const _FpType __y_lo,                                                             \
      _FpType* __res_hi,                                                                \
      _FpType* __res_lo) noexcept                                                       \
    {                                                                                   \
      const _FpType __hi[2] = {__x_hi, __y_hi};                                         \
      const _FpType __lo[2] = {__x_lo, __y_lo};                                         \
      ::cuda::__internal_fpmp2_norm<_FpType, 2, recip>(__hi, __lo, __res_hi, __res_lo); \
    }                                                                                   \
    _CCCL_FPMP_MATH_DISPATCH_2A(name)

#  define _CCCL_FPMP_NORM_3A(name, recip)                                               \
    template <typename _FpType>                                                         \
    _CCCL_FPMP_CORE_API void __internal_fpmp2_##name(                                   \
      const _FpType __a_hi,                                                             \
      const _FpType __a_lo,                                                             \
      const _FpType __b_hi,                                                             \
      const _FpType __b_lo,                                                             \
      const _FpType __c_hi,                                                             \
      const _FpType __c_lo,                                                             \
      _FpType* __res_hi,                                                                \
      _FpType* __res_lo) noexcept                                                       \
    {                                                                                   \
      const _FpType __hi[3] = {__a_hi, __b_hi, __c_hi};                                 \
      const _FpType __lo[3] = {__a_lo, __b_lo, __c_lo};                                 \
      ::cuda::__internal_fpmp2_norm<_FpType, 3, recip>(__hi, __lo, __res_hi, __res_lo); \
    }                                                                                   \
    _CCCL_FPMP_MATH_DISPATCH_3A(name)

#  define _CCCL_FPMP_NORM_4A(name, recip)                                               \
    template <typename _FpType>                                                         \
    _CCCL_FPMP_CORE_API void __internal_fpmp2_##name(                                   \
      const _FpType __a_hi,                                                             \
      const _FpType __a_lo,                                                             \
      const _FpType __b_hi,                                                             \
      const _FpType __b_lo,                                                             \
      const _FpType __c_hi,                                                             \
      const _FpType __c_lo,                                                             \
      const _FpType __d_hi,                                                             \
      const _FpType __d_lo,                                                             \
      _FpType* __res_hi,                                                                \
      _FpType* __res_lo) noexcept                                                       \
    {                                                                                   \
      const _FpType __hi[4] = {__a_hi, __b_hi, __c_hi, __d_hi};                         \
      const _FpType __lo[4] = {__a_lo, __b_lo, __c_lo, __d_lo};                         \
      ::cuda::__internal_fpmp2_norm<_FpType, 4, recip>(__hi, __lo, __res_hi, __res_lo); \
    }                                                                                   \
    _CCCL_FPMP_MATH_DISPATCH_4A(name)

/*
 * --------------------------------------------------------------------
 * x^2 + y^2 for the fp32mp2 hypot kernel
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
 * one is NaN (C99 F.10.4.3); otherwise a NaN propagates, into both
 * limbs.  A norm that overflows is +inf with a zero low limb.  No fp64.
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
    *__res_lo        = __inf ? 0.0f : *__res_hi; /* NaN in both limbs */
    return;
  }
  float __a_hi, __a_lo, __h_hi, __h_lo;
  const int __k = ::cuda::__internal_fpmp2_hypot_sum(__x_hi, __x_lo, __y_hi, __y_lo, &__a_hi, &__a_lo);
  ::cuda::__fpmp2_sqrt(__a_hi, __a_lo, &__h_hi, &__h_lo);
  const float __back = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __k) << 23));
  *__res_hi          = __h_hi * __back;
  *__res_lo          = (*__res_hi < ::cuda::std::__fp_inf<float>()) ? __h_lo * __back : 0.0f; /* (inf, 0) on overflow */
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

_CCCL_FPMP_NORM_2A(rhypot, true)
_CCCL_FPMP_NORM_3A(norm3d, false)
_CCCL_FPMP_NORM_3A(rnorm3d, true)
_CCCL_FPMP_NORM_4A(norm4d, false)
_CCCL_FPMP_NORM_4A(rnorm4d, true)

/*
 * --------------------------------------------------------------------
 * norm3d, norm4d, rnorm3d, rnorm4d (fp32mp2) at fpmp2_accuracy::high - double kernels
 * --------------------------------------------------------------------
 * The operands are converted to double, which holds a float pair exactly, and the norm is
 * taken in double (CUDA's norm3d etc. on the device), within 0.6u^2 of the norm and of its
 * reciprocal. The kernels of #12641; the other levels take the pair kernel above, which has
 * no FP64 operation and is the faster one wherever FP64 is rationed. fp64mp2 takes its double
 * fallback at every level.
 *
 * Special cases are answered on entry, as by the pair kernel (C Annex F for hypot, extended
 * to n operands and to the reciprocal): an infinite operand gives +inf (+0 for the
 * reciprocal) also next to a NaN, where the sum of the squares in double would be NaN, and
 * a NaN operand otherwise gives NaN in both limbs.
 * --------------------------------------------------------------------
 */
template <int _Np>
_CCCL_FPMP_CORE_API bool __internal_fpmp2_high_norm_special(
  const float (&__x_hi)[_Np], const bool __reciprocal, float* __res_hi, float* __res_lo) noexcept
{
  bool __has_inf = false;
  bool __has_nan = false;
  for (int __k = 0; __k < _Np; ++__k)
  {
    __has_inf = __has_inf || (::cuda::__fpmp_internal_fabs(__x_hi[__k]) == ::cuda::std::__fp_inf<float>());
    __has_nan = __has_nan || ::cuda::__fpmp_internal_isnan(__x_hi[__k]);
  }
  if (__has_inf)
  {
    *__res_hi = __reciprocal ? 0.0f : ::cuda::std::__fp_inf<float>();
    *__res_lo = 0.0f;
  }
  else if (__has_nan)
  {
    *__res_hi = *__res_lo = ::cuda::std::__fp_nan<float>();
  }
  return __has_inf || __has_nan;
}

_CCCL_FPMP_CORE_API void __internal_fpmp2_high_norm3d(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  const float __x_hi[3] = {__a_hi, __b_hi, __c_hi};
  if (::cuda::__internal_fpmp2_high_norm_special(__x_hi, false, __res_hi, __res_lo))
  {
    return;
  }
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

_CCCL_FPMP_CORE_API void __internal_fpmp2_high_norm3d(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_norm3d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_CORE_API void __internal_fpmp2_high_norm4d(
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
  const float __x_hi[4] = {__a_hi, __b_hi, __c_hi, __d_hi};
  if (::cuda::__internal_fpmp2_high_norm_special(__x_hi, false, __res_hi, __res_lo))
  {
    return;
  }
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

_CCCL_FPMP_CORE_API void __internal_fpmp2_high_norm4d(
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
  ::cuda::__internal_fpmp2_norm4d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __d_hi, __d_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_CORE_API void __internal_fpmp2_high_rnorm3d(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  const float __x_hi[3] = {__a_hi, __b_hi, __c_hi};
  if (::cuda::__internal_fpmp2_high_norm_special(__x_hi, true, __res_hi, __res_lo))
  {
    return;
  }
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

_CCCL_FPMP_CORE_API void __internal_fpmp2_high_rnorm3d(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_rnorm3d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_CORE_API void __internal_fpmp2_high_rnorm4d(
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
  const float __x_hi[4] = {__a_hi, __b_hi, __c_hi, __d_hi};
  if (::cuda::__internal_fpmp2_high_norm_special(__x_hi, true, __res_hi, __res_lo))
  {
    return;
  }
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

_CCCL_FPMP_CORE_API void __internal_fpmp2_high_rnorm4d(
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
  ::cuda::__internal_fpmp2_rnorm4d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __d_hi, __d_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_3A(high_norm3d)
_CCCL_FPMP_MATH_DISPATCH_4A(high_norm4d)
_CCCL_FPMP_MATH_DISPATCH_3A(high_rnorm3d)
_CCCL_FPMP_MATH_DISPATCH_4A(high_rnorm4d)

#  undef _CCCL_FPMP_NORM_2A
#  undef _CCCL_FPMP_NORM_3A
#  undef _CCCL_FPMP_NORM_4A

#endif // _CCCL_FPMP_USE_LIB
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_POW_H
