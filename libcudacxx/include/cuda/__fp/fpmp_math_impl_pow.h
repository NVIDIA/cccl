//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
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

namespace cuda::experimental
{
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
 *    pow(a, b) = exp(b * log(|a|))
 *
 * with sign fixup for a < 0 with integer b.  All three primitives
 * (log, mul, exp) are dedicated fp32mp2 - no fp64 operations in the
 * main path.
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
    const float __b_trunc = __fpmp_internal_trunc<float>(__b_hi);
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

  /* ---- (9) main path: exp(b * log(|a|)) ---- */
  float __loga_hi;
  float __loga_lo;
  __fpmp2_log<float>(__abs_a_hi, __abs_a_lo, &__loga_hi, &__loga_lo);

  float __prod_hi;
  float __prod_lo;
  __fpmp2_mul<float>(__b_hi, __b_lo, __loga_hi, __loga_lo, &__prod_hi, &__prod_lo);

  float __t_hi;
  float __t_lo;
  __fpmp2_exp<float>(__prod_hi, __prod_lo, &__t_hi, &__t_lo);

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
 *   5. Initial single-precision approximation:
 *         s = fast_exp2(third * fast_log2(r_hi))            (~23 bits)
 *   6. One Halley iteration in fp32mp2 arithmetic
 *      (cubic convergence -> ~70 theoretical bits, capped by the
 *       ~46-bit fp32mp2 precision):
 *         t_new = t + t * (r - t^3) / (2 t^3 + r)
 *      The numerator (r - t^3) cancels catastrophically (t^3 ~= r),
 *      so it is evaluated with the accurate fp32mp2 fma:
 *         numer = fma<accurate>(-t^2, t, r)
 *      The denominator is well-conditioned (~3 r); a single-precision
 *      reciprocal of denom.hi() is enough -- the resulting correction
 *      contributes only at the 2^-46 level after multiplication by t.
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
  using __ffloat = fp32mp2_low;

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
  const int __nexpo = __fpmp_fp2int_rn(__third_f * static_cast<float>(__expo - 126));

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

  /* Initial cbrt approximation via the SFU lg2/ex2 pair (~23 bits). */
  const float __s = __fpmp_fast_exp2(__third_f * __fpmp_fast_log2(__r_hi));

  /* Halley refinement in fp32mp2:  t_new = t + t * (r - t^3) / (2 t^3 + r).
   *
   * The catastrophic cancellation in (r - t^3) is handled by an
   * accurate fma:  fma<accurate>(-t^2, t, r) computes r - t^2 * t
   * with a single rounding error followed by an exact correction,
   * preserving the small difference that drives the iteration.
   */
  const __ffloat __r(__r_hi, __r_lo);
  const __ffloat __t(__s);

  const __ffloat __t2 = __t * __t; // t^2
  // numer = r - t^3 (computed as fma(-t^2, t, r) with accurate ff fma)
  const __ffloat __numer = fma<fpmp2_accuracy::high>(-__t2, __t, __r);
  // denom = 2 t^3 + r ~= 3 r, well-conditioned so fast add suffices
  const __ffloat __t3    = __t2 * __t;
  const __ffloat __denom = (__t3 + __t3) + __r;

  /* Single-precision reciprocal of denom.hi() is enough: the
   * correction u_corr ~ 2^-23 contributes t * u_corr ~ 2^-46 to
   * t_new -- exactly fp32mp2 precision. */
  const float __inv_denom = __fpmp_rcp_rn(__denom.hi());
  const __ffloat __u_corr = __numer * __inv_denom;
  const __ffloat __t_new  = __t + __t * __u_corr;

  /* Scale back by 2^(nexpo - denorm_div3) via an exact power-of-two
   * float multiply.  back_shift stays in a range that keeps the
   * scale factor a normal float for all valid float inputs
   * (biased exponent is always in [77, 170]). */
  const int __back_shift   = __nexpo - __denorm_div3;
  const float __scale_back = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __back_shift) * __exp_shift));
  float __t_hi_back        = __t_new.hi() * __scale_back;
  float __t_lo_back        = __t_new.lo() * __scale_back;

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
  __fpmp_fp128 __res = _CCCL_FPMP_CBRTQ(__fpmp2_to_quad(__x_hi, __x_lo));
  __fpmp2_from_quad(__res, __res_hi, __res_lo);
#  else
  double __res = ::cuda::std::cbrt(__fpmp2_to_double(__x_hi, __x_lo));
  __fpmp2_from_double(__res, __res_hi, __res_lo);
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
 *   5. Initial single-precision approximation:
 *         s = fast_exp2(-third * fast_log2(r_hi))                 ~= 1/cbrt(r_hi) (~23 bits)
 *   6. One Halley iteration in fp32mp2 (cubic convergence ->
 *      ~70 theoretical bits, capped by ~46-bit fp32mp2 precision):
 *
 *          u     = 1 - r * t^3              (catastrophic cancellation)
 *          t_new = t * (1 + u/3 + (2/9) u^2)
 *
 *      The residual u is the dominant source of error and is computed
 *      with the accurate fp32mp2 fma:
 *         u = fma<accurate>(-r, t^3, 1)
 *      preserving the small difference that drives the iteration.
 *      The Halley quadratic (1/3 + (2/9) u) is well conditioned (no
 *      cancellation), so a single fast fma is sufficient there.
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
  using __ffloat = fp32mp2_low;

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
  const int __nexpo = __fpmp_fp2int_rn(__third_f * static_cast<float>(__expo - 126));

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

  /* Initial 1/cbrt approximation via the SFU lg2/ex2 pair (~23 bits). */
  const float __s = __fpmp_fast_exp2(-__third_f * __fpmp_fast_log2(__r_hi));

  /* Halley refinement in fp32mp2:  t_new = t * (1 + u/3 + (2/9) u^2)
   * with u = 1 - r * t^3.
   *
   * The cancellation in (1 - r * t^3) is the only sensitive step;
   * everything else (t^2, t^3, the Halley quadratic, the final
   * combination) is well conditioned in fast fp32mp2 arithmetic.
   */
  const __ffloat __r(__r_hi, __r_lo);
  const __ffloat __t(__s);

  const __ffloat __t2 = __t * __t; // t^2
  const __ffloat __t3 = __t2 * __t; // t^3

  // u = 1 - r*t^3 (accurate fma to preserve catastrophic cancellation)
  const __ffloat __u = fma<fpmp2_accuracy::high>(-__r, __t3, 1.0f);

  // Halley quadratic factor:  hf = 1/3 + (2/9) u   (no cancellation)
  const __ffloat __hf = fma<fpmp2_accuracy::def>(__two_ninths_f, __u, __third_f);

  // delta = u * t * hf,  then  t_new = t + delta
  const __ffloat __ut    = __u * __t;
  const __ffloat __t_new = __t + __hf * __ut;

  /* Scale back by 2^(-nexpo + denorm_div3) via an exact power-of-two
   * float multiply.  back_shift stays in [-43, +49] for all valid
   * float inputs, so the biased exponent is always in [84, 176]. */
  const int __back_shift   = -__nexpo + __denorm_div3;
  const float __scale_back = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __back_shift) * __exp_shift));
  float __t_hi_back        = __t_new.hi() * __scale_back;
  float __t_lo_back        = __t_new.lo() * __scale_back;

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
  double __xd = __fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (__fpmp2_from_double(::rcbrt(__xd), __res_hi, __res_lo);),
                    (__fpmp2_from_double(1.0 / ::cuda::std::cbrt(__xd), __res_hi, __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_1A(rcbrt)

/* hypot has no dedicated fp32mp2 kernel: the fallback macro composes it over
 * double, and the fp64mp2 body follows. */
/*
 * ====================================================================
 * hypot(x, y) - sqrt(x^2 + y^2) without spurious overflow
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Hypotenuse hypot(x, y) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_MATH_FALLBACK_2A(hypot)

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
  __fpmp2_from_double(
    ::cuda::std::hypot(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo);
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
  double __ad = __fpmp2_to_double(__a_hi, __a_lo), __bd = __fpmp2_to_double(__b_hi, __b_lo),
         __cd = __fpmp2_to_double(__c_hi, __c_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (__fpmp2_from_double(::norm3d(__ad, __bd, __cd), __res_hi, __res_lo);),
    (__fpmp2_from_double(::cuda::std::sqrt(__ad * __ad + __bd * __bd + __cd * __cd), __res_hi, __res_lo);))
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
  double __ad = __fpmp2_to_double(__a_hi, __a_lo), __bd = __fpmp2_to_double(__b_hi, __b_lo),
         __cd = __fpmp2_to_double(__c_hi, __c_lo), __dd = __fpmp2_to_double(__d_hi, __d_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (__fpmp2_from_double(::norm4d(__ad, __bd, __cd, __dd), __res_hi, __res_lo);),
    (__fpmp2_from_double(::cuda::std::sqrt(__ad * __ad + __bd * __bd + __cd * __cd + __dd * __dd), __res_hi, __res_lo);))
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
  double __ad = __fpmp2_to_double(__a_hi, __a_lo), __bd = __fpmp2_to_double(__b_hi, __b_lo),
         __cd = __fpmp2_to_double(__c_hi, __c_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (__fpmp2_from_double(::rnorm3d(__ad, __bd, __cd), __res_hi, __res_lo);),
    (__fpmp2_from_double(1.0 / ::cuda::std::sqrt(__ad * __ad + __bd * __bd + __cd * __cd), __res_hi, __res_lo);))
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
  double __ad = __fpmp2_to_double(__a_hi, __a_lo), __bd = __fpmp2_to_double(__b_hi, __b_lo),
         __cd = __fpmp2_to_double(__c_hi, __c_lo), __dd = __fpmp2_to_double(__d_hi, __d_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (__fpmp2_from_double(::rnorm4d(__ad, __bd, __cd, __dd), __res_hi, __res_lo);),
    (__fpmp2_from_double(
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
 * Reciprocal hypotenuse rhypot(x, y) (fp32mp2) - double fallback
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
  using __mp2_t = fpmp2<float>;
  double __r    = 0.0;
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (__r = ::rhypot(static_cast<double>(__mp2_t(__x_hi, __x_lo)), static_cast<double>(__mp2_t(__y_hi, __y_lo)));),
    (__r = 1.0
         / ::cuda::std::hypot(static_cast<double>(__mp2_t(__x_hi, __x_lo)),
                              static_cast<double>(__mp2_t(__y_hi, __y_lo)));))
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
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
    (__fpmp2_from_double(
       ::rhypot(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo);),
    (__fpmp2_from_double(1.0 / ::cuda::std::hypot(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)),
                         __res_hi,
                         __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_2A(rhypot)

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_POW_H
