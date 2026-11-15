//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_IMPL_DIVSQRT_H
#define _CUDA___FP_FPMP_IMPL_DIVSQRT_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_impl_divsqrt.h - fpmp2 division, square root and reciprocal square root
    ==================================================================================================
    Per-operation implementation core split out of <cuda/__fp/fpmp_impl.h>. It carries the
    division, square root and reciprocal square root
    for the fpmp2 double-word type, for both the header-only (inline) mode and the library
    (_CCCL_FPMP_USE_LIB) mode. All shared macros, the fp128 vocabulary type, and the __fpmp_*
    error-free-transform primitives live in <cuda/__fp/fpmp_impl.h>, which this header includes.
*/

#include <cuda/__fp/fpmp_impl.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
#ifndef _CCCL_FPMP_USE_LIB
/*
 * --------------------------------------------------------------------
 * Division operations
 * --------------------------------------------------------------------
 */
/* Compute a low-accuracy quotient: the Nagai iteration of __fpmp2_div below, stopped after
the first residual correction and without the closing renormalization. The hi word is the
native quotient estimate and the lo word carries the residual a - b*q divided by b, which
puts the pair in the same precision class as the mid-accuracy quotient for about half the
arithmetic, at the cost of leaving it unnormalized - as the low accuracy level does
throughout. The residual itself is exact: q is within an ulp or two of a_hi/b_hi, so the
subtraction inside the fma cancels without rounding. The lo word is a product that only
the caller's adds consume, so it is formed by __fpmp_mul_rn_nocontract.
*/
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_low_div(
  const _FpType __a_hi,
  const _FpType __a_lo,
  const _FpType __b_hi,
  const _FpType __b_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  const _FpType __r = ::cuda::__fpmp_rcp_rn(__b_hi);
  const _FpType __q = ::cuda::__fpmp_mul_rn(__a_hi, __r);

  _FpType __e = ::cuda::__fpmp_fma_rn(__b_hi, -__q, __a_hi);
  __e         = ::cuda::__fpmp_add_rn(__a_lo, __e);
  __e         = ::cuda::__fpmp_fma_rn(__b_lo, -__q, __e);

  *__res_hi = __q;
  *__res_lo = ::cuda::__fpmp_mul_rn_nocontract(__r, __e);
} // __fpmp2_low_div

/* Compute high-accuracy quotient, using Newton-
Raphson iteration. Derived from: T. Nagai, H. Yoshida, H. Kuroda, Y. Kanada.
Fast Quadruple Precision Arithmetic Library on Parallel Computer SR11000/J2.
In Proceedings of the 8th International Conference on Computational Science,
ICCS '08, Part I, pp. 446-455.
*/
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_div(
  const _FpType __a_hi,
  const _FpType __a_lo,
  const _FpType __b_hi,
  const _FpType __b_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  _FpType __t_hi, __t_lo;
  _FpType __e, __r;
  __r    = ::cuda::__fpmp_rcp_rn(__b_hi);
  __t_hi = ::cuda::__fpmp_mul_rn(__a_hi, __r);
  __e    = ::cuda::__fpmp_fma_rn(__b_hi, -__t_hi, __a_hi);
  __t_hi = ::cuda::__fpmp_fma_rn(__r, __e, __t_hi);
  __t_lo = ::cuda::__fpmp_fma_rn(__b_hi, -__t_hi, __a_hi);
  __t_lo = ::cuda::__fpmp_add_rn(__a_lo, __t_lo);
  __t_lo = ::cuda::__fpmp_fma_rn(__b_lo, -__t_hi, __t_lo);
  __e    = ::cuda::__fpmp_mul_rn(__r, __t_lo);
  __t_lo = ::cuda::__fpmp_fma_rn(__b_hi, -__e, __t_lo);
  __t_lo = ::cuda::__fpmp_fma_rn(__r, __t_lo, __e);
  __e    = ::cuda::__fpmp_add_rn(__t_hi, __t_lo);

  *__res_lo = ::cuda::__fpmp_add_rn(__t_hi - __e, __t_lo);
  *__res_hi = __e;
} // __fpmp2_div

/*
 * --------------------------------------------------------------------
 * Accurate Division with Conditional Scaling
 * --------------------------------------------------------------------
 * This implementation handles division when operands are near the
 * denormal range by using branch-free conditional scaling.
 *
 * For division a/b:
 *   - If 'a' is small (near denormal), intermediate computations may
 *     lose precision due to denormal arithmetic
 *   - If 'b' is small, the reciprocal computation is affected
 *   - The result exponent is approximately exp(a) - exp(b)
 *
 * Strategy:
 *   1. Check if either operand is in the "danger zone" (small exponent)
 *   2. Scale the small operand up before division
 *   3. Perform division using the Nagai et al. algorithm
 *   4. Scale the result back
 *
 * The scaling keeps the reciprocal in range at BOTH ends of the exponent axis:
 * small operands near denormal, and large divisors whose reciprocal would
 * otherwise underflow to a denormal and be flushed to zero by FTZ. It costs
 * about 1.5x over the plain Nagai division and is the path taken by
 * fpmp2_accuracy::high; def/low are unaffected.
 *
 * Reference:
 *   Nagai, Yoshida, Kuroda, Kanada (2008). Fast Quadruple Precision
 *   Arithmetic Library on Parallel Computer SR11000/J2.
 *   Conditional scaling adapted from QD library techniques.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_high_div(
  const _FpType __a_hi,
  const _FpType __a_lo,
  const _FpType __b_hi,
  const _FpType __b_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  // Type-specific constants for conditional scaling
  using UintType = ::cuda::std::conditional_t<__fpmp2_is_fp32_v<_FpType>, uint32_t, uint64_t>;

  constexpr int __exp_bits      = __fpmp2_is_fp32_v<_FpType> ? 8 : 11;
  constexpr int __mant_bits     = __fpmp2_is_fp32_v<_FpType> ? 23 : 52;
  constexpr UintType __exp_mask = ((UintType(1) << __exp_bits) - 1) << __mant_bits;

  // Highest biased exponent of a finite normal (all-ones encodes inf/nan).
  constexpr int __exp_max = __fpmp2_is_fp32_v<_FpType> ? 254 : 2046;

  // The reciprocal step (__r = rcp(__sb_hi)) is the fragile part of the Nagai
  // iteration and it fails at BOTH ends of the exponent range:
  //   - divisor too SMALL (exp < low): rcp overflows / the operand is denormal;
  //   - divisor too LARGE (exp > high): rcp underflows to a denormal, which the
  //     GPU flushes to zero under FTZ, collapsing the whole quotient to 0
  //     (e.g. fp32mp2 a/b with b ~ 2^126: 1/b < 2^-126 -> denormal -> 0).
  // We pull the offending operand's exponent back into the safe band with ONE
  // bounded, always-normal power-of-two step (2^+/-K; K = 64 for float, 512 for
  // double). K exceeds the width of each danger zone, so a single step clears
  // it, and both the operand scale and the result-compensation factor stay
  // normal powers of two -- never denormal -- so the fix is itself FTZ-safe.
  // (Full exponent normalization, scaling b_hi into [1,2) with 2^-E_b, would
  //  reintroduce the bug: for b near max, 2^-E_b is denormal and FTZ-flushed.)
  constexpr int __exp_threshold_low  = __fpmp2_is_fp32_v<_FpType> ? 32 : 64;
  constexpr int __exp_threshold_high = __exp_max - __exp_threshold_low;

  // Scale factors (all normal powers of two).
  constexpr _FpType __scale_up   = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p64f) : _FpType(0x1.0p512);
  constexpr _FpType __scale_down = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p-64f) : _FpType(0x1.0p-512);

  const UintType __up_bits   = ::cuda::std::bit_cast<UintType>(__scale_up);
  const UintType __down_bits = ::cuda::std::bit_cast<UintType>(__scale_down);
  const UintType __one_bits  = ::cuda::std::bit_cast<UintType>(_FpType(1.0));

  // Extract hi-component exponents.
  const UintType __a_bits = ::cuda::std::bit_cast<UintType>(__a_hi);
  const UintType __b_bits = ::cuda::std::bit_cast<UintType>(__b_hi);
  const int __a_exp       = static_cast<int>((__a_bits & __exp_mask) >> __mant_bits);
  const int __b_exp       = static_cast<int>((__b_bits & __exp_mask) >> __mant_bits);

  // Branch-free up/down masks (-1 = active). "up" and "down" are mutually
  // exclusive: an operand cannot be both too small and too large.
  const int __a_up   = (__a_exp - __exp_threshold_low) >> 31; // a small -> scale up
  const int __a_down = (__exp_threshold_high - __a_exp) >> 31; // a large -> scale down
  const int __b_up   = (__b_exp - __exp_threshold_low) >> 31; // b small -> scale up
  const int __b_down = (__exp_threshold_high - __b_exp) >> 31; // b large -> scale down

  // Operand scale factors: up ? 2^+K : (down ? 2^-K : 1).
  const UintType __scale_a_bits =
    (__up_bits & UintType(__a_up)) | (__down_bits & UintType(__a_down)) | (__one_bits & UintType(~(__a_up | __a_down)));
  const UintType __scale_b_bits =
    (__up_bits & UintType(__b_up)) | (__down_bits & UintType(__b_down)) | (__one_bits & UintType(~(__b_up | __b_down)));
  const _FpType __scale_a = ::cuda::std::bit_cast<_FpType>(__scale_a_bits);
  const _FpType __scale_b = ::cuda::std::bit_cast<_FpType>(__scale_b_bits);

  // Scale operands (exact: power-of-two multiply).
  const _FpType __sa_hi = ::cuda::__fpmp_mul_rn(__a_hi, __scale_a);
  const _FpType __sa_lo = ::cuda::__fpmp_mul_rn(__a_lo, __scale_a);
  const _FpType __sb_hi = ::cuda::__fpmp_mul_rn(__b_hi, __scale_b);
  const _FpType __sb_lo = ::cuda::__fpmp_mul_rn(__b_lo, __scale_b);

  // Perform division on scaled operands using Nagai et al. algorithm
  _FpType __t_hi, __t_lo;
  _FpType __e, __r;
  __r    = ::cuda::__fpmp_rcp_rn(__sb_hi);
  __t_hi = ::cuda::__fpmp_mul_rn(__sa_hi, __r);
  __e    = ::cuda::__fpmp_fma_rn(__sb_hi, -__t_hi, __sa_hi);
  __t_hi = ::cuda::__fpmp_fma_rn(__r, __e, __t_hi);
  __t_lo = ::cuda::__fpmp_fma_rn(__sb_hi, -__t_hi, __sa_hi);
  __t_lo = ::cuda::__fpmp_add_rn(__sa_lo, __t_lo);
  __t_lo = ::cuda::__fpmp_fma_rn(__sb_lo, -__t_hi, __t_lo);
  __e    = ::cuda::__fpmp_mul_rn(__r, __t_lo);
  __t_lo = ::cuda::__fpmp_fma_rn(__sb_hi, -__e, __t_lo);
  __t_lo = ::cuda::__fpmp_fma_rn(__r, __t_lo, __e);
  __e    = ::cuda::__fpmp_add_rn(__t_hi, __t_lo);

  _FpType __r_hi = __e;
  _FpType __r_lo = ::cuda::__fpmp_add_rn(__t_hi - __e, __t_lo);

  // Undo the operand scaling on the result: a/b = (sa/sb) * (scale_b / scale_a).
  //   inv_scale_a = 1 / scale_a   (a up -> 2^-K, a down -> 2^+K, none -> 1)
  //   times scale_b               (b up -> 2^+K, b down -> 2^-K, none -> 1)
  // Each factor is a normal power of two; their product is exact unless the true
  // quotient itself overflows/underflows (a genuine range result, not an artifact).
  const UintType __inv_scale_a_bits =
    (__down_bits & UintType(__a_up)) | (__up_bits & UintType(__a_down)) | (__one_bits & UintType(~(__a_up | __a_down)));
  const _FpType __inv_scale_a = ::cuda::std::bit_cast<_FpType>(__inv_scale_a_bits);
  const _FpType __final_scale = ::cuda::__fpmp_mul_rn(__inv_scale_a, __scale_b);

  // Scale result back
  __r_hi = ::cuda::__fpmp_mul_rn(__r_hi, __final_scale);
  __r_lo = ::cuda::__fpmp_mul_rn(__r_lo, __final_scale);

  // Final normalization to ensure (hi, lo) invariant after scaling
  *__res_hi = ::cuda::__fpmp_fast_two_sum(__r_hi, __r_lo, __res_lo);
} // __fpmp2_high_div

/*
 * --------------------------------------------------------------------
 * Square root & reciprocal square root operations
 * --------------------------------------------------------------------
 */
/*
iteration based on equation 4 from a paper by Alan Karp and Peter Markstein,
High Precision Division and Square Root, ACM TOMS, vol. 23, no. 4, December
1997, pp. 561-589.
*/
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_rsqrt(const _FpType __a_hi, const _FpType __a_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  _FpType __z_hi, __z_lo;
  _FpType __r, __s, __e;
  _FpType __one  = static_cast<_FpType>(1.0);
  _FpType __half = static_cast<_FpType>(0.5);
  __r            = ::cuda::__fpmp_rsqrt_rn(__a_hi);
  __e            = ::cuda::__fpmp_mul_rn(__a_hi, __r);
  __s            = ::cuda::__fpmp_fma_rn(__e, -__r, __one);
  __e            = ::cuda::__fpmp_fma_rn(__a_hi, __r, -__e);
  __s            = ::cuda::__fpmp_fma_rn(__e, -__r, __s);
  __e            = ::cuda::__fpmp_mul_rn(__a_lo, __r);
  __s            = ::cuda::__fpmp_fma_rn(__e, -__r, __s);
  __e            = ::cuda::__fpmp_mul_rn(__half, __r);
  __z_hi         = ::cuda::__fpmp_mul_rn(__e, __s);
  __z_lo         = ::cuda::__fpmp_fma_rn(__e, __s, -__z_hi);
  __s            = ::cuda::__fpmp_add_rn(__r, __z_hi);
  __r            = ::cuda::__fpmp_add_rn(__r, -__s);
  __r            = ::cuda::__fpmp_add_rn(__r, __z_hi);
  __r            = ::cuda::__fpmp_add_rn(__r, __z_lo);
  __e            = ::cuda::__fpmp_add_rn(__s, __r);
  __z_lo         = ::cuda::__fpmp_add_rn(__s - __e, __r);
  __z_hi         = __e;

  *__res_hi = __z_hi;
  *__res_lo = __z_lo;
} // __fpmp2_rsqrt

/* Compute high-accuracy square root. Newton-Raphson
iteration based on equation 4 from a paper by Alan Karp and Peter Markstein,
High Precision Division and Square Root, ACM TOMS, vol. 23, no. 4, December
1997, pp. 561-589.
*/
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_sqrt(const _FpType __a_hi, const _FpType __a_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  const _FpType __zero = static_cast<_FpType>(0.0);
  const _FpType __half = static_cast<_FpType>(0.5);
  _FpType __r          = ::cuda::__fpmp_rsqrt_rn(__a_hi);
  if (__a_hi == __zero)
  {
    __r = __zero;
  }
  /* y0 = a_hi * r is within a few ulps of sqrt(a_hi) (on the device the
   * fp32 seed is the approximate rsqrt), s0 = a_hi - y0^2 its residual,
   * h = r / 2 the seed's half reciprocal root. */
  _FpType __y       = ::cuda::__fpmp_mul_rn(__a_hi, __r);
  const _FpType __h = ::cuda::__fpmp_mul_rn(__half, __r);
  _FpType __s       = ::cuda::__fpmp_fma_rn(__y, -__y, __a_hi);
  if constexpr (sizeof(_FpType) == 4)
  {
    /* fp32mp2: a_lo joins the Newton step (y ~ RN(sqrt(a_hi + a_lo)), which
     * halves the correction), then the correction (a_hi - y^2 + a_lo) is
     * scaled by the half reciprocal root refined once, h (1 + d), and the
     * pair is normalized.  Host model against binary128: 2.5 u^2 max,
     * 0.23 u^2 mean, independent of the seed's accuracy, against 5.7-52 u^2
     * before.  12 operations. */
    __y                = ::cuda::__fpmp_fma_rn(__h, __s + __a_lo, __y);
    const _FpType __d  = ::cuda::__fpmp_fma_rn(-__y, __r, static_cast<_FpType>(1.0));
    __s                = ::cuda::__fpmp_fma_rn(__y, -__y, __a_hi);
    const _FpType __hr = ::cuda::__fpmp_fma_rn(__h, __d, __h);
    const _FpType __e  = ::cuda::__fpmp_add_rn(__s, __a_lo);
    const _FpType __t  = ::cuda::__fpmp_mul_rn(__hr, __e);
    *__res_hi          = ::cuda::__fpmp_fast_two_sum(__y, __t, __res_lo);
  }
  else
  {
    /* fp64mp2: the original Nagai form, kept by choice.  The correction
     * (a_hi - y0^2 + a_lo) / (2 sqrt a) is formed as an exact product with
     * the residue of its sum, scaled by the seed's half reciprocal root,
     * and added with error-free sums.  Its error is the seed's: the
     * device's rsqrt() is close to correctly rounded and the form reads
     * 1.0 u^2 on the test suite (2.9 u^2 over 80 binades), the host's
     * two-rounding 1/sqrt(x) 5.8 u^2.  An exact-pair variant with a Newton
     * step on y (26 operations) reaches the 0.5 u^2 floor on both sides,
     * but costs 25 % of the throughput of an operation that is FP64-bound
     * on most devices; see cccl-fpdev-internal notes/fp64mp2_math.
     * 13 operations. */
    const _FpType __e      = ::cuda::__fpmp_add_rn(__s, __a_lo);
    const _FpType __tmp_lo = ::cuda::__fpmp_add_rn(__s - __e, __a_lo);
    const _FpType __t_hi   = ::cuda::__fpmp_mul_rn(__h, __e);
    _FpType __t_lo         = ::cuda::__fpmp_fma_rn(__h, __e, -__t_hi);
    __t_lo                 = ::cuda::__fpmp_fma_rn(__h, __tmp_lo, __t_lo);
    const _FpType __r1     = ::cuda::__fpmp_add_rn(__y, __t_hi);
    _FpType __s2           = ::cuda::__fpmp_add_rn(__y - __r1, __t_hi);
    __s2                   = ::cuda::__fpmp_add_rn(__s2, __t_lo);
    const _FpType __hi     = ::cuda::__fpmp_add_rn(__r1, __s2);
    *__res_lo              = ::cuda::__fpmp_add_rn(__r1 - __hi, __s2);
    *__res_hi              = __hi;
  }
} // __fpmp2_sqrt

/*
 * --------------------------------------------------------------------
 * Square root & reciprocal square root, fpmp2_accuracy::high
 * --------------------------------------------------------------------
 * Both kernels are one Newton step from a single-limb seed, in the form of Alan Karp and
 * Peter Markstein, High Precision Division and Square Root, ACM TOMS, vol. 23, no. 4,
 * December 1997, pp. 561-589, with the step's own second-order term added, so that the
 * error of the seed, squared, does not remain in the result.
 *
 * They take a positive, finite operand whose square root's low limb is a normal number
 * (the square root kernel also takes zero); the entry points below, __fpmp2_high_sqrt and
 * __fpmp2_high_rsqrt, deal with the rest. The kernels above stay branch-free and serve the
 * default and low levels, and the math functions that take a root internally.
 */

/* Compute high-accuracy reciprocal square root. With r the seed and s = 1 - a r^2,
1/sqrt(a) = r (1 - s)^(-1/2) = r + (r/2) s + (3r/8) s^2 + O(s^3). a r is the single-limb
product e and its error, to which the low limb's a_lo r is added before both enter s.
*/
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_high_rsqrt_kernel(const _FpType __a_hi, const _FpType __a_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  const _FpType __r    = ::cuda::__fpmp_rsqrt_rn(__a_hi);
  _FpType __e          = ::cuda::__fpmp_mul_rn(__a_hi, __r);
  _FpType __s          = ::cuda::__fpmp_fma_rn(__e, -__r, static_cast<_FpType>(1.0));
  __e                  = ::cuda::__fpmp_fma_rn(__a_hi, __r, -__e);
  __e                  = ::cuda::__fpmp_fma_rn(__a_lo, __r, __e);
  __s                  = ::cuda::__fpmp_fma_rn(__e, -__r, __s);
  const _FpType __h    = ::cuda::__fpmp_mul_rn(static_cast<_FpType>(0.5), __r);
  const _FpType __z_hi = ::cuda::__fpmp_mul_rn(__h, __s);
  _FpType __z_lo       = ::cuda::__fpmp_fma_rn(__h, __s, -__z_hi);
  // The second-order term (3r/8) s^2 = (3/4) s z.
  __z_lo            = ::cuda::__fpmp_fma_rn(::cuda::__fpmp_mul_rn(static_cast<_FpType>(0.75), __s), __z_hi, __z_lo);
  const _FpType __t = ::cuda::__fpmp_add_rn(__r, __z_hi);
  _FpType __u       = ::cuda::__fpmp_add_rn(::cuda::__fpmp_sub_rn(__r, __t), __z_hi);
  __u               = ::cuda::__fpmp_add_rn(__u, __z_lo);

  *__res_hi = ::cuda::__fpmp_fast_two_sum(__t, __u, __res_lo);
} // __fpmp2_high_rsqrt_kernel

/* Compute high-accuracy square root. With y ~ sqrt(a_hi) and e = a - y^2,
sqrt(a) = y + t - t^2 / (2y) + O(t^3),  t = e / (2y).  t is the product of e and the seed
h ~ 1 / (2y), corrected by its own residual e - 2y t, so that the accuracy of h does not
enter.
*/
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_high_sqrt_kernel(const _FpType __a_hi, const _FpType __a_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  _FpType __h;
  const _FpType __y    = ::cuda::__fpmp_sqrt_seed(__a_hi, &__h);
  const _FpType __s    = ::cuda::__fpmp_fma_rn(-__y, __y, __a_hi);
  const _FpType __e    = ::cuda::__fpmp_add_rn(__s, __a_lo);
  const _FpType __e_lo = ::cuda::__fpmp_add_rn(::cuda::__fpmp_sub_rn(__s, __e), __a_lo);
  const _FpType __t    = ::cuda::__fpmp_mul_rn(__h, __e);
  _FpType __t_lo       = ::cuda::__fpmp_fma_rn(-::cuda::__fpmp_add_rn(__y, __y), __t, __e);
  __t_lo               = ::cuda::__fpmp_add_rn(__t_lo, __e_lo);
  __t_lo               = ::cuda::__fpmp_fma_rn(-__t, __t, __t_lo);
  __t_lo               = ::cuda::__fpmp_mul_rn(__h, __t_lo);
  const _FpType __r    = ::cuda::__fpmp_add_rn(__y, __t);
  _FpType __u          = ::cuda::__fpmp_add_rn(::cuda::__fpmp_sub_rn(__y, __r), __t);
  __u                  = ::cuda::__fpmp_add_rn(__u, __t_lo);

  *__res_hi = ::cuda::__fpmp_fast_two_sum(__r, __u, __res_lo);
} // __fpmp2_high_sqrt_kernel

/*
 * --------------------------------------------------------------------
 * Square root and reciprocal square root over the whole range, fpmp2_accuracy::high
 * --------------------------------------------------------------------
 * The kernels above with the IEEE 754 special cases answered on entry, and with operands
 * too small for the kernels scaled by a power of four, which is exact both ways. The
 * common case is the first branch and adds nothing to the kernel but its two comparisons:
 *   sqrt(+-0) = +-0, sqrt(+inf) = +inf, sqrt(x < 0) = NaN, sqrt(NaN) = NaN
 *   rsqrt(+-0) = +-inf, rsqrt(+inf) = +0, rsqrt(x < 0) = NaN, rsqrt(NaN) = NaN
 * The square root kernel needs a high limb of at least 2^(2p) times the smallest normal
 * number (p = 24 or 53 bits), so that the residual of its Newton step and the low limb of
 * the root are normal numbers; the reciprocal's kernel needs a normal high limb, the
 * device's approximate seed reading a subnormal as zero. Against MPFR the results are
 * within 0.5 u^2 of the square root (1.9 u^2 for fp64mp2 on the device) and 2.7 u^2 of
 * its reciprocal (4 u^2 for fp32mp2 on the device), u = 2^-p.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_high_sqrt(const _FpType __a_hi, const _FpType __a_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  constexpr _FpType __max        = (::cuda::std::numeric_limits<_FpType>::max)();
  constexpr _FpType __kernel_min = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p-78f) : _FpType(0x1.0p-916);
  if (__a_hi >= __kernel_min && __a_hi <= __max)
  {
    ::cuda::__fpmp2_high_sqrt_kernel(__a_hi, __a_lo, __res_hi, __res_lo);
  }
  else if (__a_hi == _FpType(0) || __a_hi > __max)
  {
    *__res_hi = __a_hi;
    *__res_lo = _FpType(0);
  }
  else if (!(__a_hi > _FpType(0)))
  {
    *__res_hi = *__res_lo = ::cuda::std::numeric_limits<_FpType>::quiet_NaN();
  }
  else
  {
    // 2^(4p) and 2^(-2p)
    constexpr _FpType __up   = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p96f) : _FpType(0x1.0p212);
    constexpr _FpType __down = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p-48f) : _FpType(0x1.0p-106);
    _FpType __r_hi, __r_lo;
    ::cuda::__fpmp2_high_sqrt_kernel(
      ::cuda::__fpmp_mul_rn(__a_hi, __up), ::cuda::__fpmp_mul_rn(__a_lo, __up), &__r_hi, &__r_lo);
    *__res_hi = ::cuda::__fpmp_mul_rn(__r_hi, __down);
    *__res_lo = ::cuda::__fpmp_mul_rn(__r_lo, __down);
  }
}

template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_high_rsqrt(const _FpType __a_hi, const _FpType __a_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  constexpr _FpType __max = (::cuda::std::numeric_limits<_FpType>::max)();
  constexpr _FpType __inf = ::cuda::std::numeric_limits<_FpType>::infinity();
  if (__a_hi >= (::cuda::std::numeric_limits<_FpType>::min)() && __a_hi <= __max)
  {
    ::cuda::__fpmp2_high_rsqrt_kernel(__a_hi, __a_lo, __res_hi, __res_lo);
  }
  else if (__a_hi == _FpType(0))
  {
    *__res_hi = ::cuda::std::signbit(__a_hi) ? -__inf : __inf;
    *__res_lo = _FpType(0);
  }
  else if (__a_hi > __max)
  {
    *__res_hi = *__res_lo = _FpType(0);
  }
  else if (!(__a_hi > _FpType(0)))
  {
    *__res_hi = *__res_lo = ::cuda::std::numeric_limits<_FpType>::quiet_NaN();
  }
  else
  {
    // 2^(4p) and 2^(2p)
    constexpr _FpType __up      = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p96f) : _FpType(0x1.0p212);
    constexpr _FpType __up_sqrt = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p48f) : _FpType(0x1.0p106);
    _FpType __r_hi, __r_lo;
    ::cuda::__fpmp2_high_rsqrt_kernel(
      ::cuda::__fpmp_mul_rn(__a_hi, __up), ::cuda::__fpmp_mul_rn(__a_lo, __up), &__r_hi, &__r_lo);
    *__res_hi = ::cuda::__fpmp_mul_rn(__r_hi, __up_sqrt);
    *__res_lo = ::cuda::__fpmp_mul_rn(__r_lo, __up_sqrt);
  }
}

#else // _CCCL_FPMP_USE_LIB

// -- fp32 (single precision) built-in declarations --
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_div(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_mid_div(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_low_div(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_high_div(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_sqrt(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_rsqrt(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_high_sqrt(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_high_rsqrt(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept;

// -- fp64 (double precision) built-in declarations --
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_div(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_mid_div(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_low_div(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_high_div(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_sqrt(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_rsqrt(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_high_sqrt(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_high_rsqrt(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept;

// -- type-generic template declarations (dispatch to fp32/fp64) --
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void __fpmp2_div(
  const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void __fpmp2_mid_div(
  const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void __fpmp2_low_div(
  const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void __fpmp2_high_div(
  const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_sqrt(const _Tp __a_hi, const _Tp __a_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_rsqrt(const _Tp __a_hi, const _Tp __a_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_high_sqrt(const _Tp __a_hi, const _Tp __a_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_high_rsqrt(const _Tp __a_hi, const _Tp __a_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;

// -- fp32 template specializations --
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_div<float>(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_mid_div<float>(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_mid_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_low_div<float>(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_low_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_high_div<float>(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_high_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_sqrt<float>(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_sqrt(__a_hi, __a_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_rsqrt<float>(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_rsqrt(__a_hi, __a_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_high_sqrt<float>(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_high_sqrt(__a_hi, __a_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_high_rsqrt<float>(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_high_rsqrt(__a_hi, __a_lo, __res_hi, __res_lo);
}

// -- fp64 template specializations --
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_div<double>(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_mid_div<double>(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_mid_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_low_div<double>(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_low_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_high_div<double>(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_high_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_sqrt<double>(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_sqrt(__a_hi, __a_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_rsqrt<double>(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_rsqrt(__a_hi, __a_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_high_sqrt<double>(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_high_sqrt(__a_hi, __a_lo, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_high_rsqrt<double>(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_high_rsqrt(__a_hi, __a_lo, __res_hi, __res_lo);
}

#endif // _CCCL_FPMP_USE_LIB

// a / c for a scalar c that one limb holds exactly: the low and default divisions without
// the term that multiplies the divisor's zero low limb, which for finite operands leaves
// the value unchanged up to the sign of a zero low limb. high keeps the scaled division.
template <fpmp2_accuracy _Acc, typename _FpType>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE void __fpmp2_div_limb(
  const _FpType __a_hi, const _FpType __a_lo, const _FpType __c, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  if constexpr (_Acc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_div(__a_hi, __a_lo, __c, _FpType{0}, __res_hi, __res_lo);
  }
  else
  {
    const _FpType __r = ::cuda::__fpmp_rcp_rn(__c);
    _FpType __t_hi    = ::cuda::__fpmp_mul_rn(__a_hi, __r);
    _FpType __e       = ::cuda::__fpmp_fma_rn(__c, -__t_hi, __a_hi);
    if constexpr (_Acc == fpmp2_accuracy::low)
    {
      *__res_hi = __t_hi;
      *__res_lo = ::cuda::__fpmp_mul_rn_nocontract(__r, ::cuda::__fpmp_add_rn(__a_lo, __e));
    }
    else
    {
      __t_hi         = ::cuda::__fpmp_fma_rn(__r, __e, __t_hi);
      _FpType __t_lo = ::cuda::__fpmp_fma_rn(__c, -__t_hi, __a_hi);
      __t_lo         = ::cuda::__fpmp_add_rn(__a_lo, __t_lo);
      __e            = ::cuda::__fpmp_mul_rn(__r, __t_lo);
      __t_lo         = ::cuda::__fpmp_fma_rn(__c, -__e, __t_lo);
      __t_lo         = ::cuda::__fpmp_fma_rn(__r, __t_lo, __e);
      __e            = ::cuda::__fpmp_add_rn(__t_hi, __t_lo);
      *__res_lo      = ::cuda::__fpmp_add_rn(::cuda::__fpmp_sub_rn(__t_hi, __e), __t_lo);
      *__res_hi      = __e;
    }
  }
}
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_IMPL_DIVSQRT_H
