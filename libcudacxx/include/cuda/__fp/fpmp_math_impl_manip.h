//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_MANIP_H
#define _CUDA___FP_FPMP_MATH_IMPL_MANIP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_manip.h - fpmp2 floating-point manipulation (frexp, ldexp, modf, scalbn, ilogb, logb, nextafter,
   copysign, fabs)
    ==================================================================================================
*/

#include <cuda/__fp/fpmp_math_impl.h>
// modf splits the pair with trunc.
#include <cuda/__fp/fpmp_math_impl_nearint.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * ====================================================================
 * ldexp(x, n) - x * 2^n
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Scaling ldexp(x, n) = x * 2^n (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * Replaces the legacy fp64 round-trip with an all-fp32 implementation:
 * the 2^n factor is built directly from its biased fp32 exponent via
 * bit-cast and multiplied into x.  No native double is ever touched,
 * so the routine runs at full fp32 throughput on GPUs where fp64 is
 * 1:32 or worse.
 *
 * Algorithm -- 3-piece split (vs. the 2-piece used by
 * __internal_fpmp2_ldexp2 for the bounded-n exp2/exp10 callers):
 *
 *   2^n  =  2^k * 2^k * 2^(n - 2k),  k = n/3
 *
 * Each factor's biased exponent is clamped to [1, 254] (fp32 normal
 * range), so every scale is finite and strictly positive.  This
 * keeps the mul chain IEEE-clean for the special inputs that show up
 * in `ldexp` corner cases:
 *
 *   ldexp(0,   n) ->  +-0    (0 * finite positive = +-0, never NaN)
 *   ldexp(+-inf, n) ->  +-inf    (inf * finite positive = inf)
 *   ldexp(NaN, n) ->  NaN  (NaN propagates)
 *   ldexp(x,   n) -> +-inf    when |n| is huge enough to overflow fp32
 *   ldexp(x,   n) -> +-0    when |n| is huge enough to underflow fp32
 *
 * Saturation: n is pre-clamped to +-300.  Even a denormal x (|x| >=
 * 2^-149) scaled by 2^300 overflows to +inf, and any finite x scaled by
 * 2^-300 underflows to +-0 -- so further-out n values would produce the
 * same result and saturating is exact, not lossy.
 *
 * The fp64mp2 branch (compile-time-selected when FpType == double)
 * forwards to `::cuda::std::ldexp(double, int)`; fp64 hardware handles this in
 * one instruction and there is no fp64 cost concern on machines that
 * have fp64mp2 enabled.
 * --------------------------------------------------------------------
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_ldexp(const _FpType __x_hi, const _FpType __x_lo, int __n, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    using __ffloat = fp32mp2_low;

    /* Saturate |n| to +-300.  Any |n| larger than this is provably
     * monotone in the final result (overflow or underflow) for
     * every finite fp32 input, so clamping does not lose info. */
    if (__n > 300)
    {
      __n = 300;
    }
    if (__n < -300)
    {
      __n = -300;
    }

    const int __k = __n / 3;
    int __ek1     = 127 + __k;
    int __ek2     = 127 + __k;
    int __ek3     = 127 + (__n - 2 * __k);
    if (__ek1 < 1)
    {
      __ek1 = 1;
    }
    if (__ek2 < 1)
    {
      __ek2 = 1;
    }
    if (__ek3 < 1)
    {
      __ek3 = 1;
    }
    if (__ek1 > 254)
    {
      __ek1 = 254;
    }
    if (__ek2 > 254)
    {
      __ek2 = 254;
    }
    if (__ek3 > 254)
    {
      __ek3 = 254;
    }
    const float __s1 = ::cuda::std::bit_cast<float>(static_cast<unsigned>(__ek1) << 23);
    const float __s2 = ::cuda::std::bit_cast<float>(static_cast<unsigned>(__ek2) << 23);
    const float __s3 = ::cuda::std::bit_cast<float>(static_cast<unsigned>(__ek3) << 23);

    const __ffloat __result = __ffloat(__x_hi, __x_lo) * __s1 * __s2 * __s3;

    *__res_hi = __result.hi();
    *__res_lo = __result.lo();
  }
  else
  {
    /* fp64mp2 path: scale each limb by the same power of two.  That is exact for
     * both of them, so the pair keeps its low limb; the double round-trip this
     * replaces discarded lo for every fp64mp2 value, ldexp(1 + 2^-70, 4) coming
     * back as exactly 16. */
    *__res_hi = ::cuda::std::ldexp(__x_hi, __n);
    *__res_lo = ::cuda::std::ldexp(__x_lo, __n);
  }

  /* Overflow, for either element type: the low limb is meaningless once the high
   * one is infinite, and leaving it as it comes out (inf - inf = NaN from the
   * fp32mp2 scaling, or a still-finite value from the fp64mp2 one, since lo is
   * about 2^-53 of hi) makes hi + lo NaN instead of the infinity this is
   * documented to return.  Collapse to (+-inf, 0), which is what a conversion
   * from an overflowing scalar produces. */
  if (::cuda::std::isinf(static_cast<double>(*__res_hi)))
  {
    *__res_lo = _FpType(0);
  }
}

/*
 * ====================================================================
 * scalbn(x, n) - x * FLT_RADIX^n
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Scaling scalbn(x, n) = x * FLT_RADIX^n (fp32mp2 and fp64mp2) - forwards to ldexp
 * --------------------------------------------------------------------
 * For IEEE 754 binary formats (FLT_RADIX == 2 -- true on every
 * platform we target) `scalbn(x, n)` is bit-identical to
 * `ldexp(x, n)`, so we simply forward to the dedicated fp32mp2
 * ldexp implementation above and avoid duplicating the 3-piece
 * bit-cast scaling logic.  No fp64 round-trip is required.
 * --------------------------------------------------------------------
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_scalbn(const _FpType __x_hi, const _FpType __x_lo, int __n, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  __fpmp2_ldexp<_FpType>(__x_hi, __x_lo, __n, __res_hi, __res_lo);
}

/*
 * ====================================================================
 * fabs(x) - absolute value
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Absolute value fabs(x) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * fabs: |x|.  For a normalized (hi, lo) pair the value's sign is the sign
 * of `hi`, and `|hi| > |lo|`, so flipping both components when `hi` is
 * negative yields the absolute value while preserving the residual `lo`
 * exactly.  We use ::fabs on `hi` to get IEEE-correct handling of -0 and
 * NaN, and use the sign of the original `hi` to decide whether to flip
 * `lo`.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_fabs(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = ::cuda::std::fabs(__x_hi);
  *__res_lo = (__x_hi < _FpType(0)) ? -__x_lo : __x_lo;
}

/*
 * ====================================================================
 * copysign(x, y) - magnitude of x with the sign of y
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Sign transfer copysign(x, y) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * copysign: |x| carrying the sign of y.  Like fabs, this is exact on the pair and
 * needs no arithmetic at all: the sign of a renormalized value is the sign of its
 * high limb, so the operation is a conditional negation of both limbs.  y's low
 * limb cannot disagree with its high limb about the sign, so it is not consulted.
 *
 * signbit rather than a comparison against zero, so that a y of -0 is honored.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_copysign(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  [[maybe_unused]] const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  const bool __flip =
    (::cuda::std::signbit(static_cast<double>(__x_hi)) != ::cuda::std::signbit(static_cast<double>(__y_hi)));
  *__res_hi = __flip ? -__x_hi : __x_hi;
  *__res_lo = __flip ? -__x_lo : __x_lo;
}

/* nextafter has no dedicated fp32mp2 kernel: the fallback macro composes it over
 * double, and the fp64mp2 body follows. */
/*
 * ====================================================================
 * nextafter(x, y) - next representable value toward y
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Next representable nextafter(x, y) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_MATH_FALLBACK_2A(nextafter)

/*
 * --------------------------------------------------------------------
 * Next representable nextafter(x, y) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_nextafter(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fpmp2_from_double(
    ::cuda::std::nextafter(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_2A(nextafter)

/*
 * ====================================================================
 * ilogb(x) - exponent of x as an int
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Exponent ilogb(x) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * ilogb / logb: the exponent of the pair, which is the exponent of hi except in one
 * case. Where hi is a power of two and lo has the opposite sign the value sits just
 * below that power, so the exponent is one lower - the same correction frexp needs
 * further down. A widened double cannot see that case, and rounds the mirror one the
 * wrong way as well, reporting the power of two that the pair sits just short of.
 *
 * hi alone settles zero, infinity and NaN, so ::ilogb keeps whatever the platform
 * reports for them, FP_ILOGB0 and its siblings included. It is asked through a double,
 * which is exact for either element type and saves this from depending on a float
 * overload of ilogb being declared in device code.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API int __fpmp2_ilogb(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  const int __e = ::cuda::std::ilogb(static_cast<double>(__x_hi));

  int __hi_exp      = 0;
  const _FpType __m = ::cuda::std::frexp(__x_hi, &__hi_exp);
  const bool __below_power_of_two =
    (__fpmp_internal_fabs(__m) == _FpType(0.5)) && (__x_lo != _FpType(0))
    && (::cuda::std::signbit(static_cast<double>(__x_hi)) != ::cuda::std::signbit(static_cast<double>(__x_lo)));

  return __below_power_of_two ? __e - 1 : __e;
}

/*
 * ====================================================================
 * logb(x) - exponent of x as a floating-point value
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Exponent logb(x) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_logb(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  // The branch earns its keep: ilogb reports zero and the non-finite cases with ordinary
  // integer sentinels, so widening one of those would hand back a plausible finite
  // exponent where an infinity or a NaN belongs. ::logb has the right answers for them.
  // Either way the result is an integer, so the low limb stays zero.
  *__res_hi = (::cuda::std::isfinite(static_cast<double>(__x_hi)) && __x_hi != _FpType(0))
              ? static_cast<_FpType>(__fpmp2_ilogb<_FpType>(__x_hi, __x_lo))
              : static_cast<_FpType>(::cuda::std::logb(static_cast<double>(__x_hi)));
  *__res_lo = _FpType(0);
}
/*
 * ====================================================================
 * scalbln(x, n) - x * FLT_RADIX^n with a long exponent
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Scaling scalbln(x, n) (fp32mp2 and fp64mp2) - forwards to ldexp
 * --------------------------------------------------------------------
 *  scalbln differs from scalbn only in taking a long.  Clamping to int is exact
 * rather than lossy: every |n| past this point already saturates to infinity or
 * to zero for any finite input of either element type.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_scalbln(const _FpType __x_hi, const _FpType __x_lo, long int __n, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  const int __ni = (__n > 100000L) ? 100000 : ((__n < -100000L) ? -100000 : static_cast<int>(__n));
  __fpmp2_ldexp<_FpType>(__x_hi, __x_lo, __ni, __res_hi, __res_lo);
}

/*
 * ====================================================================
 * frexp(x, &n) - mantissa in [1/2, 1) and a power of two
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Decomposition frexp(x, &n) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * frexp: split into a mantissa in [1/2, 1) and a power of two.  Exact on the pair:
 * hi fixes the exponent and lo is scaled by the same power of two.
 *
 * One pair-specific correction.  When hi is an exact power of two and lo has the
 * opposite sign, the value sits just *below* that power of two, so scaling by hi's
 * exponent leaves the pair just below 1/2 and the exponent one too large.  Doubling
 * both limbs and decrementing the exponent is exact and puts the mantissa back in
 * range.  A single double cannot express the situation at all, which is why the
 * placeholder never had to think about it.
 *
 * For 0, infinity and NaN, frexp reports an exponent of 0 and returns hi unchanged,
 * and scaling lo by 2^0 leaves the pair as it was.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_frexp(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo, int* __nptr) noexcept
{
  _FpType __m_hi = ::cuda::std::frexp(__x_hi, __nptr);
  _FpType __m_lo = ::cuda::std::ldexp(__x_lo, -*__nptr);

  if (__fpmp_internal_fabs(__m_hi) == _FpType(0.5) && __m_lo != _FpType(0)
      && (::cuda::std::signbit(static_cast<double>(__m_hi)) != ::cuda::std::signbit(static_cast<double>(__m_lo))))
  {
    __m_hi += __m_hi;
    __m_lo += __m_lo;
    --*__nptr;
  }

  *__res_hi = __m_hi;
  *__res_lo = __m_lo;
}

/*
 * ====================================================================
 * modf(x, &i) - integral and fractional parts
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Decomposition modf(x, &i) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * modf: break the pair into its integral and fractional parts.
 *
 * trunc is exact on the pair and the subtraction that follows cancels exactly, so the
 * two results still add back up to x - which is the whole contract of modf, and what a
 * detour through a single double breaks as soon as the pair carries more than 53 bits.
 *
 * The accurate subtraction rather than the Dekker one: the low limbs can cancel into a
 * fraction that needs both of its own limbs (x = 2^25 - 2^-39 leaves 1 - 2^-39), which
 * only the variant running a two_sum on them keeps.
 *
 * Non-finite input is left to fall through, as elsewhere in fpmp: an infinity comes back
 * as a NaN from the inf - inf, not as a plausible finite answer. What does need saying is
 * that an integral x cancels to +0 where C asks for a zero carrying the sign of x.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_modf(
  const _FpType __x_hi,
  const _FpType __x_lo,
  _FpType* __res_hi,
  _FpType* __res_lo,
  _FpType* __iptr_hi,
  _FpType* __iptr_lo) noexcept
{
  __fpmp2_trunc<_FpType>(__x_hi, __x_lo, __iptr_hi, __iptr_lo);
  __fpmp2_high_sub<_FpType>(__x_hi, __x_lo, *__iptr_hi, *__iptr_lo, __res_hi, __res_lo);

  if (*__res_hi == _FpType(0))
  {
    *__res_hi = ::cuda::std::signbit(static_cast<double>(__x_hi)) ? -_FpType(0) : _FpType(0);
    *__res_lo = _FpType(0);
  }
}

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_MANIP_H
