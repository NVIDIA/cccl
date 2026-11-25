//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_MANIP_H
#define _CUDA___FP_FPMP_MATH_IMPL_MANIP_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
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
#include <cuda/std/climits>
#include <cuda/std/cmath>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
#ifndef _CCCL_FPMP_USE_LIB

/*
 * ====================================================================
 * ldexp(x, n) - x * 2^n
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Scaling ldexp(x, n) = x * 2^n (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * Each limb is multiplied by 2^n, which is exact but for a subnormal product, so no pair
 * arithmetic is needed and, for fp32mp2, no FP64 operation.
 *
 * Where 2^n is a normal number of the limb format (-126 <= n <= 127 for fp32mp2,
 * -1022 <= n <= 1023 for fp64mp2) that is one multiplication per limb, and a result in the
 * subnormal range is rounded once.  Any other n goes to the limb format's own ldexp, which
 * splits the factor and rounds once as well.
 *
 * The special operands come out as IEEE-754 has it:
 *
 *   ldexp(+-0, n)   ->  +-0
 *   ldexp(+-inf, n) ->  +-inf
 *   ldexp(NaN, n)   ->  NaN
 *   ldexp(x, n)     ->  +-inf  when the result overflows
 *   ldexp(x, n)     ->  +-0    when it underflows
 * --------------------------------------------------------------------
 */
// 2^n for an n in the normal exponent range of the limb format.
template <typename _FpType>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE _FpType __fpmp_pow2(const int __n) noexcept
{
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    return ::cuda::std::bit_cast<float>(static_cast<uint32_t>(__n + 127) << 23);
  }
  else
  {
    return ::cuda::std::bit_cast<double>(static_cast<uint64_t>(__n + 1023) << 52);
  }
}

template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_ldexp(const _FpType __x_hi, const _FpType __x_lo, int __n, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  constexpr int __e_min = __fpmp2_is_fp32_v<_FpType> ? -126 : -1022;
  constexpr int __e_max = __fpmp2_is_fp32_v<_FpType> ? 127 : 1023;

  _FpType __hi, __lo;
  if (__n >= __e_min && __n <= __e_max)
  {
    const _FpType __s = ::cuda::__fpmp_pow2<_FpType>(__n);
    __hi              = ::cuda::__fpmp_mul_rn(__x_hi, __s);
    __lo              = ::cuda::__fpmp_mul_rn(__x_lo, __s);
  }
  else
  {
    __hi = ::cuda::std::ldexp(__x_hi, __n);
    __lo = ::cuda::std::ldexp(__x_lo, __n);
  }

  /* Overflow: the low limb is meaningless once the high one is infinite, and it can
   * still be finite, since lo is about one part in 2^24 (2^53) of hi.  Collapse to
   * (+-inf, 0), which is what a conversion from an overflowing scalar produces. */
  *__res_hi = __hi;
  *__res_lo = ::cuda::std::isinf(__hi) ? _FpType(0) : __lo;
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
 * `ldexp(x, n)`, so we simply forward to the ldexp implementation
 * above.  No fp64 round-trip is required.
 * --------------------------------------------------------------------
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_scalbn(const _FpType __x_hi, const _FpType __x_lo, int __n, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  ::cuda::__fpmp2_ldexp<_FpType>(__x_hi, __x_lo, __n, __res_hi, __res_lo);
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
  const bool __flip = (::cuda::std::signbit(__x_hi) != ::cuda::std::signbit(__y_hi));
  *__res_hi         = __flip ? -__x_hi : __x_hi;
  *__res_lo         = __flip ? -__x_lo : __x_lo;
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
  ::cuda::__fpmp2_from_double(
    ::cuda::std::nextafter(::cuda::__fpmp2_to_double(__x_hi, __x_lo), ::cuda::__fpmp2_to_double(__y_hi, __y_lo)),
    __res_hi,
    __res_lo);
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
 * For a finite nonzero hi the exponent is frexp's minus one, computed on the limb's
 * own type.  hi alone settles zero, infinity and NaN, where ::ilogb keeps whatever the
 * platform reports (FP_ILOGB0 and its siblings); only that path asks through a double,
 * which is exact for either element type.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API int __fpmp2_ilogb(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  int __hi_exp      = 0;
  const _FpType __m = ::cuda::std::frexp(__x_hi, &__hi_exp);
  if ((__m == _FpType(0)) || !::cuda::std::isfinite(__x_hi))
  {
    /* The C99 sentinels directly (FP_ILOGB0 for a zero, FP_ILOGBNAN for a NaN, INT_MAX for an
     * infinity), without the detour through a double. */
    return (__m == _FpType(0)) ? FP_ILOGB0 : ((__x_hi != __x_hi) ? FP_ILOGBNAN : INT_MAX);
  }
  const bool __below_power_of_two = (::cuda::__fpmp_internal_fabs(__m) == _FpType(0.5)) && (__x_lo != _FpType(0))
                                 && (::cuda::std::signbit(__x_hi) != ::cuda::std::signbit(__x_lo));

  return __below_power_of_two ? __hi_exp - 2 : __hi_exp - 1;
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
  // exponent where an infinity or a NaN belongs: logb(+-0) = -inf, logb(+-inf) = +inf,
  // logb(NaN) = NaN.  Either way the result is an integer, so the low limb stays zero.
  if (::cuda::std::isfinite(__x_hi) && __x_hi != _FpType(0))
  {
    *__res_hi = static_cast<_FpType>(::cuda::__fpmp2_ilogb<_FpType>(__x_hi, __x_lo));
  }
  else
  {
    *__res_hi = (__x_hi == _FpType(0)) ? -::cuda::std::__fp_inf<_FpType>()
                                       : ((__x_hi != __x_hi) ? __x_hi : ::cuda::std::__fp_inf<_FpType>());
  }
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
  ::cuda::__fpmp2_ldexp<_FpType>(__x_hi, __x_lo, __ni, __res_hi, __res_lo);
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
 * For 0, frexp reports an exponent of 0, and scaling lo by 2^0 leaves the pair as it was.
 * For infinity and NaN, frexp returns hi unchanged and the exponent it reports is
 * unspecified.  The exponent is clamped before lo is scaled, so a zero lo stays zero.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_frexp(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo, int* __nptr) noexcept
{
  constexpr int __e_min = __fpmp2_is_fp32_v<_FpType> ? -126 : -1022;
  constexpr int __e_max = __fpmp2_is_fp32_v<_FpType> ? 127 : 1023;

  _FpType __m_hi = ::cuda::std::frexp(__x_hi, __nptr);
  /* lo * 2^k with k = -*nptr, rounded once.  k lies in [-128, 148] ([-1024, 1073]).
   * One normal power of two, 2^k1, covers k unless hi is in the top two binades or below
   * 2^-128 (2^-1024).  In those cases an extra factor 2^(k - k1) is applied first.  That
   * factor is exact: it scales lo up, or down by at most two binades.  A lo that it takes
   * into the subnormal range ends as zero either way.
   *
   * The exponent that the limb format's frexp stores for an infinity or a NaN is unspecified.
   * Clamping it to twice the normal range keeps both factors finite.  The zero low limb of
   * an infinity therefore stays zero. */
  const int __e  = (*__nptr < -2 * __e_max) ? -2 * __e_max : ((*__nptr > -2 * __e_min) ? -2 * __e_min : *__nptr);
  const int __k  = -__e; // negated after the clamp: -*__nptr overflows for INT_MIN
  const int __k1 = (__k < __e_min) ? __e_min : ((__k > __e_max) ? __e_max : __k);
  _FpType __m_lo = __x_lo;
  if (__k != __k1)
  {
    __m_lo = ::cuda::__fpmp_mul_rn(__m_lo, ::cuda::__fpmp_pow2<_FpType>(__k - __k1));
  }
  __m_lo = ::cuda::__fpmp_mul_rn(__m_lo, ::cuda::__fpmp_pow2<_FpType>(__k1));

  if (::cuda::__fpmp_internal_fabs(__m_hi) == _FpType(0.5) && __m_lo != _FpType(0)
      && (::cuda::std::signbit(__m_hi) != ::cuda::std::signbit(__m_lo)))
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
  if (!(::cuda::__fpmp_internal_fabs(__x_hi) < ::cuda::std::__fp_inf<_FpType>()))
  {
    /* modf(+-inf) = (+-0, +-inf); a NaN is a NaN in both parts (C99 F.10.3.12). */
    const _FpType __nan = __x_hi + __x_lo;
    *__iptr_hi          = __x_hi;
    *__iptr_lo          = _FpType(0);
    *__res_hi           = (__x_hi == __x_hi) ? ::cuda::std::copysign(_FpType(0), __x_hi) : __nan;
    *__res_lo           = (__x_hi == __x_hi) ? _FpType(0) : __nan;
    return;
  }
  ::cuda::__fpmp2_trunc<_FpType>(__x_hi, __x_lo, __iptr_hi, __iptr_lo);
  ::cuda::__fpmp2_high_sub<_FpType>(__x_hi, __x_lo, *__iptr_hi, *__iptr_lo, __res_hi, __res_lo);

  if (*__res_hi == _FpType(0))
  {
    *__res_hi = ::cuda::std::signbit(__x_hi) ? -_FpType(0) : _FpType(0);
    *__res_lo = _FpType(0);
  }
}

#endif // _CCCL_FPMP_USE_LIB
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_MANIP_H
