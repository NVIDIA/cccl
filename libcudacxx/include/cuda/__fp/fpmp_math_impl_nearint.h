//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_NEARINT_H
#define _CUDA___FP_FPMP_MATH_IMPL_NEARINT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_nearint.h - fpmp2 nearest-integer and remainder functions (ceil/floor/trunc/round/rint, fmod,
   remainder, remquo)
    ==================================================================================================
*/

#include <cuda/__fp/fpmp_math_impl.h>
// Sibling families whose kernels this family calls (fmin/fmax/min/max are used here).
#include <cuda/__fp/fpmp_math_impl_classify.h>
#include <cuda/std/__floating_point/constants.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * --------------------------------------------------------------------
 * fmod / remainder (fp32mp2) - dedicated implementations
 * --------------------------------------------------------------------
 * Decompose each operand into a 64-bit integer mantissa M and a
 * binary exponent E with  value = M * 2^E,  then reduce
 *   Mx * 2^(Ex-Ey)  (mod My)
 * with exact 64-bit integer arithmetic, chunking the left shift so
 * the running value never exceeds 2^63.  No native fp64 is touched,
 * so this runs at full fp32 throughput on GPUs where double is 1:32
 * or worse, and it is *more* accurate than the old
 * `::cuda::std::fmod(double, double)` round-trip for fp32mp2 values whose two
 * limbs straddle a wide exponent gap (the double cast collapsed lo
 * into hi).
 *
 * A renormalized fp32mp2 carries at most ~48 significant bits from
 * the top of hi down to the bottom of lo, so one uint64 mantissa
 * captures the whole value exactly in the common case.  When the
 * limbs are more than ~48 bits apart the far-below-lo bits are
 * dropped (fp2int_rn rounding of the scaled lo limb), contributing
 * only ~|lo| of absolute error -- negligible relative to a result of
 * magnitude < |y|.  Denormal results round once through the
 * power-of-two reconstruction; we do not chase sub-ulp denormal
 * accuracy here.
 * --------------------------------------------------------------------
 */

/* Scale a float by 2^s using two power-of-two factors, so the
 * intermediate stays in range for the |s| we feed it here. */
_CCCL_FPMP_CORE_API float __internal_fpmp2_scale2_scalar(float __v, int __s) noexcept
{
  const int __s1   = __s >> 1; /* floor(s/2) */
  const int __s2   = __s - __s1;
  const float __f1 = ::cuda::std::bit_cast<float>(static_cast<uint32_t>(127 + __s1) << 23);
  const float __f2 = ::cuda::std::bit_cast<float>(static_cast<uint32_t>(127 + __s2) << 23);
  return __v * __f1 * __f2;
}

/* Decompose a strictly-positive renormalized fp32mp2 value
 * (hi > 0, finite; lo any sign, |lo| <= ulp(hi)/2) into M * 2^E with
 * M normalized to [2^52, 2^53).
 *
 * The 53-bit window matches IEEE double's significand: a renormalized
 * fp32mp2 carries 24 (hi) + 24 (lo) significant bits separated by a tiny
 * gap (the test generator places lo at exponent Eh-25), so its full span
 * is ~49 bits.  A 48-bit window would round away the bottom of lo whenever
 * lo sits below ulp(hi)/2 - that loss is invisible for most ops but is
 * catastrophically amplified by the cancellation inherent to fmod when the
 * result is far smaller than the inputs.  Capturing 53 bits keeps the value
 * exactly (equivalent to the fp64 fallback's double round-trip). */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_modf_decompose(float __hi, float __lo, unsigned long long* __m, int* __e) noexcept
{
  const uint32_t __hb = ::cuda::std::bit_cast<uint32_t>(__hi);
  int __eh;
  if ((__hb & 0x7F800000u) == 0u)
  {
    /* denormal hi (hi > 0): value = mant * 2^-149 */
    const uint32_t __mant = __hb & 0x007FFFFFu;
    const float __fm      = static_cast<float>(__mant);
    const uint32_t __fmb  = ::cuda::std::bit_cast<uint32_t>(__fm);
    __eh                  = static_cast<int>((__fmb >> 23) & 0xFFu) - 127 - 149;
  }
  else
  {
    __eh = static_cast<int>((__hb >> 23) & 0xFFu) - 127;
  }

  const int __s     = 52 - __eh;
  const float __shi = __internal_fpmp2_scale2_scalar(__hi, __s); /* integer in [2^52, 2^53) */
  const float __slo = __internal_fpmp2_scale2_scalar(__lo, __s); /* |slo| <= 2^28           */

  long long __mant =
    static_cast<long long>(static_cast<unsigned long long>(__shi)) + static_cast<long long>(__fpmp_fp2int_rn(__slo));
  int __exp = __eh - 52;

  /* lo > 0 may push m just past 2^53; bring it back. */
  if ((static_cast<unsigned long long>(__mant) >> 53) != 0ULL)
  {
    __mant >>= 1;
    __exp += 1;
  }
  /* Off-by-one fix: when hi is an exact power of two and lo < 0 the
   * true value sits just below 2^Eh, so m lands just under 2^52.
   * One left shift renormalizes it back into [2^52, 2^53). */
  if (__mant != 0 && (static_cast<unsigned long long>(__mant) >> 52) == 0ULL)
  {
    __mant <<= 1;
    __exp -= 1;
  }

  *__m = static_cast<unsigned long long>(__mant);
  *__e = __exp;
}

/* Build a renormalized fp32mp2 from  (neg ? -1 : 1) * mag * 2^E.
 * mag may carry up to 53 significant bits; it is first rounded
 * (round-half-to-even) down to the 48 bits an fp32mp2 can hold, then split
 * into two <= 24-bit halves so each casts to float exactly. */
_CCCL_FPMP_CORE_API void __internal_fpmp2_modf_reconstruct(
  unsigned long long __mag, int __e, bool __neg, float* __res_hi, float* __res_lo) noexcept
{
  if (__mag == 0ULL)
  {
    *__res_hi = __neg ? -0.0f : 0.0f;
    *__res_lo = 0.0f;
    return;
  }

  /* Round mag down to <= 48 significant bits. */
  int __extra = 0;
  for (unsigned long long __t = __mag; __t >= (1ULL << 48); __t >>= 1)
  {
    ++__extra;
  }
  if (__extra > 0)
  {
    const unsigned long long __half = 1ULL << (__extra - 1);
    const unsigned long long __frac = __mag & ((1ULL << __extra) - 1ULL);
    unsigned long long __q          = __mag >> __extra;
    if (__frac > __half || (__frac == __half && (__q & 1ULL) != 0ULL))
    {
      ++__q;
    }
    __mag = __q;
    __e += __extra;
    if ((__mag >> 48) != 0ULL)
    {
      __mag >>= 1;
      ++__e;
    } /* rounding carried out */
  }

  const unsigned __hipart = static_cast<unsigned>(__mag >> 24); /* < 2^24 */
  const unsigned __lopart = static_cast<unsigned>(__mag & 0xFFFFFFULL); /* < 2^24 */
  float __rhi             = __internal_fpmp2_scale2_scalar(static_cast<float>(__hipart), __e + 24);
  float __rlo             = __internal_fpmp2_scale2_scalar(static_cast<float>(__lopart), __e);
  if (__neg)
  {
    __rhi = -__rhi;
    __rlo = -__rlo;
  }

  float __lo;
  const float __hi = __fpmp_two_sum(__rhi, __rlo, &__lo); /* exact, no magnitude assumption */
  *__res_hi        = __hi;
  *__res_lo        = __lo;
}

/* Core reduction: assumes ax > ay > 0 (both finite, nonzero), inputs
 * given as positive renormalized (hi, lo) pairs.  Returns the fmod
 * remainder mantissa ia (< My), the divisor mantissa My, its
 * exponent Ey, and the low bits of the integer quotient
 * floor(ax/ay) in quo. */
_CCCL_FPMP_CORE_API void __internal_fpmp2_fmod_kernel(
  float __ax_hi,
  float __ax_lo,
  float __ay_hi,
  float __ay_lo,
  unsigned long long* __ia_out,
  unsigned long long* __My_out,
  int* __Ey_out,
  unsigned long long* __quo_out) noexcept
{
  unsigned long long __Mx;
  unsigned long long __my;
  int __Ex;
  int __ey;
  __internal_fpmp2_modf_decompose(__ax_hi, __ax_lo, &__Mx, &__Ex);
  __internal_fpmp2_modf_decompose(__ay_hi, __ay_lo, &__my, &__ey);

  int __d = __Ex - __ey; /* >= 0 since ax > ay and both M in [2^52,2^53) */
  if (__d < 0)
  {
    __d = 0; /* defensive */
  }

  unsigned long long __quo = __Mx / __my;
  unsigned long long __ia  = __Mx % __my;
  int __remaining          = __d;
  while (__remaining > 0)
  {
    const int __s                  = (__remaining < 11) ? __remaining : 11;
    const unsigned long long __num = __ia << __s; /* ia < My < 2^53, so num < 2^64 */
    __quo                          = (__quo << __s) + (__num / __my); /* low bits of quotient (parity only) */
    __ia                           = __num % __my;
    __remaining -= __s;
  }

  *__ia_out  = __ia;
  *__My_out  = __my;
  *__Ey_out  = __ey;
  *__quo_out = __quo;
}

/*
 * ====================================================================
 * fmod(x, y) - floating-point remainder of x / y
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Remainder fmod(x, y) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * fmod(x, y): result has the sign of x and magnitude in [0, |y|).
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_fmod(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  /* (hi + lo) != (hi + lo) also catches a degenerate (+inf, -inf) limb
   * pair, which the fp128 reference widens to inf + (-inf) = NaN. */
  const bool __x_nan  = (__x_hi != __x_hi) || (__x_lo != __x_lo) || ((__x_hi + __x_lo) != (__x_hi + __x_lo));
  const bool __y_nan  = (__y_hi != __y_hi) || (__y_lo != __y_lo) || ((__y_hi + __y_lo) != (__y_hi + __y_lo));
  const float __axh   = (__x_hi < 0.0f) ? -__x_hi : __x_hi;
  const float __ayh   = (__y_hi < 0.0f) ? -__y_hi : __y_hi;
  const bool __x_inf  = (__axh == ::cuda::std::__fp_inf<float>());
  const bool __y_inf  = (__ayh == ::cuda::std::__fp_inf<float>());
  const bool __y_zero = (__y_hi == 0.0f);

  if (__x_nan || __y_nan || __x_inf || __y_zero)
  {
    *__res_hi = ::cuda::std::__fp_nan<float>();
    *__res_lo = ::cuda::std::__fp_nan<float>();
    return;
  }
  if (__y_inf) /* fmod(finite, inf) = x */
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
    return;
  }

  const float __axl = (__x_hi < 0.0f) ? -__x_lo : __x_lo;
  const float __ayl = (__y_hi < 0.0f) ? -__y_lo : __y_lo;

  int __c;
  if (__axh != __ayh)
  {
    __c = (__axh < __ayh) ? -1 : 1;
  }
  else if (__axl != __ayl)
  {
    __c = (__axl < __ayl) ? -1 : 1;
  }
  else
  {
    __c = 0;
  }

  if (__c < 0)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
    return;
  } /* |x| < |y| -> x   */
  if (__c == 0)
  {
    *__res_hi = (__x_hi < 0.0f) ? -0.0f : 0.0f;
    *__res_lo = 0.0f;
    return;
  }

  unsigned long long __ia;
  unsigned long long __my;
  unsigned long long __quo;
  int __ey;
  __internal_fpmp2_fmod_kernel(__axh, __axl, __ayh, __ayl, &__ia, &__my, &__ey, &__quo);
  __internal_fpmp2_modf_reconstruct(__ia, __ey, (__x_hi < 0.0f), __res_hi, __res_lo);
}

/*
 * --------------------------------------------------------------------
 * Remainder fmod(x, y) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_fmod(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH_2A(fmod, _CCCL_FPMP_FMODQ, __x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_2A(fmod)

/*
 * ====================================================================
 * remainder(x, y) - IEEE remainder of x / y
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * IEEE remainder remainder(x, y) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * remainder(x, y): IEEE remainder, |result| <= |y|/2, round-to-nearest
 * with ties to even quotient.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_remainder(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  /* (hi + lo) != (hi + lo) also catches a degenerate (+inf, -inf) limb
   * pair, which the fp128 reference widens to inf + (-inf) = NaN. */
  const bool __x_nan  = (__x_hi != __x_hi) || (__x_lo != __x_lo) || ((__x_hi + __x_lo) != (__x_hi + __x_lo));
  const bool __y_nan  = (__y_hi != __y_hi) || (__y_lo != __y_lo) || ((__y_hi + __y_lo) != (__y_hi + __y_lo));
  const float __axh   = (__x_hi < 0.0f) ? -__x_hi : __x_hi;
  const float __ayh   = (__y_hi < 0.0f) ? -__y_hi : __y_hi;
  const bool __x_inf  = (__axh == ::cuda::std::__fp_inf<float>());
  const bool __y_inf  = (__ayh == ::cuda::std::__fp_inf<float>());
  const bool __y_zero = (__y_hi == 0.0f);
  const bool __xneg   = (__x_hi < 0.0f);

  if (__x_nan || __y_nan || __x_inf || __y_zero)
  {
    *__res_hi = ::cuda::std::__fp_nan<float>();
    *__res_lo = ::cuda::std::__fp_nan<float>();
    return;
  }
  if (__y_inf) /* remainder(finite, inf) = x */
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
    return;
  }

  const float __axl = (__x_hi < 0.0f) ? -__x_lo : __x_lo;
  const float __ayl = (__y_hi < 0.0f) ? -__y_lo : __y_lo;

  int __c;
  if (__axh != __ayh)
  {
    __c = (__axh < __ayh) ? -1 : 1;
  }
  else if (__axl != __ayl)
  {
    __c = (__axl < __ayl) ? -1 : 1;
  }
  else
  {
    __c = 0;
  }

  if (__c == 0) /* |x| == |y| -> remainder 0 (sign of x) */
  {
    *__res_hi = __xneg ? -0.0f : 0.0f;
    *__res_lo = 0.0f;
    return;
  }

  if (__c < 0)
  {
    /* |x| < |y|: quotient is 0 or +-1.  Compare 2|x| against |y|. */
    const float __t_hi = 2.0f * __axh;
    const float __t_lo = 2.0f * __axl; /* 2|x| exact */
    int __c2;
    if (__t_hi != __ayh)
    {
      __c2 = (__t_hi < __ayh) ? -1 : 1;
    }
    else if (__t_lo != __ayl)
    {
      __c2 = (__t_lo < __ayl) ? -1 : 1;
    }
    else
    {
      __c2 = 0; /* tie -> quotient 0 (even) */
    }

    if (__c2 <= 0)
    {
      *__res_hi = __x_hi;
      *__res_lo = __x_lo;
      return;
    } /* r = x */

    /* 2|x| > |y|: r = |x| - |y|  (negative in the |x| frame) */
    const __ffloat __r = sub<fpmp2_accuracy::high>(__ffloat(__axh, __axl), __ffloat(__ayh, __ayl));
    float __rh         = __r.hi();
    float __rl         = __r.lo();
    if (__xneg)
    {
      __rh = -__rh;
      __rl = -__rl;
    }
    *__res_hi = __rh;
    *__res_lo = __rl;
    return;
  }

  /* |x| > |y|: full integer reduction, then round-to-nearest-even. */
  unsigned long long __ia;
  unsigned long long __my;
  unsigned long long __quo;
  int __ey;
  __internal_fpmp2_fmod_kernel(__axh, __axl, __ayh, __ayl, &__ia, &__my, &__ey, &__quo);

  const unsigned long long __two_ia = __ia << 1;
  const bool __round_up             = (__two_ia > __my) || ((__two_ia == __my) && ((__quo & 1ULL) != 0ULL));

  unsigned long long __mag;
  bool __neg_xframe;
  if (__round_up)
  {
    __mag        = __my - __ia;
    __neg_xframe = true;
  } /* r = (|x| mod |y|) - |y| < 0 */
  else
  {
    __mag        = __ia;
    __neg_xframe = false;
  }

  __internal_fpmp2_modf_reconstruct(__mag, __ey, static_cast<bool>(__neg_xframe ^ __xneg), __res_hi, __res_lo);
}

/*
 * --------------------------------------------------------------------
 * IEEE remainder remainder(x, y) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_remainder(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH_2A(remainder, _CCCL_FPMP_REMAINDERQ, __x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_2A(remainder)

/*
 * ====================================================================
 * floor(x) - round down to an integer
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round down floor(x) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * ceil, floor, trunc and round need no separate fp64mp2 body.
 * The templates above are exact on the limb pair for both element types -
 * they select the integer threshold from _FpType and renormalize through
 * __fpmp2_acc - so fp64mp2 uses them directly. Routing fp64mp2 through fp128
 * instead was exact only where fp128 was available and silently collapsed the pair
 * into one double everywhere else, which is wrong rather than imprecise: the low
 * limb is what decides the answer whenever hi is integral, so ceil could return a
 * value below its own argument.
 *
 * floor/ceil/round: dedicated fpmp2 implementations that operate directly
 * on the (hi, lo) pair and avoid collapsing through an intermediate double.
 *
 * floor:
 *   Let n = floor(x_hi). If x >= n, result is n. Otherwise result is n - 1.
 *
 * ceil:
 *   Let n = ceil(x_hi). If x <= n, result is n. Otherwise result is n + 1.
 *
 * round:
 *   C semantics (halfway away from zero), implemented as
 *     round(x) = floor(x + 0.5)  for x >= 0
 *     round(x) = ceil (x - 0.5)  for x <  0
 * using fpmp2_acc to keep the adjustment in pair precision.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_floor(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  // NaN check
  if ((__x_hi != __x_hi) || (__x_lo != __x_lo))
  {
    _FpType __res = __x_hi + __x_lo;
    *__res_hi     = __res;
    *__res_lo     = __res;
    return;
  }

  const _FpType __abs_hi    = __fpmp_internal_fabs(__x_hi);
  const _FpType __int_scale = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p23f) : _FpType(0x1.0p52);
  if (__abs_hi >= __int_scale)
  {
    // x_hi is already an integer at this scale; floor(x_hi + x_lo) = x_hi + floor(x_lo).
    const _FpType __lo_floor = __fpmp_internal_floor<_FpType>(__x_lo);
    _FpType __t_hi = __x_hi, __t_lo = _FpType(0);
    __fpmp2_acc<_FpType>(__lo_floor, &__t_hi, &__t_lo);
    *__res_hi = __t_hi;
    *__res_lo = __t_lo;
    return;
  }

  const _FpType __n = __fpmp_internal_floor<_FpType>(__x_hi);
  if (__x_hi != __n || __x_lo >= _FpType(0))
  {
    *__res_hi = __n;
    *__res_lo = _FpType(0);
    return;
  }

  _FpType __t_hi = __n, __t_lo = _FpType(0);
  __fpmp2_acc<_FpType>(_FpType(-1), &__t_hi, &__t_lo);
  *__res_hi = __t_hi;
  *__res_lo = __t_lo;
}

/*
 * ====================================================================
 * ceil(x) - round up to an integer
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round up ceil(x) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_ceil(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  // NaN check
  if ((__x_hi != __x_hi) || (__x_lo != __x_lo))
  {
    _FpType __res = __x_hi + __x_lo;
    *__res_hi     = __res;
    *__res_lo     = __res;
    return;
  }

  const _FpType __abs_hi    = __fpmp_internal_fabs(__x_hi);
  const _FpType __int_scale = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p23f) : _FpType(0x1.0p52);
  if (__abs_hi >= __int_scale)
  {
    // x_hi is already an integer at this scale; ceil(x_hi + x_lo) = x_hi + ceil(x_lo).
    const _FpType __lo_ceil = __fpmp_internal_ceil<_FpType>(__x_lo);
    _FpType __t_hi = __x_hi, __t_lo = _FpType(0);
    __fpmp2_acc<_FpType>(__lo_ceil, &__t_hi, &__t_lo);
    *__res_hi = __t_hi;
    *__res_lo = __t_lo;
    return;
  }

  const _FpType __n = __fpmp_internal_ceil<_FpType>(__x_hi);
  if (__x_hi != __n || __x_lo <= _FpType(0))
  {
    *__res_hi = __n;
    *__res_lo = _FpType(0);
    return;
  }

  _FpType __t_hi = __n, __t_lo = _FpType(0);
  __fpmp2_acc<_FpType>(_FpType(1), &__t_hi, &__t_lo);
  *__res_hi = __t_hi;
  *__res_lo = __t_lo;
}

/*
 * ====================================================================
 * round(x) - round to nearest, ties away from zero
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round to nearest round(x) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_round(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  // NaN check
  if ((__x_hi != __x_hi) || (__x_lo != __x_lo))
  {
    _FpType __res = __x_hi + __x_lo;
    *__res_hi     = __res;
    *__res_lo     = __res;
    return;
  }

  const bool __x_neg = (__x_hi < _FpType(0)) || (__x_hi == _FpType(0) && __x_lo < _FpType(0));

  _FpType __t_hi = __x_hi, __t_lo = __x_lo;
  __fpmp2_acc<_FpType>(__x_neg ? _FpType(-0.5) : _FpType(0.5), &__t_hi, &__t_lo);

  if (__x_neg)
  {
    __fpmp2_ceil(__t_hi, __t_lo, __res_hi, __res_lo);
  }
  else
  {
    __fpmp2_floor(__t_hi, __t_lo, __res_hi, __res_lo);
  }
}

/*
 * ====================================================================
 * trunc(x) - round toward zero
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round toward zero trunc(x) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_trunc(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  // NaN check
  if ((__x_hi != __x_hi) || (__x_lo != __x_lo))
  {
    _FpType __res = __x_hi + __x_lo;
    *__res_hi     = __res;
    *__res_lo     = __res;
    return;
  }

  const _FpType __abs_hi    = __fpmp_internal_fabs(__x_hi);
  const _FpType __int_scale = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p23f) : _FpType(0x1.0p52);
  if (__abs_hi >= __int_scale)
  {
    // x_hi is integral at this scale and dominates sign, so trunc is:
    //   x_hi > 0 : floor(x_hi + x_lo) = x_hi + floor(x_lo)
    //   x_hi < 0 : ceil (x_hi + x_lo) = x_hi + ceil (x_lo)
    const _FpType __lo_trunc =
      (__x_hi < _FpType(0)) ? __fpmp_internal_ceil<_FpType>(__x_lo) : __fpmp_internal_floor<_FpType>(__x_lo);
    _FpType __t_hi = __x_hi, __t_lo = _FpType(0);
    __fpmp2_acc<_FpType>(__lo_trunc, &__t_hi, &__t_lo);
    *__res_hi = __t_hi;
    *__res_lo = __t_lo;
    return;
  }

  // Fast small-magnitude path:
  // Start from trunc(x_hi), then apply at most a +/-1 correction only when
  // x_hi is already integral and x_lo nudges the exact value across that integer.
  const _FpType __n = __fpmp_internal_trunc<_FpType>(__x_hi);
  if (__x_hi != __n)
  {
    *__res_hi = __n;
    *__res_lo = _FpType(0);
    return;
  }

  const bool __x_neg = (__x_hi < _FpType(0)) || (__x_hi == _FpType(0) && __x_lo < _FpType(0));
  const int __delta  = (!__x_neg && __x_lo < _FpType(0)) ? -1 : (__x_neg && __x_lo > _FpType(0)) ? 1 : 0;
  if (__delta != 0)
  {
    _FpType __t_hi = __n, __t_lo = _FpType(0);
    __fpmp2_acc<_FpType>(static_cast<_FpType>(__delta), &__t_hi, &__t_lo);
    *__res_hi = __t_hi;
    *__res_lo = __t_lo;
    return;
  }

  *__res_hi = __n;
  *__res_lo = _FpType(0);
}

/*
 * ====================================================================
 * llrint(x) - round to nearest long long, ties to even
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round to long long llrint(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_MATH_FALLBACK_1A_RETLL(llrint)

/*
 * --------------------------------------------------------------------
 * Round to long long llrint(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API long long int __internal_fpmp2_llrint(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::std::llrint(__fpmp2_to_double(__x_hi, __x_lo));
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETLL(llrint)

/*
 * ====================================================================
 * llround(x) - round to nearest long long, ties away from zero
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round to long long llround(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_MATH_FALLBACK_1A_RETLL(llround)

/*
 * --------------------------------------------------------------------
 * Round to long long llround(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API long long int __internal_fpmp2_llround(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::std::llround(__fpmp2_to_double(__x_hi, __x_lo));
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETLL(llround)

/*
 * ====================================================================
 * lrint(x) - round to nearest long, ties to even
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round to long lrint(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_MATH_FALLBACK_1A_RETL(lrint)

/*
 * --------------------------------------------------------------------
 * Round to long lrint(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API long int __internal_fpmp2_lrint(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::std::lrint(__fpmp2_to_double(__x_hi, __x_lo));
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETL(lrint)

/*
 * ====================================================================
 * lround(x) - round to nearest long, ties away from zero
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round to long lround(x) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_MATH_FALLBACK_1A_RETL(lround)

/*
 * --------------------------------------------------------------------
 * Round to long lround(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API long int __internal_fpmp2_lround(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::std::lround(__fpmp2_to_double(__x_hi, __x_lo));
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETL(lround)

/* Internal helper: parity of an integral value, for the rint tie-break. */
template <typename _FpType>
_CCCL_FPMP_CORE_API bool __internal_fpmp2_nearint_is_odd(const _FpType __n) noexcept
{
  // n is an integer. n/2 and 2*floor(n/2) are both exact, so this is a parity
  // test that also works past 2^53, where every value is even anyway.
  const _FpType __half_n = __n * _FpType(0.5);
  return __n != _FpType(2) * __fpmp_internal_floor<_FpType>(__half_n);
}

/*
 * ====================================================================
 * rint(x) - round to nearest, ties to even
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round to even rint(x) (fp32mp2 and fp64mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * Both limbs participate, so neither the pair nor an intermediate double is
 * collapsed. Two regimes, as in floor/ceil above:
 *
 *   |hi| >= int_scale: hi is an integer already, so only lo has a fractional
 *     part. A tie in lo is broken on the parity of hi + floor(lo), the two
 *     candidates being that value and one more.
 *
 *   |hi| <  int_scale: the fractional part of hi decides, except when hi sits
 *     exactly on a midpoint, where lo picks the side. lo cannot create or
 *     destroy a tie otherwise: a renormalized pair has |lo| <= ulp(hi)/2, and
 *     both hi and the midpoints are multiples of ulp(hi), so hi + lo stays
 *     strictly on hi's side of the nearest midpoint.
 *
 * The result is an integer below int_scale in the second regime, hence exact in
 * hi with lo = 0.
 *
 * nearbyint is rint without the inexact exception, which this implementation
 * never raises, so it forwards.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_rint(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  // NaN check
  if ((__x_hi != __x_hi) || (__x_lo != __x_lo))
  {
    _FpType __res = __x_hi + __x_lo;
    *__res_hi     = __res;
    *__res_lo     = __res;
    return;
  }

  const _FpType __half      = _FpType(0.5);
  const _FpType __abs_hi    = __fpmp_internal_fabs(__x_hi);
  const _FpType __int_scale = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p23f) : _FpType(0x1.0p52);

  if (__abs_hi >= __int_scale)
  {
    // Compare against the midpoint rather than subtracting: x - floor(x) is not
    // always exact (for x = -(0.5 - 2^-54) the true fraction 0.5 + 2^-54 rounds to
    // 0.5 and fabricates a tie), while floor(x) + 1/2 is representable below
    // int_scale and the comparison is exact.
    const _FpType __lo_floor = __fpmp_internal_floor<_FpType>(__x_lo);
    const _FpType __lo_mid   = __lo_floor + __half;
    _FpType __lo_r           = __lo_floor;
    if (__x_lo > __lo_mid)
    {
      __lo_r = __lo_floor + _FpType(1);
    }
    else if (__x_lo == __lo_mid)
    {
      // Tie: step to whichever of hi + lo_floor and hi + lo_floor + 1 is even.
      const bool __sum_odd =
        __internal_fpmp2_nearint_is_odd<_FpType>(__x_hi) != __internal_fpmp2_nearint_is_odd<_FpType>(__lo_floor);
      if (__sum_odd)
      {
        __lo_r = __lo_floor + _FpType(1);
      }
    }

    _FpType __t_hi = __x_hi, __t_lo = _FpType(0);
    __fpmp2_acc<_FpType>(__lo_r, &__t_hi, &__t_lo);
    *__res_hi = __t_hi;
    *__res_lo = __t_lo;
    return;
  }

  // Same midpoint comparison as above. hi strictly on one side of mid puts x on
  // that side too, since hi and mid are both multiples of ulp(hi) here and
  // |lo| <= ulp(hi)/2, so lo cannot cross the midpoint - it can only break a tie.
  const _FpType __n   = __fpmp_internal_floor<_FpType>(__x_hi);
  const _FpType __mid = __n + __half;
  _FpType __r         = __n;
  if (__x_hi > __mid)
  {
    __r = __n + _FpType(1);
  }
  else if (__x_hi == __mid)
  {
    if (__x_lo > _FpType(0))
    {
      __r = __n + _FpType(1);
    }
    else if (__x_lo == _FpType(0) && __internal_fpmp2_nearint_is_odd<_FpType>(__n))
    {
      __r = __n + _FpType(1); // exact tie, and n is the odd neighbour
    }
  }

  if (__r == _FpType(0) && __x_hi < _FpType(0))
  {
    __r = -__r; // rint(-0.25) is -0.0: the sign survives rounding to zero
  }

  *__res_hi = __r;
  *__res_lo = _FpType(0);
}

/*
 * ====================================================================
 * nearbyint(x) - round to nearest, ties to even
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round to even nearbyint(x) (fp32mp2 and fp64mp2) - forwards to rint
 * --------------------------------------------------------------------
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__fpmp2_nearbyint(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  __fpmp2_rint<_FpType>(__x_hi, __x_lo, __res_hi, __res_lo);
}

/*
 * ====================================================================
 * remquo(x, y, &q) - remainder and low bits of the quotient
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Remainder and quotient remquo(x, y, &q) (fp32mp2) - double fallback
 * --------------------------------------------------------------------
 * remquo: compute remainder and part of quotient
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_remquo(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo,
  int* __quo) noexcept
{
  using __mp2_t = fpmp2<float>;
  double __r    = ::cuda::std::remquo(
    static_cast<double>(__mp2_t(__x_hi, __x_lo)), static_cast<double>(__mp2_t(__y_hi, __y_lo)), __quo);
  __mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Remainder and quotient remquo(x, y, &q) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_remquo(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo,
  int* __quo) noexcept
{
  __fpmp2_from_double(::cuda::std::remquo(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo), __quo),
                      __res_hi,
                      __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_2A_QUO(remquo)

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_NEARINT_H
