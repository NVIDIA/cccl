//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
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
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__floating_point/constants.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
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
    /* A head that is a power of two with a tail of the other sign puts the
     * value in the binade below, where the 53-bit window reaches one bit
     * further down (as the double rounding of hi + lo does). Rounding the
     * tail at the upper binade's position and shifting afterwards (the
     * renormalization below) would lose that bit: nothing for a dense pair,
     * a different remainder for a sparse one (|lo| < 2^-29 ulp(hi)). */
    if (((__hb & 0x007FFFFFu) == 0u) && (__lo < 0.0f))
    {
      __eh -= 1;
    }
  }

  const int __s     = 52 - __eh;
  const float __shi = ::cuda::__internal_fpmp2_scale2_scalar(__hi, __s); /* integer in [2^52, 2^53) */
  const float __slo = ::cuda::__internal_fpmp2_scale2_scalar(__lo, __s); /* |slo| <= 2^28           */

  long long __mant = static_cast<long long>(static_cast<unsigned long long>(__shi))
                   + static_cast<long long>(::cuda::__fpmp_fp2int_rn(__slo));
  int __exp        = __eh - 52;

  /* lo > 0 may push m just past 2^53; bring it back. */
  if ((static_cast<unsigned long long>(__mant) >> 53) != 0ULL)
  {
    __mant >>= 1;
    __exp += 1;
  }
  /* A tail that rounds to zero below a power-of-two head leaves m at
   * 2^53 (caught above); a value that lands just under 2^52 can only come
   * from a denormal head, which the window above does not adjust. One
   * left shift renormalizes it back into [2^52, 2^53). */
  if (__mant != 0 && (static_cast<unsigned long long>(__mant) >> 52) == 0ULL)
  {
    __mant <<= 1;
    __exp -= 1;
  }

  *__m = static_cast<unsigned long long>(__mant);
  *__e = __exp;
}

/*
 * --------------------------------------------------------------------
 * Exact reduction for sparse pairs
 * --------------------------------------------------------------------
 * A renormalized pair whose tail sits more than 29 binades below its head
 * carries more than 53 significant bits, which the 53-bit window of the
 * kernel below cannot hold: it rounds the operand, and a remainder moves by
 * the whole amount dropped (a result far smaller than the operands is then
 * unrelated to the exact one).  Such pairs come out of arithmetic (a head
 * that lands on a power of two with a small tail), not out of conversions,
 * so they are rare; they take this path.  Only the divisor and the running
 * remainder need to be wide (the remainder is smaller than the divisor and
 * lives on its bit grid, which for a sparse divisor spans more than 64
 * bits): they are two 64-bit words, in the frame of the finer operand.  The
 * dividend is reduced limb by limb in 64-bit arithmetic - each limb is a
 * 24-bit integer times a power of two, reduced by doublings modulo the
 * divisor - and the two residues are combined; the quotient's low 64 bits
 * come along.  Exact as long as the divisor fits 126 bits in that frame
 * (tails down to 2^-102 ulp(hi) when |x| >= |y|); beyond that the 53-bit
 * kernel takes over.  Dense pairs never come here.  fmod, remainder and
 * remquo take this path in fpmp2_accuracy::high only, where every result is
 * the nearest pair; mid and low keep the 53-bit window and the 48-bit result
 * rounding of the original code (see the cores).
 */
struct __fpmp2_u128
{
  unsigned long long __hi;
  unsigned long long __lo;
};

_CCCL_FPMP_CORE_API bool __fpmp2_u128_ge(const __fpmp2_u128 __a, const __fpmp2_u128 __b) noexcept
{
  return (__a.__hi > __b.__hi) || ((__a.__hi == __b.__hi) && (__a.__lo >= __b.__lo));
}

_CCCL_FPMP_CORE_API __fpmp2_u128 __fpmp2_u128_sub(const __fpmp2_u128 __a, const __fpmp2_u128 __b) noexcept
{
  __fpmp2_u128 __r;
  __r.__lo = __a.__lo - __b.__lo;
  __r.__hi = __a.__hi - __b.__hi - ((__a.__lo < __b.__lo) ? 1ULL : 0ULL);
  return __r;
}

/* a << s for 0 <= s < 128 */
_CCCL_FPMP_CORE_API __fpmp2_u128 __fpmp2_u128_shl(const unsigned long long __a, const int __s) noexcept
{
  __fpmp2_u128 __r;
  if (__s >= 64)
  {
    __r.__hi = __a << (__s - 64);
    __r.__lo = 0ULL;
  }
  else if (__s > 0)
  {
    __r.__hi = __a >> (64 - __s);
    __r.__lo = __a << __s;
  }
  else
  {
    __r.__hi = 0ULL;
    __r.__lo = __a;
  }
  return __r;
}

/* (a << 1) | bit */
_CCCL_FPMP_CORE_API __fpmp2_u128 __fpmp2_u128_shl1(const __fpmp2_u128 __a, const unsigned long long __bit) noexcept
{
  __fpmp2_u128 __r;
  __r.__hi = (__a.__hi << 1) | (__a.__lo >> 63);
  __r.__lo = (__a.__lo << 1) | __bit;
  return __r;
}

/* index of the top set bit, -1 for zero */
_CCCL_FPMP_CORE_API int __fpmp2_u128_msb(const __fpmp2_u128 __a) noexcept
{
  if (__a.__hi != 0ULL)
  {
    return 127 - ::cuda::std::countl_zero(__a.__hi);
  }
  if (__a.__lo != 0ULL)
  {
    return 63 - ::cuda::std::countl_zero(__a.__lo);
  }
  return -1;
}

/* a >> s, low 64 bits of the result, 0 <= s < 128 */
_CCCL_FPMP_CORE_API unsigned long long __fpmp2_u128_shr_lo(const __fpmp2_u128 __a, const int __s) noexcept
{
  if (__s >= 64)
  {
    return __a.__hi >> (__s - 64);
  }
  if (__s > 0)
  {
    return (__a.__lo >> __s) | (__a.__hi << (64 - __s));
  }
  return __a.__lo;
}

/* a & (2^n - 1) != 0, 0 <= n < 128 */
_CCCL_FPMP_CORE_API bool __fpmp2_u128_any_below(const __fpmp2_u128 __a, const int __n) noexcept
{
  if (__n <= 0)
  {
    return false;
  }
  if (__n >= 64)
  {
    return (__a.__lo != 0ULL) || ((__a.__hi & ((1ULL << (__n - 64)) - 1ULL)) != 0ULL);
  }
  return (__a.__lo & ((1ULL << __n) - 1ULL)) != 0ULL;
}

/* bit n of a, 0 <= n < 128 */
_CCCL_FPMP_CORE_API unsigned long long __fpmp2_u128_bit(const __fpmp2_u128 __a, const int __n) noexcept
{
  return (__n >= 64) ? ((__a.__hi >> (__n - 64)) & 1ULL) : ((__a.__lo >> __n) & 1ULL);
}

/* A positive finite float as an integer times a power of two: the 24-bit
 * significand (the denormal field for a denormal) and the exponent of its
 * lowest bit. */
_CCCL_FPMP_CORE_API void __internal_fpmp2_limb_int(const float __f, unsigned long long* __m, int* __lsb) noexcept
{
  const uint32_t __b = ::cuda::std::bit_cast<uint32_t>(__f) & 0x7FFFFFFFu;
  const uint32_t __e = __b >> 23;
  if (__e == 0u)
  {
    *__m   = __b;
    *__lsb = -149;
  }
  else
  {
    *__m   = (__b & 0x7FFFFFu) | 0x800000u;
    *__lsb = static_cast<int>(__e) - 150;
  }
}

/* More than 53 significant bits in the pair: a nonzero tail more than 29
 * binades below the head (a denormal tail counts with exponent 1). */
_CCCL_FPMP_CORE_API bool __internal_fpmp2_pair_sparse(const float __hi, const float __lo) noexcept
{
  if (__lo == 0.0f)
  {
    return false;
  }
  const int __eh = static_cast<int>((::cuda::std::bit_cast<uint32_t>(__hi) >> 23) & 0xFFu);
  int __el       = static_cast<int>((::cuda::std::bit_cast<uint32_t>(__lo) >> 23) & 0xFFu);
  if (__el == 0)
  {
    __el = 1;
  }
  return (__eh - __el) > 29;
}

/* The exact integer of a positive pair in the frame 2^e (e <= its lowest
 * bit); false if it does not fit 126 bits. */
_CCCL_FPMP_CORE_API bool
__internal_fpmp2_pair_u128(const float __hi, const float __lo, const int __e, __fpmp2_u128* __v) noexcept
{
  unsigned long long __mh;
  int __lh;
  ::cuda::__internal_fpmp2_limb_int(__hi, &__mh, &__lh);
  const int __sh = __lh - __e;
  if (__sh + 24 > 126)
  {
    return false;
  }
  __fpmp2_u128 __r = ::cuda::__fpmp2_u128_shl(__mh, __sh);
  if (__lo != 0.0f)
  {
    unsigned long long __ml;
    int __ll;
    ::cuda::__internal_fpmp2_limb_int((__lo < 0.0f) ? -__lo : __lo, &__ml, &__ll);
    const __fpmp2_u128 __t = ::cuda::__fpmp2_u128_shl(__ml, __ll - __e);
    if (__lo < 0.0f)
    {
      __r = ::cuda::__fpmp2_u128_sub(__r, __t); /* |lo| <= ulp(hi)/2: no borrow out */
    }
    else
    {
      __r.__lo += __t.__lo;
      __r.__hi += __t.__hi + ((__r.__lo < __t.__lo) ? 1ULL : 0ULL);
    }
  }
  *__v = __r;
  return true;
}

/* Exponent of the lowest set bit of a positive pair. */
_CCCL_FPMP_CORE_API int __internal_fpmp2_pair_lsb(const float __hi, const float __lo) noexcept
{
  unsigned long long __m;
  int __l;
  ::cuda::__internal_fpmp2_limb_int((__lo != 0.0f) ? ((__lo < 0.0f) ? -__lo : __lo) : __hi, &__m, &__l);
  return __l;
}

/* The pair nearest to  (neg ? -1 : 1) * v * 2^e  for a 128-bit v: head
 * rounded to nearest even at 24 bits, tail the residual rounded the same
 * way, then Fast2Sum for the canonical form. */
_CCCL_FPMP_CORE_API void __internal_fpmp2_pair_from_u128(
  const __fpmp2_u128 __v, const int __e, const bool __neg, float* __res_hi, float* __res_lo) noexcept
{
  const int __k = ::cuda::__fpmp2_u128_msb(__v);
  if (__k < 0)
  {
    *__res_hi = __neg ? -0.0f : 0.0f;
    *__res_lo = 0.0f;
    return;
  }
  float __hi;
  float __lo = 0.0f;
  if (__k <= 23)
  {
    __hi = ::cuda::__internal_fpmp2_scale2_scalar(static_cast<float>(__v.__lo), __e);
  }
  else
  {
    const int __sh          = __k - 23;
    unsigned long long __hm = ::cuda::__fpmp2_u128_shr_lo(__v, __sh) & 0xFFFFFFULL;
    const bool __rb         = ::cuda::__fpmp2_u128_bit(__v, __sh - 1) != 0ULL;
    const bool __st         = ::cuda::__fpmp2_u128_any_below(__v, __sh - 1);
    bool __up               = __rb && (__st || ((__hm & 1ULL) != 0ULL));
    if (__up)
    {
      __hm += 1ULL; /* 2^24 at most: still a float */
    }
    __hi = ::cuda::__internal_fpmp2_scale2_scalar(static_cast<float>(__hm), __e + __sh);
    /* residual v - hm * 2^sh, below 2^(sh-1) in magnitude (2^(sh-1) on a tie) */
    const __fpmp2_u128 __h = ::cuda::__fpmp2_u128_shl(__hm, __sh);
    const __fpmp2_u128 __r = __up ? ::cuda::__fpmp2_u128_sub(__h, __v) : ::cuda::__fpmp2_u128_sub(__v, __h);
    const int __kr         = ::cuda::__fpmp2_u128_msb(__r);
    if (__kr >= 0)
    {
      if (__kr <= 23)
      {
        __lo = ::cuda::__internal_fpmp2_scale2_scalar(static_cast<float>(__r.__lo), __e);
      }
      else
      {
        const int __sr          = __kr - 23;
        unsigned long long __lm = ::cuda::__fpmp2_u128_shr_lo(__r, __sr) & 0xFFFFFFULL;
        const bool __rb2        = ::cuda::__fpmp2_u128_bit(__r, __sr - 1) != 0ULL;
        const bool __st2        = ::cuda::__fpmp2_u128_any_below(__r, __sr - 1);
        if (__rb2 && (__st2 || ((__lm & 1ULL) != 0ULL)))
        {
          __lm += 1ULL;
        }
        __lo = ::cuda::__internal_fpmp2_scale2_scalar(static_cast<float>(__lm), __e + __sr);
      }
      if (__up)
      {
        __lo = -__lo;
      }
    }
  }
  if (__neg)
  {
    __hi = -__hi;
    __lo = -__lo;
  }
  *__res_hi = ::cuda::__fpmp_fast_two_sum(__hi, __lo, __res_lo);
}

/* (m * 2^d) mod y for a 24-bit m and d >= 0, by doublings; the low 64 bits
 * of the quotient in q. */
_CCCL_FPMP_CORE_API void __internal_fpmp2_reduce_limb(
  const unsigned long long __m,
  const int __d,
  const __fpmp2_u128 __y,
  __fpmp2_u128* __r,
  unsigned long long* __q) noexcept
{
  __fpmp2_u128 __rr;
  unsigned long long __qq;
  __rr.__hi = 0ULL;
  if (__y.__hi != 0ULL || __y.__lo > __m)
  {
    __rr.__lo = __m;
    __qq      = 0ULL;
  }
  else
  {
    __rr.__lo = __m % __y.__lo;
    __qq      = __m / __y.__lo;
  }
  for (int __i = 0; __i < __d; ++__i)
  {
    __rr = ::cuda::__fpmp2_u128_shl1(__rr, 0ULL);
    __qq <<= 1;
    if (::cuda::__fpmp2_u128_ge(__rr, __y))
    {
      __rr = ::cuda::__fpmp2_u128_sub(__rr, __y);
      __qq |= 1ULL;
    }
  }
  *__r = __rr;
  *__q = __qq;
}

/* Exact  ax mod ay  for positive pairs (ax, ay finite nonzero): the
 * remainder R and the divisor Y in the common frame 2^e, and the low 64
 * bits of the integer quotient.  Returns false (nothing written) when the
 * divisor does not fit 126 bits in that frame. */
_CCCL_FPMP_CORE_API bool __internal_fpmp2_fmod_kernel_wide(
  const float __ax_hi,
  const float __ax_lo,
  const float __ay_hi,
  const float __ay_lo,
  __fpmp2_u128* __r_out,
  __fpmp2_u128* __y_out,
  int* __e_out,
  unsigned long long* __quo_out) noexcept
{
  const int __lx = ::cuda::__internal_fpmp2_pair_lsb(__ax_hi, __ax_lo);
  const int __ly = ::cuda::__internal_fpmp2_pair_lsb(__ay_hi, __ay_lo);
  const int __e  = (__lx < __ly) ? __lx : __ly;
  __fpmp2_u128 __y;
  if (!::cuda::__internal_fpmp2_pair_u128(__ay_hi, __ay_lo, __e, &__y))
  {
    return false;
  }
  /* the head limb, then the tail limb, each reduced on its own */
  unsigned long long __mh;
  int __lh;
  ::cuda::__internal_fpmp2_limb_int(__ax_hi, &__mh, &__lh);
  __fpmp2_u128 __r;
  unsigned long long __q;
  ::cuda::__internal_fpmp2_reduce_limb(__mh, __lh - __e, __y, &__r, &__q);
  if (__ax_lo != 0.0f)
  {
    unsigned long long __ml;
    int __ll;
    ::cuda::__internal_fpmp2_limb_int((__ax_lo < 0.0f) ? -__ax_lo : __ax_lo, &__ml, &__ll);
    __fpmp2_u128 __rl;
    unsigned long long __ql;
    ::cuda::__internal_fpmp2_reduce_limb(__ml, __ll - __e, __y, &__rl, &__ql);
    if (__ax_lo < 0.0f)
    {
      /* r - rl, one divisor back if that goes below zero */
      __q -= __ql;
      if (::cuda::__fpmp2_u128_ge(__r, __rl))
      {
        __r = ::cuda::__fpmp2_u128_sub(__r, __rl);
      }
      else
      {
        __r = ::cuda::__fpmp2_u128_sub(__y, ::cuda::__fpmp2_u128_sub(__rl, __r)); /* y - (rl - r) */
        __q -= 1ULL;
      }
    }
    else
    {
      /* r + rl (< 2y < 2^127), one divisor off if that reaches y */
      __q += __ql;
      __fpmp2_u128 __t;
      __t.__lo = __r.__lo + __rl.__lo;
      __t.__hi = __r.__hi + __rl.__hi + ((__t.__lo < __rl.__lo) ? 1ULL : 0ULL);
      if (::cuda::__fpmp2_u128_ge(__t, __y))
      {
        __t = ::cuda::__fpmp2_u128_sub(__t, __y);
        __q += 1ULL;
      }
      __r = __t;
    }
  }
  *__r_out   = __r;
  *__y_out   = __y;
  *__e_out   = __e;
  *__quo_out = __q;
  return true;
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
  float __rhi             = ::cuda::__internal_fpmp2_scale2_scalar(static_cast<float>(__hipart), __e + 24);
  float __rlo             = ::cuda::__internal_fpmp2_scale2_scalar(static_cast<float>(__lopart), __e);
  if (__neg)
  {
    __rhi = -__rhi;
    __rlo = -__rlo;
  }

  float __lo;
  const float __hi = ::cuda::__fpmp_two_sum(__rhi, __rlo, &__lo); /* exact, no magnitude assumption */
  *__res_hi        = __hi;
  *__res_lo        = __lo;
}

/* The same for fpmp2_accuracy::high: the pair nearest to mag * 2^E (head and
 * tail each rounded to nearest even) -- the window above drops tail bits the
 * pair can hold whenever the head has few bits, up to 2^-48 relative. */
_CCCL_FPMP_CORE_API void __internal_fpmp2_modf_reconstruct_exact(
  unsigned long long __mag, int __e, bool __neg, float* __res_hi, float* __res_lo) noexcept
{
  __fpmp2_u128 __v;
  __v.__hi = 0ULL;
  __v.__lo = __mag;
  ::cuda::__internal_fpmp2_pair_from_u128(__v, __e, __neg, __res_hi, __res_lo);
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
  ::cuda::__internal_fpmp2_modf_decompose(__ax_hi, __ax_lo, &__Mx, &__Ex);
  ::cuda::__internal_fpmp2_modf_decompose(__ay_hi, __ay_lo, &__my, &__ey);

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
 * fmod(x, y): result has the sign of x and magnitude in [0, |y|).  With
 * exact set (fpmp2_accuracy::high) a sparse operand takes the wide reduction
 * and the result is the nearest pair; otherwise (mid, low) the operands are
 * rounded to the 53-bit window and the result to 48 bits as before, which
 * keeps the function's size where it was (277 against 1087 SASS lines and
 * 22 against 32 registers: the exact block weighs on a caller's register
 * budget even when it is never entered).
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_fmod_core(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo,
  const bool __exact) noexcept
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

  if (__exact
      && (::cuda::__internal_fpmp2_pair_sparse(__axh, __axl) || ::cuda::__internal_fpmp2_pair_sparse(__ayh, __ayl)))
  {
    __fpmp2_u128 __r;
    __fpmp2_u128 __y;
    int __e;
    unsigned long long __q;
    if (::cuda::__internal_fpmp2_fmod_kernel_wide(__axh, __axl, __ayh, __ayl, &__r, &__y, &__e, &__q))
    {
      ::cuda::__internal_fpmp2_pair_from_u128(__r, __e, (__x_hi < 0.0f), __res_hi, __res_lo);
      return;
    }
  }

  unsigned long long __ia;
  unsigned long long __my;
  unsigned long long __quo;
  int __ey;
  ::cuda::__internal_fpmp2_fmod_kernel(__axh, __axl, __ayh, __ayl, &__ia, &__my, &__ey, &__quo);
  if (__exact)
  {
    ::cuda::__internal_fpmp2_modf_reconstruct_exact(__ia, __ey, (__x_hi < 0.0f), __res_hi, __res_lo);
  }
  else
  {
    ::cuda::__internal_fpmp2_modf_reconstruct(__ia, __ey, (__x_hi < 0.0f), __res_hi, __res_lo);
  }
}

_CCCL_FPMP_CORE_API void __internal_fpmp2_fmod(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_fmod_core(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, false);
}

/* fpmp2_accuracy::high: exact on sparse pairs as well, nearest-pair result */
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_fmod(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_fmod_core(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, true);
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

/* fp64mp2 has one fmod for every level */
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_fmod(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_fmod(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_2A(fmod)
_CCCL_FPMP_MATH_DISPATCH_2A(high_fmod)

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
 * with ties to even quotient.  The core also hands out the integer
 * quotient rint(|x| / |y|) (its low 64 bits) for remquo.  With exact set
 * (fpmp2_accuracy::high) a sparse operand takes the wide reduction and the
 * result is the nearest pair; otherwise (mid, low) the operands are rounded
 * to the 53-bit window and the result to 48 bits as before, which keeps the
 * function's size and speed where they were (the wide block, present or
 * not, costs the dense path ~14 % latency and a resident block per SM
 * through its registers).
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_remainder_core(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo,
  unsigned long long* __n,
  const bool __exact) noexcept
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
    *__n      = 0ULL;
    return;
  }
  if (__y_inf) /* remainder(finite, inf) = x */
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
    *__n      = 0ULL;
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

  if (__c == 0) /* |x| == |y| -> remainder 0 (sign of x), quotient 1 */
  {
    *__res_hi = __xneg ? -0.0f : 0.0f;
    *__res_lo = 0.0f;
    *__n      = 1ULL;
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
      *__n      = 0ULL;
      return;
    } /* r = x, quotient 0 */

    /* 2|x| > |y|: r = |x| - |y|  (negative in the |x| frame) */
    if (__exact
        && (::cuda::__internal_fpmp2_pair_sparse(__axh, __axl) || ::cuda::__internal_fpmp2_pair_sparse(__ayh, __ayl)))
    {
      __fpmp2_u128 __xr;
      __fpmp2_u128 __yr;
      int __e;
      unsigned long long __q;
      if (::cuda::__internal_fpmp2_fmod_kernel_wide(__axh, __axl, __ayh, __ayl, &__xr, &__yr, &__e, &__q))
      {
        /* x < y: the division left x itself; the result is y - x, negated */
        ::cuda::__internal_fpmp2_pair_from_u128(::cuda::__fpmp2_u128_sub(__yr, __xr), __e, !__xneg, __res_hi, __res_lo);
        *__n = 1ULL;
        return;
      }
    }
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
    *__n      = 1ULL;
    return;
  }

  /* |x| > |y|: full integer reduction, then round-to-nearest-even. */
  if (__exact
      && (::cuda::__internal_fpmp2_pair_sparse(__axh, __axl) || ::cuda::__internal_fpmp2_pair_sparse(__ayh, __ayl)))
  {
    __fpmp2_u128 __r;
    __fpmp2_u128 __y;
    int __e;
    unsigned long long __q;
    if (::cuda::__internal_fpmp2_fmod_kernel_wide(__axh, __axl, __ayh, __ayl, &__r, &__y, &__e, &__q))
    {
      const __fpmp2_u128 __two_r = ::cuda::__fpmp2_u128_shl1(__r, 0ULL);
      const bool __up            = (!::cuda::__fpmp2_u128_ge(__y, __two_r)) /* 2r > y */
                                || ((__two_r.__hi == __y.__hi) && (__two_r.__lo == __y.__lo) && ((__q & 1ULL) != 0ULL));
      *__n                       = __q + (__up ? 1ULL : 0ULL);
      if (__up)
      {
        ::cuda::__internal_fpmp2_pair_from_u128(::cuda::__fpmp2_u128_sub(__y, __r), __e, !__xneg, __res_hi, __res_lo);
      }
      else
      {
        ::cuda::__internal_fpmp2_pair_from_u128(__r, __e, __xneg, __res_hi, __res_lo);
      }
      return;
    }
  }
  unsigned long long __ia;
  unsigned long long __my;
  unsigned long long __quo;
  int __ey;
  ::cuda::__internal_fpmp2_fmod_kernel(__axh, __axl, __ayh, __ayl, &__ia, &__my, &__ey, &__quo);

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

  *__n = __quo + (__round_up ? 1ULL : 0ULL); /* rint(|x| / |y|), low 64 bits */
  if (__exact)
  {
    ::cuda::__internal_fpmp2_modf_reconstruct_exact(
      __mag, __ey, static_cast<bool>(__neg_xframe ^ __xneg), __res_hi, __res_lo);
  }
  else
  {
    ::cuda::__internal_fpmp2_modf_reconstruct(__mag, __ey, static_cast<bool>(__neg_xframe ^ __xneg), __res_hi, __res_lo);
  }
}

_CCCL_FPMP_CORE_API void __internal_fpmp2_remainder(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  unsigned long long __n;
  ::cuda::__internal_fpmp2_remainder_core(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, &__n, false);
}

/* fpmp2_accuracy::high: exact on sparse pairs as well */
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_remainder(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  unsigned long long __n;
  ::cuda::__internal_fpmp2_remainder_core(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, &__n, true);
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

/* fp64mp2 has one remainder for every level */
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_remainder(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_remainder(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_2A(remainder)
_CCCL_FPMP_MATH_DISPATCH_2A(high_remainder)

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

  const _FpType __abs_hi    = ::cuda::__fpmp_internal_fabs(__x_hi);
  const _FpType __int_scale = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p23f) : _FpType(0x1.0p52);
  if (__abs_hi >= __int_scale)
  {
    // x_hi is already an integer at this scale; floor(x_hi + x_lo) = x_hi + floor(x_lo).
    const _FpType __lo_floor = ::cuda::__fpmp_internal_floor<_FpType>(__x_lo);
    _FpType __t_hi = __x_hi, __t_lo = _FpType(0);
    ::cuda::__fpmp2_acc<_FpType>(__lo_floor, &__t_hi, &__t_lo);
    *__res_hi = __t_hi;
    *__res_lo = __t_lo;
    return;
  }

  const _FpType __n = ::cuda::__fpmp_internal_floor<_FpType>(__x_hi);
  if (__x_hi != __n || __x_lo >= _FpType(0))
  {
    *__res_hi = __n;
    *__res_lo = _FpType(0);
    return;
  }

  _FpType __t_hi = __n, __t_lo = _FpType(0);
  ::cuda::__fpmp2_acc<_FpType>(_FpType(-1), &__t_hi, &__t_lo);
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

  const _FpType __abs_hi    = ::cuda::__fpmp_internal_fabs(__x_hi);
  const _FpType __int_scale = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p23f) : _FpType(0x1.0p52);
  if (__abs_hi >= __int_scale)
  {
    // x_hi is already an integer at this scale; ceil(x_hi + x_lo) = x_hi + ceil(x_lo).
    const _FpType __lo_ceil = ::cuda::__fpmp_internal_ceil<_FpType>(__x_lo);
    _FpType __t_hi = __x_hi, __t_lo = _FpType(0);
    ::cuda::__fpmp2_acc<_FpType>(__lo_ceil, &__t_hi, &__t_lo);
    *__res_hi = __t_hi;
    *__res_lo = __t_lo;
    return;
  }

  const _FpType __n = ::cuda::__fpmp_internal_ceil<_FpType>(__x_hi);
  if (__x_hi != __n || __x_lo <= _FpType(0))
  {
    *__res_hi = __n;
    *__res_lo = _FpType(0);
    return;
  }

  _FpType __t_hi = __n, __t_lo = _FpType(0);
  ::cuda::__fpmp2_acc<_FpType>(_FpType(1), &__t_hi, &__t_lo);
  *__res_hi = __t_hi;
  *__res_lo = __t_lo;
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

  const _FpType __abs_hi    = ::cuda::__fpmp_internal_fabs(__x_hi);
  const _FpType __int_scale = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p23f) : _FpType(0x1.0p52);
  if (__abs_hi >= __int_scale)
  {
    // x_hi is integral at this scale and dominates sign, so trunc is:
    //   x_hi > 0 : floor(x_hi + x_lo) = x_hi + floor(x_lo)
    //   x_hi < 0 : ceil (x_hi + x_lo) = x_hi + ceil (x_lo)
    const _FpType __lo_trunc = (__x_hi < _FpType(0)) ? ::cuda::__fpmp_internal_ceil<_FpType>(__x_lo)
                                                     : ::cuda::__fpmp_internal_floor<_FpType>(__x_lo);
    _FpType __t_hi = __x_hi, __t_lo = _FpType(0);
    ::cuda::__fpmp2_acc<_FpType>(__lo_trunc, &__t_hi, &__t_lo);
    *__res_hi = __t_hi;
    *__res_lo = __t_lo;
    return;
  }

  // Fast small-magnitude path:
  // Start from trunc(x_hi), then apply at most a +/-1 correction only when
  // x_hi is already integral and x_lo nudges the exact value across that integer.
  const _FpType __n = ::cuda::__fpmp_internal_trunc<_FpType>(__x_hi);
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
    ::cuda::__fpmp2_acc<_FpType>(static_cast<_FpType>(__delta), &__t_hi, &__t_lo);
    *__res_hi = __t_hi;
    *__res_lo = __t_lo;
    return;
  }

  *__res_hi = __n;
  *__res_lo = _FpType(0);
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
 * Two regimes, as in trunc and rint:
 *
 *   |hi| >= int_scale: hi is an integer and the fraction is lo's own,
 *     fr = lo - trunc(lo), exact.  Where fr points away from zero (the
 *     sign of hi), |x| = |hi + i| + |fr| and the result steps past
 *     |fr| >= 1/2; where it points toward zero, |x| = |hi + i| - |fr|,
 *     trunc(x) is hi + i - s and the step back is taken unless
 *     |fr| > 1/2.  i + adj is exact: a nonzero fr means |lo| < 2^23
 *     (fp32), and the pair is renormalized by an error-free sum.
 *
 *   |hi| <  int_scale: hi carries the fraction; |lo| <= ulp(hi)/2 <= 1/4
 *     cannot reach a half on its own and only decides an exact half of
 *     hi, in the direction of its sign.
 *
 * The former form added +-1/2 to the pair and floored: once |lo| reaches
 * 2^23 the limb's ulp is already 1, the half was rounded away, and round(n)
 * came back as n +- 1 on an integer-valued pair.  Taking x - trunc(x) limb
 * by limb is not safe either: trunc may renormalize the same integer with
 * hi one ulp away, and lo - trunc(x)_lo then rounds.
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

  const _FpType __zero      = _FpType(0);
  const _FpType __half      = _FpType(0.5);
  const _FpType __one       = _FpType(1);
  const _FpType __abs_hi    = ::cuda::__fpmp_internal_fabs(__x_hi);
  const _FpType __int_scale = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p23f) : _FpType(0x1.0p52);

  if (__abs_hi >= __int_scale)
  {
    const _FpType __i   = ::cuda::__fpmp_internal_trunc<_FpType>(__x_lo);
    const _FpType __fr  = __x_lo - __i; /* exact */
    const _FpType __afr = ::cuda::__fpmp_internal_fabs(__fr);
    const _FpType __s   = (__x_hi < __zero) ? -__one : __one;
    _FpType __adj       = __zero;
    if (__fr != __zero)
    {
      __adj = ((__fr < __zero) == (__x_hi < __zero))
              ? ((__afr >= __half) ? __s : __zero)
              : ((__afr > __half) ? -__s : __zero);
    }
    *__res_hi = ::cuda::__fpmp_two_sum(__x_hi, __i + __adj, __res_lo);
    return;
  }

  if (__x_hi == __zero)
  {
    /* A renormalized pair has lo = 0 here (round(+-0) = +-0); a loose one is the scalar case. */
    const _FpType __i  = ::cuda::__fpmp_internal_trunc<_FpType>(__x_lo);
    const _FpType __fr = __x_lo - __i;
    *__res_hi = __i + ((::cuda::__fpmp_internal_fabs(__fr) >= __half) ? ((__x_lo < __zero) ? -__one : __one) : __zero);
    *__res_lo = __zero;
    return;
  }

  const _FpType __th  = ::cuda::__fpmp_internal_trunc<_FpType>(__x_hi);
  const _FpType __fh  = __x_hi - __th; /* exact */
  const _FpType __afh = ::cuda::__fpmp_internal_fabs(__fh);
  const _FpType __s   = (__x_hi < __zero) ? -__one : __one;
  bool __away;
  if (__afh > __half)
  {
    __away = true;
  }
  else if (__afh < __half)
  {
    __away = false; /* includes an integral hi with |lo| <= 1/4 */
  }
  else
  {
    __away = (__x_lo == __zero) || ((__x_lo < __zero) == (__x_hi < __zero));
  }
  *__res_hi = __away ? __th + __s : __th; /* exact: |th| < int_scale */
  *__res_lo = __zero;
}

/* Internal helper: parity of an integral value, for the rint tie-break. */
template <typename _FpType>
_CCCL_FPMP_CORE_API bool __internal_fpmp2_nearint_is_odd(const _FpType __n) noexcept
{
  // n is an integer. n/2 and 2*floor(n/2) are both exact, so this is a parity
  // test that also works past 2^53, where every value is even anyway.
  const _FpType __half_n = __n * _FpType(0.5);
  return __n != _FpType(2) * ::cuda::__fpmp_internal_floor<_FpType>(__half_n);
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
  const _FpType __abs_hi    = ::cuda::__fpmp_internal_fabs(__x_hi);
  const _FpType __int_scale = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p23f) : _FpType(0x1.0p52);

  if (__abs_hi >= __int_scale)
  {
    // Compare against the midpoint rather than subtracting: x - floor(x) is not
    // always exact (for x = -(0.5 - 2^-54) the true fraction 0.5 + 2^-54 rounds to
    // 0.5 and fabricates a tie), while floor(x) + 1/2 is representable below
    // int_scale and the comparison is exact.
    const _FpType __lo_floor = ::cuda::__fpmp_internal_floor<_FpType>(__x_lo);
    const _FpType __lo_mid   = __lo_floor + __half;
    _FpType __lo_r           = __lo_floor;
    if (__x_lo > __lo_mid)
    {
      __lo_r = __lo_floor + _FpType(1);
    }
    else if (__x_lo == __lo_mid)
    {
      // Tie: step to whichever of hi + lo_floor and hi + lo_floor + 1 is even.
      const bool __sum_odd = ::cuda::__internal_fpmp2_nearint_is_odd<_FpType>(__x_hi)
                          != ::cuda::__internal_fpmp2_nearint_is_odd<_FpType>(__lo_floor);
      if (__sum_odd)
      {
        __lo_r = __lo_floor + _FpType(1);
      }
    }

    _FpType __t_hi = __x_hi, __t_lo = _FpType(0);
    ::cuda::__fpmp2_acc<_FpType>(__lo_r, &__t_hi, &__t_lo);
    *__res_hi = __t_hi;
    *__res_lo = __t_lo;
    return;
  }

  // Same midpoint comparison as above. hi strictly on one side of mid puts x on
  // that side too, since hi and mid are both multiples of ulp(hi) here and
  // |lo| <= ulp(hi)/2, so lo cannot cross the midpoint - it can only break a tie.
  const _FpType __n   = ::cuda::__fpmp_internal_floor<_FpType>(__x_hi);
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
    else if (__x_lo == _FpType(0) && ::cuda::__internal_fpmp2_nearint_is_odd<_FpType>(__n))
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
  ::cuda::__fpmp2_rint<_FpType>(__x_hi, __x_lo, __res_hi, __res_lo);
}

/*
 * ====================================================================
 * llrint(x) - round to nearest long long, ties to even
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round to long long llrint(x) (fp32mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * The exact rint of the pair, converted limb by limb (both limbs are
 * integers then, and |lo| <= ulp(hi)/2 keeps the sum in range).  NaN,
 * infinities and |x| >= 2^63 go to the platform's own scalar function on
 * the high limb, so those inputs keep the sentinel it reports (the only
 * path through a double, and it never computes).
 */
_CCCL_FPMP_CORE_API long long int __internal_fpmp2_llrint(const float __x_hi, const float __x_lo) noexcept
{
  if (!(::cuda::__fpmp_internal_fabs(__x_hi) < 0x1p63f))
  {
    return ::cuda::std::llrint(static_cast<double>(__x_hi));
  }
  float __r_hi, __r_lo;
  ::cuda::__fpmp2_rint<float>(__x_hi, __x_lo, &__r_hi, &__r_lo);
  return static_cast<long long int>(__r_hi) + static_cast<long long int>(__r_lo);
}

/*
 * --------------------------------------------------------------------
 * Round to long long llrint(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API long long int __internal_fpmp2_llrint(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::std::llrint(::cuda::__fpmp2_to_double(__x_hi, __x_lo));
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETLL(llrint)

/*
 * ====================================================================
 * llround(x) - round to nearest long long, ties away from zero
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round to long long llround(x) (fp32mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * The exact round of the pair, converted limb by limb (both limbs are
 * integers then, and |lo| <= ulp(hi)/2 keeps the sum in range).  NaN,
 * infinities and |x| >= 2^63 go to the platform's own scalar function on
 * the high limb, so those inputs keep the sentinel it reports (the only
 * path through a double, and it never computes).
 */
_CCCL_FPMP_CORE_API long long int __internal_fpmp2_llround(const float __x_hi, const float __x_lo) noexcept
{
  if (!(::cuda::__fpmp_internal_fabs(__x_hi) < 0x1p63f))
  {
    return ::cuda::std::llround(static_cast<double>(__x_hi));
  }
  float __r_hi, __r_lo;
  ::cuda::__fpmp2_round<float>(__x_hi, __x_lo, &__r_hi, &__r_lo);
  return static_cast<long long int>(__r_hi) + static_cast<long long int>(__r_lo);
}

/*
 * --------------------------------------------------------------------
 * Round to long long llround(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API long long int __internal_fpmp2_llround(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::std::llround(::cuda::__fpmp2_to_double(__x_hi, __x_lo));
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETLL(llround)

/*
 * ====================================================================
 * lrint(x) - round to nearest long, ties to even
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round to long lrint(x) (fp32mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API long int __internal_fpmp2_lrint(const float __x_hi, const float __x_lo) noexcept
{
  return static_cast<long int>(::cuda::__internal_fpmp2_llrint(__x_hi, __x_lo));
}

/*
 * --------------------------------------------------------------------
 * Round to long lrint(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API long int __internal_fpmp2_lrint(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::std::lrint(::cuda::__fpmp2_to_double(__x_hi, __x_lo));
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETL(lrint)

/*
 * ====================================================================
 * lround(x) - round to nearest long, ties away from zero
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Round to long lround(x) (fp32mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API long int __internal_fpmp2_lround(const float __x_hi, const float __x_lo) noexcept
{
  return static_cast<long int>(::cuda::__internal_fpmp2_llround(__x_hi, __x_lo));
}

/*
 * --------------------------------------------------------------------
 * Round to long lround(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API long int __internal_fpmp2_lround(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::std::lround(::cuda::__fpmp2_to_double(__x_hi, __x_lo));
}

_CCCL_FPMP_MATH_DISPATCH_1A_RETL(lround)

/*
 * ====================================================================
 * remquo(x, y, &q) - remainder and low bits of the quotient
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Remainder and quotient remquo(x, y, &q) (fp32mp2) - exact on the limb pair
 * --------------------------------------------------------------------
 * The remainder is remainder's, bit for bit (same core); the quotient is
 * the low bits of rint(x / y) with the sign of x / y, taken modulo the
 * modulus the platform's own remquo uses (2^3 on the device, as CUDA's,
 * 2^31 on the host, as glibc's; C requires at least 2^3), so that the
 * fp32mp2 and the scalar / fp64mp2 results agree on each platform.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_remquo_core(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo,
  int* __quo,
  const bool __exact) noexcept
{
  unsigned long long __n;
  ::cuda::__internal_fpmp2_remainder_core(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, &__n, __exact);
  int __q;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__q = static_cast<int>(__n & 7ULL);), (__q = static_cast<int>(__n & 0x7FFFFFFFULL);))
  const bool __neg = (__x_hi < 0.0f) != (__y_hi < 0.0f);
  *__quo           = __neg ? -__q : __q;
}
_CCCL_FPMP_CORE_API void __internal_fpmp2_remquo(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo,
  int* __quo) noexcept
{
  ::cuda::__internal_fpmp2_remquo_core(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, __quo, false);
}
/* fpmp2_accuracy::high: exact on sparse pairs as well */
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_remquo(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo,
  int* __quo) noexcept
{
  ::cuda::__internal_fpmp2_remquo_core(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, __quo, true);
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
  ::cuda::__fpmp2_from_double(
    ::cuda::std::remquo(::cuda::__fpmp2_to_double(__x_hi, __x_lo), ::cuda::__fpmp2_to_double(__y_hi, __y_lo), __quo),
    __res_hi,
    __res_lo);
}

/* fp64mp2 has one remquo for every level */
_CCCL_FPMP_CORE_API void __internal_fpmp2_high_remquo(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo,
  int* __quo) noexcept
{
  ::cuda::__internal_fpmp2_remquo(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, __quo);
}
_CCCL_FPMP_MATH_DISPATCH_2A_QUO(remquo)
_CCCL_FPMP_MATH_DISPATCH_2A_QUO(high_remquo)

#endif // _CCCL_FPMP_USE_LIB
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_NEARINT_H
