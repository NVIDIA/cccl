//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_IMPL_DIVSQRT_H
#define _CUDA___FP_FPMP_IMPL_DIVSQRT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
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
#include <cuda/__fp/fpmp_impl_muladd.h> // div/sqrt reuse __fpmp2_low_mul (muladd family)

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if !(defined _CCCL_FPMP_USE_LIB)
/*
 * --------------------------------------------------------------------
 * Division operations
 * --------------------------------------------------------------------
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_low_div(
  const _FpType __a_hi,
  const _FpType __a_lo,
  const _FpType __b_hi,
  const _FpType __b_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  // Get an estimate from *this->hi:
  _FpType __recip_hi = __fpmp_rcp_rn(__b_hi);

  // Do a Newton-Raphson iteration:
  // This line can break for some uninvestigated reason,
  // Use the one below:
  // recip_hi = recip_hi*(2.0 - (x.get_hi())*recip_hi);
  _FpType __two = static_cast<_FpType>(2.0);
  __recip_hi    = __fpmp_fma_rn(-__b_hi * __recip_hi, __recip_hi, __two * __recip_hi);

  _FpType __recip2_hi = __recip_hi * __recip_hi;
  _FpType __recip2_lo = __fpmp_fma_rn(__recip_hi, __recip_hi, -__recip2_hi);

  // recip^2 * this->(hi/lo), Dekker multiplication:
  _FpType __mul_hi = __recip2_hi * (__b_hi);
  _FpType __mul_lo = __fpmp_fma_rn(__recip2_hi, (__b_hi), -__mul_hi);
  __mul_lo += (__recip2_hi * (__b_lo) + __recip2_lo * (__b_hi));

  // Our answer is now 2*recip_hi + mul_hi + mul_lo
  _FpType __final_recip_hi = __two * __recip_hi - __mul_hi;
  _FpType __final_recip_lo = __two * __recip_hi - __fpmp_add_rn(__final_recip_hi, __mul_hi);
  __final_recip_lo -= __mul_lo;

  // Multiply the reciprocal by the numerator
  __fpmp2_low_mul(__a_hi, __a_lo, __final_recip_hi, __final_recip_lo, __res_hi, __res_lo);
} // __fpmp2_low_div

/* Compute high-accuracy quotient, using Newton-
Raphson iteration. Derived from: T. Nagai, H. Yoshida, H. Kuroda, Y. Kanada.
Fast Quadruple Precision Arithmetic Library on Parallel Computer SR11000/J2.
In Proceedings of the 8th International Conference on Computational Science,
ICCS '08, Part I, pp. 446-455.
*/
template <typename _FpType = float>
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
  __r    = __fpmp_rcp_rn(__b_hi);
  __t_hi = __fpmp_mul_rn(__a_hi, __r);
  __e    = __fpmp_fma_rn(__b_hi, -__t_hi, __a_hi);
  __t_hi = __fpmp_fma_rn(__r, __e, __t_hi);
  __t_lo = __fpmp_fma_rn(__b_hi, -__t_hi, __a_hi);
  __t_lo = __fpmp_add_rn(__a_lo, __t_lo);
  __t_lo = __fpmp_fma_rn(__b_lo, -__t_hi, __t_lo);
  __e    = __fpmp_mul_rn(__r, __t_lo);
  __t_lo = __fpmp_fma_rn(__b_hi, -__e, __t_lo);
  __t_lo = __fpmp_fma_rn(__r, __t_lo, __e);
  __e    = __fpmp_add_rn(__t_hi, __t_lo);

  *__res_lo = __fpmp_add_rn(__t_hi - __e, __t_lo);
  *__res_hi = __e;
} // __fpmp2_div

#  if _CCCL_FPMP_USE_ACCURATE_DIV == 1
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
 * Reference:
 *   Nagai, Yoshida, Kuroda, Kanada (2008). Fast Quadruple Precision
 *   Arithmetic Library on Parallel Computer SR11000/J2.
 *   Conditional scaling adapted from QD library techniques.
 */
template <typename _FpType = float>
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
  const _FpType __sa_hi = __fpmp_mul_rn(__a_hi, __scale_a);
  const _FpType __sa_lo = __fpmp_mul_rn(__a_lo, __scale_a);
  const _FpType __sb_hi = __fpmp_mul_rn(__b_hi, __scale_b);
  const _FpType __sb_lo = __fpmp_mul_rn(__b_lo, __scale_b);

  // Perform division on scaled operands using Nagai et al. algorithm
  _FpType __t_hi, __t_lo;
  _FpType __e, __r;
  __r    = __fpmp_rcp_rn(__sb_hi);
  __t_hi = __fpmp_mul_rn(__sa_hi, __r);
  __e    = __fpmp_fma_rn(__sb_hi, -__t_hi, __sa_hi);
  __t_hi = __fpmp_fma_rn(__r, __e, __t_hi);
  __t_lo = __fpmp_fma_rn(__sb_hi, -__t_hi, __sa_hi);
  __t_lo = __fpmp_add_rn(__sa_lo, __t_lo);
  __t_lo = __fpmp_fma_rn(__sb_lo, -__t_hi, __t_lo);
  __e    = __fpmp_mul_rn(__r, __t_lo);
  __t_lo = __fpmp_fma_rn(__sb_hi, -__e, __t_lo);
  __t_lo = __fpmp_fma_rn(__r, __t_lo, __e);
  __e    = __fpmp_add_rn(__t_hi, __t_lo);

  _FpType __r_hi = __e;
  _FpType __r_lo = __fpmp_add_rn(__t_hi - __e, __t_lo);

  // Undo the operand scaling on the result: a/b = (sa/sb) * (scale_b / scale_a).
  //   inv_scale_a = 1 / scale_a   (a up -> 2^-K, a down -> 2^+K, none -> 1)
  //   times scale_b               (b up -> 2^+K, b down -> 2^-K, none -> 1)
  // Each factor is a normal power of two; their product is exact unless the true
  // quotient itself overflows/underflows (a genuine range result, not an artifact).
  const UintType __inv_scale_a_bits =
    (__down_bits & UintType(__a_up)) | (__up_bits & UintType(__a_down)) | (__one_bits & UintType(~(__a_up | __a_down)));
  const _FpType __inv_scale_a = ::cuda::std::bit_cast<_FpType>(__inv_scale_a_bits);
  const _FpType __final_scale = __fpmp_mul_rn(__inv_scale_a, __scale_b);

  // Scale result back
  __r_hi = __fpmp_mul_rn(__r_hi, __final_scale);
  __r_lo = __fpmp_mul_rn(__r_lo, __final_scale);

  // Final normalization to ensure (hi, lo) invariant after scaling
  *__res_hi = __fpmp_fast_two_sum(__r_hi, __r_lo, __res_lo);
} // __fpmp2_high_div
#  endif // _CCCL_FPMP_USE_ACCURATE_DIV == 1

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
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void
__fpmp2_rsqrt(const _FpType __a_hi, const _FpType __a_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  _FpType __z_hi, __z_lo;
  _FpType __r, __s, __e;
  _FpType __one  = static_cast<_FpType>(1.0);
  _FpType __half = static_cast<_FpType>(0.5);
  __r            = __fpmp_rsqrt_rn(__a_hi);
  __e            = __fpmp_mul_rn(__a_hi, __r);
  __s            = __fpmp_fma_rn(__e, -__r, __one);
  __e            = __fpmp_fma_rn(__a_hi, __r, -__e);
  __s            = __fpmp_fma_rn(__e, -__r, __s);
  __e            = __fpmp_mul_rn(__a_lo, __r);
  __s            = __fpmp_fma_rn(__e, -__r, __s);
  __e            = __fpmp_mul_rn(__half, __r);
  __z_hi         = __fpmp_mul_rn(__e, __s);
  __z_lo         = __fpmp_fma_rn(__e, __s, -__z_hi);
  __s            = __fpmp_add_rn(__r, __z_hi);
  __r            = __fpmp_add_rn(__r, -__s);
  __r            = __fpmp_add_rn(__r, __z_hi);
  __r            = __fpmp_add_rn(__r, __z_lo);
  __e            = __fpmp_add_rn(__s, __r);
  __z_lo         = __fpmp_add_rn(__s - __e, __r);
  __z_hi         = __e;

  *__res_hi = __z_hi;
  *__res_lo = __z_lo;
} // __fpmp2_rsqrt

/* Compute high-accuracy square root. Newton-Raphson
iteration based on equation 4 from a paper by Alan Karp and Peter Markstein,
High Precision Division and Square Root, ACM TOMS, vol. 23, no. 4, December
1997, pp. 561-589.
*/
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void
__fpmp2_sqrt(const _FpType __a_hi, const _FpType __a_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  _FpType __t_hi, __t_lo, __tmp_lo;
  _FpType __e, __y, __s, __r;
  _FpType __zero = static_cast<_FpType>(0.0);
  _FpType __half = static_cast<_FpType>(0.5);
  __r            = __fpmp_rsqrt_rn(__a_hi);
  if (__a_hi == __zero)
  {
    __r = __zero;
  }
  __y      = __fpmp_mul_rn(__a_hi, __r);
  __s      = __fpmp_fma_rn(__y, -__y, __a_hi);
  __r      = __fpmp_mul_rn(__half, __r);
  __e      = __fpmp_add_rn(__s, __a_lo);
  __tmp_lo = __fpmp_add_rn(__s - __e, __a_lo);
  __t_hi   = __fpmp_mul_rn(__r, __e);
  __t_lo   = __fpmp_fma_rn(__r, __e, -__t_hi);
  __t_lo   = __fpmp_fma_rn(__r, __tmp_lo, __t_lo);
  __r      = __fpmp_add_rn(__y, __t_hi);
  __s      = __fpmp_add_rn(__y - __r, __t_hi);
  __s      = __fpmp_add_rn(__s, __t_lo);
  __e      = __fpmp_add_rn(__r, __s);

  *__res_lo = __fpmp_add_rn(__r - __e, __s);
  *__res_hi = __e;
} // __fpmp2_sqrt

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
#  if _CCCL_FPMP_USE_ACCURATE_DIV == 1
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_high_div(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
#  endif // _CCCL_FPMP_USE_ACCURATE_DIV == 1
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_sqrt(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_rsqrt(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept;

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
#  if _CCCL_FPMP_USE_ACCURATE_DIV == 1
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_high_div(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
#  endif // _CCCL_FPMP_USE_ACCURATE_DIV == 1
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_sqrt(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_rsqrt(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept;

// -- type-generic template declarations (dispatch to fp32/fp64) --
template <typename _Tp>
_CCCL_API inline void __fpmp2_div(
  const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_mid_div(
  const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_low_div(
  const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
#  if _CCCL_FPMP_USE_ACCURATE_DIV == 1
template <typename T>
_CCCL_API inline void
__fpmp2_high_div(const T __a_hi, const T __a_lo, const T __b_hi, const T __b_lo, T* __res_hi, T* __res_lo) noexcept;
#  endif // _CCCL_FPMP_USE_ACCURATE_DIV == 1
template <typename _Tp>
_CCCL_API inline void __fpmp2_sqrt(const _Tp __a_hi, const _Tp __a_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_rsqrt(const _Tp __a_hi, const _Tp __a_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;

// -- fp32 template specializations --
template <>
_CCCL_API inline void __fpmp2_div<float>(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_div<float>(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_mid_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_div<float>(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_low_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
#  if _CCCL_FPMP_USE_ACCURATE_DIV == 1
template <>
_CCCL_API inline void __fpmp2_high_div<float>(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_high_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
#  endif // _CCCL_FPMP_USE_ACCURATE_DIV == 1
template <>
_CCCL_API inline void
__fpmp2_sqrt<float>(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_sqrt(__a_hi, __a_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_rsqrt<float>(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_rsqrt(__a_hi, __a_lo, __res_hi, __res_lo);
}

// -- fp64 template specializations --
template <>
_CCCL_API inline void __fpmp2_div<double>(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_div<double>(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_mid_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_div<double>(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_low_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
#  if _CCCL_FPMP_USE_ACCURATE_DIV == 1
template <>
_CCCL_API inline void __fpmp2_high_div<double>(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_high_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
#  endif // _CCCL_FPMP_USE_ACCURATE_DIV == 1
template <>
_CCCL_API inline void
__fpmp2_sqrt<double>(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_sqrt(__a_hi, __a_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_rsqrt<double>(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_rsqrt(__a_hi, __a_lo, __res_hi, __res_lo);
}

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_IMPL_DIVSQRT_H
