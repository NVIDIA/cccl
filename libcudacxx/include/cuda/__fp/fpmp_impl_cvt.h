//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_IMPL_CVT_H
#define _CUDA___FP_FPMP_IMPL_CVT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_impl_cvt.h - fpmp2 conversions, casts and bit-cast
    ==================================================================================================
    Per-operation implementation core split out of <cuda/__fp/fpmp_impl.h>. It carries the
    conversions, casts and bit-cast
    for the fpmp2 double-word type, for both the header-only (inline) mode and the library
    (_CCCL_FPMP_USE_LIB) mode. All shared macros, the fp128 vocabulary type, and the __fpmp_*
    error-free-transform primitives live in <cuda/__fp/fpmp_impl.h>, which this header includes.
*/

#include <cuda/__fp/fpmp_impl.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if !(defined _CCCL_FPMP_USE_LIB)
/*
 * --------------------------------------------------------------------
 * Conversion operations
 * --------------------------------------------------------------------
 */
/*
// -----------------------------------------------------------------------
// __fpmp2_from_double: Convert double → fpmp2 (hi, lo) pair
// -----------------------------------------------------------------------
// Splits a 64-bit double into two FpType components such that:
//   x ≈ hi + lo    (with hi carrying the leading bits, lo the remainder)
//
// When CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP == 1 and FpType == float:
//   Uses integer bit manipulation to extract the hi and lo components
//   directly from the IEEE 754 double bit pattern, avoiding all FP64
//   arithmetic. This is beneficial on GPUs with limited FP64 throughput
//   (e.g., consumer GPUs with 1:64 FP64:FP32 ratio).
//
//   Notes:
//   * hi rounding is round-to-nearest, ties AWAY-FROM-ZERO (single
//     add+shift), instead of ties-to-even. As a result, hi may differ
//     from (float)x at exact tie midpoints; for non-tie inputs the
//     two rules agree.
//   * lo is computed by reinterpreting the bottom 29 mantissa bits as
//     a signed 32-bit integer (the round bit is placed at the sign
//     position so a rounded-up hi automatically yields a negative
//     residual via two's complement), converting it to float with
//     round-to-nearest-even, and rescaling by exact powers of two.
//   * A final Fast2Sum re-establishes the canonical fl(hi+lo) == hi
//     invariant. This is required because the round-to-nearest lo
//     can land at +/-ulp(hi)/2 and overflow the canonical range.
//
// When CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP == 0 or FpType != float:
//   Uses the standard cast-based approach:
//     hi = (FpType)x;  lo = (FpType)(x - (double)hi);
//   This relies on two FP64 operations (cast + subtract).
//
// IEEE 754 bit layout reference:
//   double (64-bit): [1 sign][11 exponent][52 mantissa]
//   float  (32-bit): [1 sign][ 8 exponent][23 mantissa]
//   Exponent bias: double = 1023, float = 127, difference = 896
//
// The 52-bit double mantissa is split into:
//   - hi: top 23 bits  → float mantissa (bits [29:51] of double mantissa)
//   - lo: bottom 29 bits → second float  (bits [0:28] of double mantissa)
// -----------------------------------------------------------------------
*/
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_from_double(const double __x, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
#  if _CCCL_FPMP_USE_OPT_FROM_DOUBLE == 1
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    uint64_t __dbits = ::cuda::std::bit_cast<uint64_t>(__x);
    uint32_t __sign  = (uint32_t) (__dbits >> 63);
    uint32_t __d_exp = (uint32_t) ((__dbits >> 52) & 0x7FFU);
    uint64_t __mant  = __dbits & 0x000FFFFFFFFFFFFFULL;

    // hi biased exponent in float space: f_exp = (d_exp - 1023) + 127.
    int32_t __f_exp = (int32_t) __d_exp - 896;

    // Fallback for: zero/denormal double (d_exp == 0), float underflow
    // (f_exp <= 0), and float overflow / Inf / NaN (f_exp >= 255).
    // Defers to the standard cast for these edge cases; lo is flushed.
    if (__d_exp == 0 || (__f_exp <= 0 || __f_exp >= 255))
    {
      *__res_hi = (float) __x;
      *__res_lo = 0.0f;
    }
    else
    {
      // hi mantissa: top 23 explicit bits with round-to-nearest,
      // ties away from zero. (((mant >> 28) + 1) >> 1) takes the
      // top 24 bits of mant, adds 1 at the round position, then
      // drops it. The carry can ripple into the exponent (when
      // hi_round == 0x800000), so we use '+' (not '|') to merge
      // hi_round into the shifted exponent field.
      uint32_t __hi_round = (((uint32_t) (__mant >> 28)) + 1U) >> 1;
      uint32_t __hi_bits  = (__sign << 31) | (((uint32_t) __f_exp << 23) + __hi_round);
      *__res_hi           = ::cuda::std::bit_cast<float>(__hi_bits);

      // Encode the residual as a signed 32-bit integer with the
      // round bit placed at the sign position. Bottom 32 bits of
      // mant are bits [31:0]; shifting left by 3 in 32-bit
      // arithmetic discards bits [31:29] (already absorbed into
      // hi) and places bit 28 (the round bit) at bit 31. When the
      // round bit is 1 (hi was rounded up), rsd is negative in
      // two's complement, exactly representing the signed residual
      // x - hi at mantissa scale * 2^3.
      int32_t __rsd = (int32_t) ((uint32_t) __mant << 3);

      // Convert rsd to float with round-to-nearest-even (default for
      // host int->float and CUDA cvt.rn.f32.s32). Then scale:
      //   * 2^-55  : undoes the << 3 (-3) and the mantissa-position
      //              offset (-52) to recover residual at unit scale.
      //   * scale  : 2^(f_exp - 127) with the sign of x. Both
      //              multiplications are exact (powers of two).
      float __scale = ::cuda::std::bit_cast<float>((__sign << 31) | ((uint32_t) __f_exp << 23));
      *__res_lo     = (static_cast<float>(__rsd) * 0x1p-55f) * __scale;

      // Fast2Sum to enforce canonical form fl(hi+lo) == hi.
      // Required because round-to-nearest on r can leave |lo|
      // exactly at ulp(hi)/2; if hi has an odd low mantissa bit,
      // fl(hi+lo) would otherwise round away from hi.
      *__res_hi = __fpmp_fast_two_sum(*__res_hi, *__res_lo, __res_lo);
    }
  }
  else if constexpr (__fpmp2_is_fp64_v<_FpType>)
  {
    // FpType == double (fp64mp2): the cast-based split below would
    // compute (double)(x - (double)x) == 0.0 and the compiler folds
    // it; spell that out at the source level so the intent is
    // explicit and the lo store is guaranteed not to depend on any
    // FP64 instruction.
    *__res_hi = __x;
    *__res_lo = 0.0;
  }
  else
  {
    // Generic fallback for any future non-float, non-double FpType:
    // cast-based split.
    *__res_hi = static_cast<_FpType>(__x);
    *__res_lo = static_cast<_FpType>(__x - static_cast<double>(*__res_hi));
  }
#  else // !_CCCL_FPMP_USE_OPT_FROM_DOUBLE == 1
  if constexpr (__fpmp2_is_fp64_v<_FpType>)
  {
    // FpType == double (fp64mp2): trivial split, see comment above.
    *__res_hi = __x;
    *__res_lo = 0.0;
  }
  else
  {
    // Non-optimized path: two FP64 operations (cast + subtract).
    *__res_hi = static_cast<_FpType>(__x);
    *__res_lo = static_cast<_FpType>(__x - static_cast<double>(*__res_hi));
  }
#  endif // !_CCCL_FPMP_USE_OPT_FROM_DOUBLE == 1
} // __fpmp2_from_double

// int -> (hi, lo) conversions
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_from_int(const int32_t __i, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = __fpmp_int2fp_rz<_FpType>(__i);
  *__res_lo = __fpmp_int2fp_rz<_FpType>(__i - __fpmp_fp2int_rz(*__res_hi));
}

// uint -> (hi, lo) conversions
// Note: Use signed arithmetic to compute residual, since __fpmp_fp2uint_rz(*res_hi)
// might be larger than i when rounding direction differs
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_from_uint(const uint32_t __i, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = __fpmp_uint2fp_rz<_FpType>(__i);
  // Compute residual using signed arithmetic to handle case where hi rounds up
  int32_t __residual = static_cast<int32_t>(__i) - static_cast<int32_t>(__fpmp_fp2uint_rz(*__res_hi));
  *__res_lo          = __fpmp_int2fp_rz<_FpType>(__residual);
}

// ll -> (hi, lo) conversions
// With __fpmp_ll2fp_rz properly rounding toward zero, hi is always <= i for positive i
// and >= i for negative i, so __fpmp_fp2ll_rz(hi) is always representable as int64_t.
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_from_ll(const int64_t __i, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = __fpmp_ll2fp_rz<_FpType>(__i);
  *__res_lo = __fpmp_ll2fp_rz<_FpType>(__i - __fpmp_fp2ll_rz(*__res_hi));
}

// ull -> (hi, lo) conversions
// With ull2fp_rz properly rounding toward zero, hi <= i always,
// so the residual i - hi is always non-negative.
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_from_ull(const uint64_t __i, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = __fpmp_ull2fp_rz<_FpType>(__i);
  // Residual is always non-negative and fits in int64_t (< 2^53 for double)
  uint64_t __residual = __i - __fpmp_fp2ull_rz(*__res_hi);
  *__res_lo           = __fpmp_ull2fp_rz<_FpType>(__residual);
}

// (hi, lo) -> double conversions
//
// Optimized path (_CCCL_FPMP_USE_OPT_TO_DOUBLE == 1):
//   Reconstructs the IEEE 754 double bit pattern from two float
//   components using FP32 + integer arithmetic. No FP64 instructions on
//   the hot path — avoids the slow FP64 pipeline on GPUs with limited
//   double-precision throughput (1:64 ratio on consumer GPUs).
//
//   Assumes both inputs are normal floats (no denormals, inf, or NaN).
//   This is always true for well-formed fp32mp2 values produced by the
//   library's double→fp32mp2 conversion.
//
//   Step 0 — input renormalization (2Sum):
//     The hot path below relies on the canonical-form invariant
//     |lo| <= ulp(hi)/2 in two places:
//       (a) x_lo * scale must be exact (24-bit mantissa headroom);
//       (b) the resulting r must fit in int32 (|r| < 2^31, i.e.
//           |lo|/|hi| < 2^-21).
//     Pairs produced by add_fast or long FAST accumulator chains may
//     have |lo|/|hi| as large as ~2^-8, which silently overflows r.
//     A 2Sum at the top makes (hi, lo) canonical (|err| <= ulp(s)/2)
//     for any input magnitudes — 6 FP32 ops on the hot path, runs
//     largely in parallel with the bit extraction below on the FP/INT
//     pipes.
//
//   Hot-path algorithm:
//     1. Build a signed power-of-two scale = (sign_a ? -1 : +1) * 2^(179 - fexp_a)
//        by direct bit construction. The sign of hi is baked into the
//        scale so the next step yields lo's contribution *relative to hi*
//        regardless of lo's own sign — no same-sign / diff-sign branch.
//     2. r = (int32_t)(x_lo * scale)  — one FMUL + one F2I. For canonical
//        fp32mp2 (|lo| <= ulp(hi)/2) the multiplication is exact and
//        |r| <= 2^28. The signed r is exactly lo's contribution to the
//        double mantissa at hi's scale.
//     3. M = (mantissa-of-hi-with-implicit-1 at bit 52) + r — single
//        64-bit add. Range [2^52 - 2^28, 2^53 - 2^29 + 2^28], always > 0.
//     4. Renormalize by at most 1 bit (subtraction can borrow the
//        implicit-1 down to bit 51). Single conditional shift; no CLZ.
//     5. Splice sign / biased-exponent / mantissa into the double.
//
//   Cold-path (FP64 fallback) covers cases where the FP32 scale would
//   overflow / be ill-defined:
//     - fexp_a == 0      : hi is +/-0 or subnormal
//     - fexp_a in [1,51] : 2^(179 - fexp_a) overflows float (need biased
//                          exponent <= 254 ⇒ fexp_a >= 52)
//     - fexp_a == 0xFF   : hi is +/-Inf or NaN
//   Both ends of this range are extremely rare for typical fp32mp2 data.
//
// Non-optimized path:
//   static_cast<double>(x_hi) + static_cast<double>(x_lo)
//   (2x F2D + 1x DADD = 3 FP64 operations)
//
template <typename _FpType = float>
_CCCL_FPMP_CORE_API double __fpmp2_to_double(const _FpType __x_hi_in, const _FpType __x_lo_in) noexcept
{
#  if _CCCL_FPMP_USE_OPT_TO_DOUBLE == 1
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    // Renormalize input to canonical form. See "Step 0" in the
    // function's docstring for why this is required for the integer
    // path to be safe on non-canonical pairs (e.g. produced by
    // add_fast / long FAST accumulator chains).
    float __x_lo;
    float __x_hi = __fpmp_two_sum(__x_hi_in, __x_lo_in, &__x_lo);

    uint32_t __hi_bits = ::cuda::std::bit_cast<uint32_t>(__x_hi);
    uint32_t __sign_a  = __hi_bits >> 31;
    uint32_t __fexp_a  = (__hi_bits >> 23) & 0xFFU;
    uint32_t __fmant_a = __hi_bits & 0x7FFFFFU;

    // Cold fallback for hi outside the FP32-scale-representable range:
    // zero/subnormal, very tiny normal (fexp_a in [1, 51]), Inf, NaN.
    // NB: sum the ORIGINAL inputs, not the 2Sum-renormalized (__x_hi, __x_lo).
    // The 2Sum prologue exists only to canonicalize finite operands for the
    // integer hot path; on special values it poisons the low word (e.g.
    // two_sum(inf, 0) yields lo = inf - inf = NaN), which would turn a clean
    // (inf, 0) / (nan, 0) into NaN here. The plain double sum of the originals
    // handles inf/nan/zero/subnormal correctly.
    if (__fexp_a < 52U || __fexp_a == 0xFFU)
    {
      return static_cast<double>(__x_hi_in) + static_cast<double>(__x_lo_in);
    }

    // scale = (sign_a ? -1 : +1) * 2^(179 - fexp_a). Always a normal
    // float for fexp_a in [52, 254] (biased scale_exp = 306 - fexp_a
    // in [52, 254]). Sign of hi baked in so the signed r below is
    // already lo's contribution relative to hi.
    float __scale = ::cuda::std::bit_cast<float>((__sign_a << 31) | ((306U - __fexp_a) << 23));

    // r exactly represents lo at hi's mantissa scale (signed). For
    // canonical fp32mp2 (|lo| <= ulp(hi)/2) the multiplication is
    // exact (power-of-two scaling) and |r| <= 2^28.
    int32_t __r = __fpmp_fp2int_rn(__x_lo * __scale);

    // M = (hi's 53-bit mantissa with implicit 1, at bit 52)
    //     + (signed lo contribution at the same scale).
    // Range: [2^52 - 2^28, 2^53 - 2^29 + 2^28]. Always positive.
    int64_t __m = (int64_t) (((uint64_t) (0x800000U | __fmant_a)) << 29) + (int64_t) __r;

    // Subtraction can borrow at most one bit (|r| << 2^52), so the
    // implicit-1 lands at bit 52 (no shift) or bit 51 (shift up by 1).
    // Single conditional shift, no __clzll on the critical path.
    uint64_t __mu         = (uint64_t) __m;
    uint64_t __need_shift = ((__mu >> 52) & 1ULL) ^ 1ULL;
    __mu <<= __need_shift;

    return ::cuda::std::bit_cast<double>(
      ((uint64_t) __sign_a << 63) | ((uint64_t) (__fexp_a + 896U - (uint32_t) __need_shift) << 52)
      | (__mu & 0x000FFFFFFFFFFFFFULL));
  }
  else
  {
    return static_cast<double>(__x_hi_in) + static_cast<double>(__x_lo_in);
  }
#  else
  return static_cast<double>(__x_hi_in) + static_cast<double>(__x_lo_in);
#  endif
}

// (hi, lo) -> float conversions (returns the sum as single FpType)
template <typename _FpType = float>
_CCCL_FPMP_CORE_API _FpType __fpmp2_to_float(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  return __x_hi + __x_lo;
}

// (hi, lo) -> int conversions
template <typename _FpType = float>
_CCCL_FPMP_CORE_API int32_t __fpmp2_to_int(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  _FpType __abs_hi = __fpmp_internal_fabs(__x_hi);
  // Check threshold BEFORE computing sum - for large values, addition loses precision
  // 2^24 for float, 2^53 for double
  _FpType __threshold = __fpmp2_is_fp32_v<_FpType> ? 0x1.0p24f : 0x1.0p53;
  if (__abs_hi < __threshold)
  {
    // Small value: use round-toward-zero addition
    _FpType __res = __fpmp_add_rz(__x_hi, __x_lo);
    return __fpmp_fp2int_rz(__res);
  }
  else
  {
    // Large value: use integer addition to preserve exactness
    int32_t __hi_int = __fpmp_fp2int_rz(__x_hi);
    int32_t __lo_int = __fpmp_fp2int_rz(__x_lo);
    return __hi_int + __lo_int;
  }
} // __fpmp2_to_int

// (hi, lo) -> uint conversions
template <typename _FpType = float>
_CCCL_FPMP_CORE_API uint32_t __fpmp2_to_uint(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  // Check threshold BEFORE computing sum
  // 2^24 for float, 2^53 for double
  _FpType __threshold = __fpmp2_is_fp32_v<_FpType> ? 0x1.0p24f : 0x1.0p53;
  if (__x_hi < __threshold)
  {
    // Small value: use round-toward-zero addition
    _FpType __res = __fpmp_add_rz(__x_hi, __x_lo);
    return __fpmp_fp2uint_rz(__res);
  }
  else
  {
    // Large value: use integer addition to preserve exactness
    uint32_t __hi_uint = __fpmp_fp2uint_rz(__x_hi);
    int32_t __lo_int   = __fpmp_fp2int_rz(__x_lo);
    return __hi_uint + __lo_int;
  }
} // __fpmp2_to_uint

// (hi, lo) -> ll conversions
template <typename _FpType = float>
_CCCL_FPMP_CORE_API int64_t __fpmp2_to_ll(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  _FpType __abs_hi = __fpmp_internal_fabs(__x_hi);
  // Check threshold BEFORE computing sum
  // 2^24 for float, 2^53 for double
  _FpType __threshold = __fpmp2_is_fp32_v<_FpType> ? 0x1.0p24f : 0x1.0p53;
  if (__abs_hi < __threshold)
  {
    // Small value: use round-toward-zero addition
    _FpType __res = __fpmp_add_rz(__x_hi, __x_lo);
    return __fpmp_fp2ll_rz(__res);
  }
  else
  {
    // Large value: use integer addition to preserve exactness
    int64_t __hi_ll = __fpmp_fp2ll_rz(__x_hi);
    int64_t __lo_ll = __fpmp_fp2ll_rz(__x_lo);
    return __hi_ll + __lo_ll;
  }
} // __fpmp2_to_ll

// (hi, lo) -> ull conversions
template <typename _FpType = float>
_CCCL_FPMP_CORE_API uint64_t __fpmp2_to_ull(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  // Check threshold BEFORE computing sum
  // 2^24 for float, 2^53 for double
  _FpType __threshold = __fpmp2_is_fp32_v<_FpType> ? 0x1.0p24f : 0x1.0p53;
  if (__x_hi < __threshold)
  {
    // Small value: use round-toward-zero addition
    _FpType __res = __fpmp_add_rz(__x_hi, __x_lo);
    return __fpmp_fp2ull_rz(__res);
  }
  else
  {
    // Large value: use integer addition to preserve exactness
    uint64_t __hi_ull = __fpmp_fp2ull_rz(__x_hi);
    int64_t __lo_ll   = __fpmp_fp2ll_rz(__x_lo);
    return __hi_ull + __lo_ll;
  }
} // __fpmp2_to_ull

/*
 * --------------------------------------------------------------------
 * Bit cast operations (IEEE-754 format)
 * --------------------------------------------------------------------
 */
// bit_cast to IEEE-754 format bits
template <typename _FpType = float>
_CCCL_FPMP_CORE_API uint64_t __fpmp2_bit_cast(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  double __d = __fpmp2_to_double(__x_hi, __x_lo);
  return ::cuda::std::bit_cast<uint64_t>(__d);
}

// __fpmp_fp128 operations (only for FpType == double)
// available only for CUDA architectures >= 1000 or when _CCCL_FPMP_FP128_ENABLE is defined
#  if _CCCL_FPMP_FP128_ENABLE == 1
template <typename _FpType = double>
constexpr _CCCL_FPMP_CORE_API void
__fpmp2_from_quad(const __fpmp_fp128 __x, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = static_cast<_FpType>(__x);
  *__res_lo = static_cast<_FpType>(__x - static_cast<__fpmp_fp128>(*__res_hi));
}

template <typename _FpType = double>
_CCCL_FPMP_CORE_API __fpmp_fp128 __fpmp2_to_quad(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  return static_cast<__fpmp_fp128>(__x_hi) + static_cast<__fpmp_fp128>(__x_lo);
}
#  endif // _CCCL_FPMP_FP128_ENABLE == 1
#else // _CCCL_FPMP_USE_LIB

// -- fp32 (single precision) built-in declarations --
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_from_double(const double __x, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_from_int(const int32_t __i, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_from_uint(const uint32_t __i, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_from_ll(const int64_t __i, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_from_ull(const uint64_t __i, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL double __fp32mp2_to_double(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL float __fp32mp2_to_float(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int32_t __fp32mp2_to_int(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL uint32_t __fp32mp2_to_uint(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int64_t __fp32mp2_to_ll(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL uint64_t __fp32mp2_to_ull(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL uint64_t __fp32mp2_bit_cast(const float __x_hi, const float __x_lo) noexcept;

// -- fp64 (double precision) built-in declarations --
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_from_double(const double __x, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_from_int(const int32_t __i, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_from_uint(const uint32_t __i, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_from_ll(const int64_t __i, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_from_ull(const uint64_t __i, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL double __fp64mp2_to_double(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL float __fp64mp2_to_float(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int32_t __fp64mp2_to_int(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL uint32_t __fp64mp2_to_uint(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int64_t __fp64mp2_to_ll(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL uint64_t __fp64mp2_to_ull(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL uint64_t __fp64mp2_bit_cast(const double __x_hi, const double __x_lo) noexcept;
#  if _CCCL_FPMP_FP128_ENABLE == 1
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_from_quad(const __fpmp_fp128 __x, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL __fpmp_fp128 __fp64mp2_to_quad(const double __x_hi, const double __x_lo) noexcept;
#  endif // _CCCL_FPMP_FP128_ENABLE == 1

// -- type-generic template declarations (dispatch to fp32/fp64) --
template <typename _Tp>
_CCCL_API inline void __fpmp2_from_double(const double __x, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_from_int(const int32_t __i, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_from_uint(const uint32_t __i, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_from_ll(const int64_t __i, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_from_ull(const uint64_t __i, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline double __fpmp2_to_double(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline float __fpmp2_to_float(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline int32_t __fpmp2_to_int(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline uint32_t __fpmp2_to_uint(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline int64_t __fpmp2_to_ll(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline uint64_t __fpmp2_to_ull(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline uint64_t __fpmp2_bit_cast(const _Tp __x_hi, const _Tp __x_lo) noexcept;
#  if _CCCL_FPMP_FP128_ENABLE == 1
template <typename _Tp>
_CCCL_API inline void __fpmp2_from_quad(const __fpmp_fp128 __x, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline __fpmp_fp128 __fpmp2_to_quad(const _Tp __x_hi, const _Tp __x_lo) noexcept;
#  endif // _CCCL_FPMP_FP128_ENABLE == 1

// -- fp32 template specializations --
template <>
_CCCL_API inline void __fpmp2_from_double<float>(const double __x, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_from_double(__x, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_from_int<float>(const int32_t __i, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_from_int(__i, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_from_uint<float>(const uint32_t __i, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_from_uint(__i, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_from_ll<float>(const int64_t __i, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_from_ll(__i, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_from_ull<float>(const uint64_t __i, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_from_ull(__i, __res_hi, __res_lo);
}
template <>
_CCCL_API inline double __fpmp2_to_double<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_to_double(__x_hi, __x_lo);
}
template <>
_CCCL_API inline float __fpmp2_to_float<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_to_float(__x_hi, __x_lo);
}
template <>
_CCCL_API inline int32_t __fpmp2_to_int<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_to_int(__x_hi, __x_lo);
}
template <>
_CCCL_API inline uint32_t __fpmp2_to_uint<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_to_uint(__x_hi, __x_lo);
}
template <>
_CCCL_API inline int64_t __fpmp2_to_ll<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_to_ll(__x_hi, __x_lo);
}
template <>
_CCCL_API inline uint64_t __fpmp2_to_ull<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_to_ull(__x_hi, __x_lo);
}
template <>
_CCCL_API inline uint64_t __fpmp2_bit_cast<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_bit_cast(__x_hi, __x_lo);
}

// -- fp64 template specializations --
template <>
_CCCL_API inline void __fpmp2_from_double<double>(const double __x, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_from_double(__x, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_from_int<double>(const int32_t __i, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_from_int(__i, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_from_uint<double>(const uint32_t __i, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_from_uint(__i, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_from_ll<double>(const int64_t __i, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_from_ll(__i, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_from_ull<double>(const uint64_t __i, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_from_ull(__i, __res_hi, __res_lo);
}
template <>
_CCCL_API inline double __fpmp2_to_double<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_to_double(__x_hi, __x_lo);
}
template <>
_CCCL_API inline float __fpmp2_to_float<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_to_float(__x_hi, __x_lo);
}
template <>
_CCCL_API inline int32_t __fpmp2_to_int<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_to_int(__x_hi, __x_lo);
}
template <>
_CCCL_API inline uint32_t __fpmp2_to_uint<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_to_uint(__x_hi, __x_lo);
}
template <>
_CCCL_API inline int64_t __fpmp2_to_ll<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_to_ll(__x_hi, __x_lo);
}
template <>
_CCCL_API inline uint64_t __fpmp2_to_ull<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_to_ull(__x_hi, __x_lo);
}
template <>
_CCCL_API inline uint64_t __fpmp2_bit_cast<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_bit_cast(__x_hi, __x_lo);
}
#  if _CCCL_FPMP_FP128_ENABLE == 1
template <>
_CCCL_API inline void __fpmp2_from_quad<double>(const __fpmp_fp128 __x, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_from_quad(__x, __res_hi, __res_lo);
}
template <>
_CCCL_API inline __fpmp_fp128 __fpmp2_to_quad<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_to_quad(__x_hi, __x_lo);
}
#  endif // _CCCL_FPMP_FP128_ENABLE == 1

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_IMPL_CVT_H
