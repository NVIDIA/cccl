//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_IMPL_MULADD_H
#define _CUDA___FP_FPMP_IMPL_MULADD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_impl_muladd.h - fpmp2 addition, subtraction, accumulate, multiply, renormalize, fma/mad and negation
    ==================================================================================================
    Per-operation implementation core split out of <cuda/__fp/fpmp_impl.h>. It carries the
    addition, subtraction, accumulate, multiply, renormalize, fma/mad and negation
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
 * Re-normalization operations
 * --------------------------------------------------------------------
 */
// Renormalize a multi-precision (double-float) number
// to ensure that the hi and lo parts are non-overlapping
// This is useful for fast mode to ensure that the result is accurate
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void
__fpmp2_renormalize(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = __fpmp_fast_two_sum(__x_hi, __x_lo, __res_lo);
}

/*
 * --------------------------------------------------------------------
 * Addition operations
 * --------------------------------------------------------------------
 */
/*
 * Fast addition operation
 * This is a simple addition operation with no normalization.
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_low_add(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  _FpType __r_hi, __r_lo;

  // Add high parts using general 2-Sum (no magnitude assumption)
  __r_hi = __fpmp_two_sum(__x_hi, __y_hi, &__r_lo);
  // Add low parts
  __r_lo = __fpmp_add_rn(__fpmp_add_rn(__x_lo, __y_lo), __r_lo);

  *__res_hi = __r_hi;
  *__res_lo = __r_lo;
} // __fpmp2_low_add

/*
 * Dekker addition operation
 * This is classic split and error accumulation addition operation with normalization.
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_add(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  _FpType __r_lo_refine;
  _FpType __r_hi, __r_lo;

  // Add high parts using general 2-Sum (no magnitude assumption)
  __r_hi = __fpmp_two_sum(__x_hi, __y_hi, &__r_lo);
  // Add low parts
  __r_lo_refine = __fpmp_add_rn(__fpmp_add_rn(__x_lo, __y_lo), __r_lo);
  // Normalize:
  *__res_hi = __fpmp_fast_two_sum(__r_hi, __r_lo_refine, __res_lo);
} // __fpmp2_add

/*
 * FPAN-style accurate addition (compile-time selectable normalization strategy)
 * Optimized for instruction-level parallelism with branch-free structure.
 * Algorithm:
 *   (s_h, s_l) = TwoSum(a_hi, b_hi)   // Level 1 (parallel)
 *   (t_h, t_l) = TwoSum(a_lo, b_lo)   // Level 1 (parallel)
 *   c = s_l + t_h                      // Level 2: merge middle terms
 *   (v_h, v_l) = Fast2Sum(s_h, c)      // Level 3: normalize
 *   w = t_l + v_l                      // Level 4: absorb error
 *   (r_h, r_l) = Fast2Sum(v_h, w)      // Level 5: final normalize
 *
 * Total: 20 ops, Critical path: 14 ops (vs Thall's 17 ops sequential)
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __internal_nv_fpmp2_add_fpan(
  const _FpType __a_hi,
  const _FpType __a_lo,
  const _FpType __b_hi,
  const _FpType __b_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  // Level 1: Two independent 2Sums - can execute in parallel
  // Inline two_sum for a_hi + b_hi to help compiler see independence
  _FpType __s_h = __fpmp_add_rn(__a_hi, __b_hi);
  _FpType __s_a = __fpmp_sub_rn(__s_h, __b_hi);
  _FpType __s_b = __fpmp_sub_rn(__s_h, __s_a);

  // Inline two_sum for a_lo + b_lo (parallel with above)
  _FpType __t_h = __fpmp_add_rn(__a_lo, __b_lo);
  _FpType __t_a = __fpmp_sub_rn(__t_h, __b_lo);
  _FpType __t_b = __fpmp_sub_rn(__t_h, __t_a);

  // Complete the error calculations (can interleave)
  _FpType __s_da = __fpmp_sub_rn(__a_hi, __s_a);
  _FpType __s_db = __fpmp_sub_rn(__b_hi, __s_b);
  _FpType __s_l  = __fpmp_add_rn(__s_da, __s_db);

  _FpType __t_da = __fpmp_sub_rn(__a_lo, __t_a);
  _FpType __t_db = __fpmp_sub_rn(__b_lo, __t_b);
  _FpType __t_l  = __fpmp_add_rn(__t_da, __t_db);

  // Level 2: Merge middle terms
  _FpType __c = __fpmp_add_rn(__s_l, __t_h);

  // Level 3: First normalization (Fast2Sum since |s_h| >= |c| typically)
  _FpType __v_h   = __fpmp_add_rn(__s_h, __c);
  _FpType __v_tmp = __fpmp_sub_rn(__v_h, __s_h);
  _FpType __v_l   = __fpmp_sub_rn(__c, __v_tmp);

  // Level 4: Absorb remaining error
  _FpType __w = __fpmp_add_rn(__t_l, __v_l);

  // Level 5: Final normalization
  *__res_hi       = __fpmp_add_rn(__v_h, __w);
  _FpType __r_tmp = __fpmp_sub_rn(*__res_hi, __v_h);
  *__res_lo       = __fpmp_sub_rn(__w, __r_tmp);
} // __fpmp2_add_fpan

/*
 * Thall addition operation via expansion series
 * This implementation is based on: Andrew Thall, Extended-Precision
 * Floating-Point Numbers for GPU Computation. Retrieved on 7/12/2011
 * from http://andrewthall.org/papers/df64_qf128.pdf.
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __internal_nv_fpmp2_add_exp(
  const _FpType __a_hi,
  const _FpType __a_lo,
  const _FpType __b_hi,
  const _FpType __b_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  _FpType __t1, __t2, __t3, __t4, __t5, __e;
  __t1 = __fpmp_add_rn(__a_hi, __b_hi);
  __t2 = __fpmp_sub_rn(__t1, __a_hi);
  __t3 = __fpmp_add_rn(__fpmp_add_rn(__a_hi, __fpmp_sub_rn(__t2, __t1)), __fpmp_sub_rn(__b_hi, __t2));
  __t4 = __fpmp_add_rn(__a_lo, __b_lo);
  __t2 = __fpmp_sub_rn(__t4, __a_lo);
  __t5 = __fpmp_add_rn(__fpmp_add_rn(__a_lo, __fpmp_sub_rn(__t2, __t4)), __fpmp_sub_rn(__b_lo, __t2));
  __t3 = __fpmp_add_rn(__t3, __t4);
  __t4 = __fpmp_add_rn(__t1, __t3);
  __t3 = __fpmp_add_rn(__fpmp_sub_rn(__t1, __t4), __t3);
  __t3 = __fpmp_add_rn(__t3, __t5);
  __e  = __fpmp_add_rn(__t4, __t3);

  *__res_lo = __fpmp_add_rn(__fpmp_sub_rn(__t4, __e), __t3);
  *__res_hi = __e;
} // __fpmp2_high_add

#  define _CCCL_FPMP_FPAN_METHOD

template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_high_add(
  const _FpType __a_hi,
  const _FpType __a_lo,
  const _FpType __b_hi,
  const _FpType __b_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
#  if defined _CCCL_FPMP_FPAN_METHOD
  __internal_nv_fpmp2_add_fpan(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
#  else
  __internal_nv_fpmp2_add_exp(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
#  endif
}

/*
 * --------------------------------------------------------------------
 * Subtraction operations
 * --------------------------------------------------------------------
 */
/*
 * Fast subtraction operation
 * This is a simple subtraction operation with no normalization.
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_low_sub(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  __fpmp2_low_add(__x_hi, __x_lo, -__y_hi, -__y_lo, __res_hi, __res_lo);
}
/*
 * Classic split and error accumulation subtraction operation
 * This is a Dekker subtraction operation with normalization.
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_sub(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  __fpmp2_add(__x_hi, __x_lo, -__y_hi, -__y_lo, __res_hi, __res_lo);
}
/*
 * Thall accurate subtraction operation
 * This implementation is based on: Andrew Thall, Extended-Precision
 * Floating-Point Numbers for GPU Computation. Retrieved on 7/12/2011
 * from http://andrewthall.org/papers/df64_qf128.pdf.
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_high_sub(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  __fpmp2_high_add(__x_hi, __x_lo, -__y_hi, -__y_lo, __res_hi, __res_lo);
}

/*
 * --------------------------------------------------------------------
 * Accumulate operations (single-component addition to multi-precision)
 * --------------------------------------------------------------------
 * These functions efficiently accumulate a single-precision value into
 * a multi-precision (hi, lo) pair. More efficient than full mp2+mp2
 * addition since only one 2Sum is needed (the contribution has lo=0).
 *
 * Algorithm (Dekker-style):
 *   (new_hi, err) = 2Sum(acc_hi, c)      // Add c to high part
 *   new_lo = acc_lo + err                 // Accumulate error into low part
 *   (res_hi, res_lo) = Fast2Sum(new_hi, new_lo)  // Normalize
 *
 * This saves ~6 operations vs full addition (no 2Sum for low parts).
 */

/*
 * Fast accumulate: no final normalization
 * Result may have overlapping hi/lo components until renormalized.
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_low_acc(const _FpType __c, _FpType* __acc_hi, _FpType* __acc_lo) noexcept
{
  _FpType __err;
  // Add c to high part with error capture
  _FpType __new_hi = __fpmp_two_sum(*__acc_hi, __c, &__err);
  // Accumulate error into low part (no normalization)
  *__acc_hi = __new_hi;
  *__acc_lo = __fpmp_add_rn(*__acc_lo, __err);
}

/*
 * Default accumulate: Dekker-style with normalization
 * Result is properly normalized (non-overlapping hi/lo).
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_acc(const _FpType __c, _FpType* __acc_hi, _FpType* __acc_lo) noexcept
{
  _FpType __err;
  // Add c to high part with error capture
  _FpType __new_hi = __fpmp_two_sum(*__acc_hi, __c, &__err);
  // Combine error with existing low part
  _FpType __new_lo = __fpmp_add_rn(*__acc_lo, __err);
  // Normalize result
  *__acc_hi = __fpmp_fast_two_sum(__new_hi, __new_lo, __acc_lo);
}

/*
 * Accurate accumulate: Full error propagation (FPAN-style)
 * Provides maximum precision by properly ordering all error terms.
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_high_acc(const _FpType __c, _FpType* __acc_hi, _FpType* __acc_lo) noexcept
{
  _FpType __err;
  // Add c to high part with error capture
  _FpType __s_hi = __fpmp_two_sum(*__acc_hi, __c, &__err);
  // Add error to low part
  _FpType __t = __fpmp_add_rn(*__acc_lo, __err);
  // First normalization
  _FpType __v_hi  = __fpmp_add_rn(__s_hi, __t);
  _FpType __v_tmp = __fpmp_sub_rn(__v_hi, __s_hi);
  _FpType __v_lo  = __fpmp_sub_rn(__t, __v_tmp);
  // Final normalization
  *__acc_hi       = __fpmp_add_rn(__v_hi, __v_lo);
  _FpType __r_tmp = __fpmp_sub_rn(*__acc_hi, __v_hi);
  *__acc_lo       = __fpmp_sub_rn(__v_lo, __r_tmp);
}

/*
 * --------------------------------------------------------------------
 * Multiplication operations
 * --------------------------------------------------------------------
 */
/*
 * Fast multiplication operation
 * This is a simple multiplication operation with no normalization.
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_low_mul(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  _FpType __t_hi = __fpmp_mul_rn(__x_hi, __y_hi);
  _FpType __t_lo = __fpmp_fma_rn(__x_hi, __y_hi, -__t_hi);
  __t_lo         = __fpmp_fma_rn(__x_lo, __y_lo, __t_lo);
  __t_lo         = __fpmp_fma_rn(__x_hi, __y_lo, __t_lo);
  __t_lo         = __fpmp_fma_rn(__x_lo, __y_hi, __t_lo);

  *__res_hi = __t_hi;
  *__res_lo = __t_lo;
} // __fpmp2_low_mul

/*
 * Dekker multiplication operation
 * This is a Dekker multiplication operation with normalization.
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_mul(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  _FpType __p1, __p2, __c_hi, __c_lo, __res_hi_tmp, __res_lo_tmp;
  __c_hi = __fpmp_two_mult_fma(__x_hi, __y_hi, &__c_lo);
  __p1   = __fpmp_mul_rn(__x_hi, __y_lo);
  __p2   = __fpmp_mul_rn(__x_lo, __y_hi);
  __c_lo = __fpmp_add_rn(__c_lo, __fpmp_add_rn(__p1, __p2));
  // Normalize:
  __res_hi_tmp = __fpmp_fast_two_sum(__c_hi, __c_lo, &__res_lo_tmp);

  *__res_hi = __res_hi_tmp;
  *__res_lo = __res_lo_tmp;
} // __fpmp2_mul

#  if _CCCL_FPMP_USE_ACCURATE_MUL == 1
/*
 * Dekker multiplication with branch-free conditional scaling
 * ===========================================================
 *
 * This implementation uses the standard Dekker multiplication algorithm
 * with branch-free conditional scaling to handle subnormal results accurately.
 *
 * The standard Dekker algorithm fails when the product approaches the
 * subnormal range because the error term from two_mult_fma loses precision
 * due to gradual underflow. This implementation detects such cases using
 * bit manipulation and applies scaling to ensure all intermediate computations
 * happen in the normal range where error-free transformations are exact.
 *
 * ALGORITHM:
 *   1. Compute conditional scale factor based on operand exponents (branch-free).
 *   2. Scale first operand if product would be small.
 *   3. Perform standard Dekker multiplication.
 *   4. Scale result back with inverse factor.
 *
 * CONDITIONAL SCALE COMPUTATION (branch-free):
 *   - Extract sum of exponents from x_hi and y_hi.
 *   - If sum < threshold: scale = 2^64 (float) or 2^512 (double), else scale = 1.0
 *   - Use bit manipulation to select between scale values without branches.
 *
 * PERFORMANCE:
 *   - Normal case: scale = 1.0, minimal overhead (MUL by 1.0 is fast).
 *   - Subnormal case: full scaling applied for accuracy.
 *   - Branch-free: no GPU warp divergence.
 *   - Overhead vs __fpmp2_mul: ~6 integer ops + 4 MUL (often identity) + 1 fast_two_sum.
 *
 * REFERENCE:
 *   Dekker, T. (1971). A floating-point technique for extending available precision.
 *   Conditional scaling adapted from QD library techniques.
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_high_mul(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  // Type-specific constants for conditional scaling
  using UintType = ::cuda::std::conditional_t<__fpmp2_is_fp32_v<_FpType>, uint32_t, uint64_t>;

  constexpr int __exp_bits      = __fpmp2_is_fp32_v<_FpType> ? 8 : 11;
  constexpr int __mant_bits     = __fpmp2_is_fp32_v<_FpType> ? 23 : 52;
  constexpr int __exp_bias      = __fpmp2_is_fp32_v<_FpType> ? 127 : 1023;
  constexpr UintType __exp_mask = ((UintType(1) << __exp_bits) - 1) << __mant_bits;

  // Threshold: if combined exponent < this, we need scaling
  // For float: scale_shift=64, threshold=190 (2*127-64)
  // For double: scale_shift=512, threshold=1534 (2*1023-512)
  constexpr int __scale_shift   = __fpmp2_is_fp32_v<_FpType> ? 64 : 512;
  constexpr int __exp_threshold = 2 * __exp_bias - __scale_shift;

  // Scale factors
  constexpr _FpType __scale_up   = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p64f) : _FpType(0x1.0p512);
  constexpr _FpType __scale_down = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p-64f) : _FpType(0x1.0p-512);

  // Extract exponents and compute conditional scale (branch-free)
  UintType __x_bits = ::cuda::std::bit_cast<UintType>(__x_hi);
  UintType __y_bits = ::cuda::std::bit_cast<UintType>(__y_hi);
  int __x_exp       = static_cast<int>((__x_bits & __exp_mask) >> __mant_bits);
  int __y_exp       = static_cast<int>((__y_bits & __exp_mask) >> __mant_bits);
  int __result_exp  = __x_exp + __y_exp;

  // Create mask: -1 (all 1s) if needs scaling, 0 otherwise
  int __needs_scale = (__result_exp - __exp_threshold) >> 31;

  // Select scale factor using bit manipulation (branch-free)
  UintType __scale_up_bits = ::cuda::std::bit_cast<UintType>(__scale_up);
  UintType __one_bits      = ::cuda::std::bit_cast<UintType>(_FpType(1.0));
  UintType __scale_bits    = (__scale_up_bits & UintType(__needs_scale)) | (__one_bits & UintType(~__needs_scale));
  _FpType __scale          = ::cuda::std::bit_cast<_FpType>(__scale_bits);

  UintType __scale_down_bits = ::cuda::std::bit_cast<UintType>(__scale_down);
  UintType __inv_scale_bits  = (__scale_down_bits & UintType(__needs_scale)) | (__one_bits & UintType(~__needs_scale));
  _FpType __inv_scale        = ::cuda::std::bit_cast<_FpType>(__inv_scale_bits);

  // Scale first operand
  _FpType __a_hi = __fpmp_mul_rn(__x_hi, __scale);
  _FpType __a_lo = __fpmp_mul_rn(__x_lo, __scale);

  // Standard Dekker multiplication
  _FpType __c_lo;
  _FpType __c_hi = __fpmp_two_mult_fma(__a_hi, __y_hi, &__c_lo);
  _FpType __p1   = __fpmp_mul_rn(__a_hi, __y_lo);
  _FpType __p2   = __fpmp_mul_rn(__a_lo, __y_hi);
  __c_lo         = __fpmp_add_rn(__c_lo, __fpmp_add_rn(__p1, __p2));

  // Normalize
  _FpType __r_lo;
  _FpType __r_hi = __fpmp_fast_two_sum(__c_hi, __c_lo, &__r_lo);

  // Scale back
  __r_hi = __fpmp_mul_rn(__r_hi, __inv_scale);
  __r_lo = __fpmp_mul_rn(__r_lo, __inv_scale);

  // Final normalization to ensure (hi, lo) invariant after scaling
  *__res_hi = __fpmp_fast_two_sum(__r_hi, __r_lo, __res_lo);
} // __fpmp2_high_mul
#  endif // _CCCL_FPMP_USE_ACCURATE_MUL == 1

/* Compute fast fused multiply-add: x*y+z  (16 ops, no normalization)
    Uses hardware FMA for the main term (single rounding), then recovers
    the exact error via the Boldo-Muller EFT:
      x_hi*y_hi = p + q  (exact, via two_mult_fma)
      p + z_hi  = s + t  (exact, via two_sum)
      => error  = (s - r_hi) + t + q
    where (s - r_hi) is exact by the Boldo-Muller theorem.
*/
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_low_fma(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  const _FpType __z_hi,
  const _FpType __z_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  _FpType __r_hi = __fpmp_fma_rn(__x_hi, __y_hi, __z_hi);

  _FpType __q;
  _FpType __p = __fpmp_two_mult_fma(__x_hi, __y_hi, &__q);
  _FpType __t;
  _FpType __s    = __fpmp_two_sum(__p, __z_hi, &__t);
  _FpType __r_lo = __fpmp_add_rn(__fpmp_sub_rn(__s, __r_hi), __fpmp_add_rn(__t, __q));

  __r_lo = __fpmp_fma_rn(__x_hi, __y_lo, __r_lo);
  __r_lo = __fpmp_fma_rn(__x_lo, __y_hi, __r_lo);
  __r_lo = __fpmp_fma_rn(__x_lo, __y_lo, __r_lo);
  __r_lo = __fpmp_add_rn(__r_lo, __z_lo);

  *__res_hi = __r_hi;
  *__res_lo = __r_lo;
} // __fpmp2_low_fma

/* Compute high-accuracy fused multiply-add: x*y+z
    Uses hardware FMA for the main term (single rounding), then recovers
    the exact error via the Boldo-Muller EFT:
      x_hi*y_hi = p + q  (exact, via two_mult_fma)
      p + z_hi  = s + t  (exact, via two_sum)
      => error  = (s - r_hi) + t + q
    where (s - r_hi) is exact by the Boldo-Muller theorem.
*/
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_fma(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  const _FpType __z_hi,
  const _FpType __z_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  // Hardware FMA: x_hi*y_hi + z_hi with single rounding (optimal)
  _FpType __r_hi = __fpmp_fma_rn(__x_hi, __y_hi, __z_hi);

  // Exact error recovery for the main FMA
  _FpType __q;
  _FpType __p = __fpmp_two_mult_fma(__x_hi, __y_hi, &__q);
  _FpType __t;
  _FpType __s    = __fpmp_two_sum(__p, __z_hi, &__t);
  _FpType __r_lo = __fpmp_add_rn(__fpmp_sub_rn(__s, __r_hi), __fpmp_add_rn(__t, __q));

  // Cross terms and remaining contributions
  __r_lo = __fpmp_fma_rn(__x_hi, __y_lo, __r_lo);
  __r_lo = __fpmp_fma_rn(__x_lo, __y_hi, __r_lo);
  __r_lo = __fpmp_fma_rn(__x_lo, __y_lo, __r_lo);
  __r_lo = __fpmp_add_rn(__r_lo, __z_lo);

  // Normalize
  *__res_hi = __fpmp_fast_two_sum(__r_hi, __r_lo, __res_lo);
} // __fpmp2_fma

/* Compute accurate fused multiply-add: x*y+z
    Same EFT-based main term as __fpmp2_fma, but cross terms are
    computed exactly via two_mult_fma and accumulated with two_sum
    error tracking. This avoids precision loss when cross terms are
    of similar magnitude to r_lo (e.g. catastrophic cancellation in
    the main term).
*/
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_high_fma(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  const _FpType __z_hi,
  const _FpType __z_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  _FpType __r_hi = __fpmp_fma_rn(__x_hi, __y_hi, __z_hi);

  _FpType __q;
  _FpType __p = __fpmp_two_mult_fma(__x_hi, __y_hi, &__q);
  _FpType __t;
  _FpType __s    = __fpmp_two_sum(__p, __z_hi, &__t);
  _FpType __r_lo = __fpmp_add_rn(__fpmp_sub_rn(__s, __r_hi), __fpmp_add_rn(__t, __q));

  _FpType __c1_lo;
  _FpType __c1_hi = __fpmp_two_mult_fma(__x_hi, __y_lo, &__c1_lo);

  _FpType __c2_lo;
  _FpType __c2_hi = __fpmp_two_mult_fma(__x_lo, __y_hi, &__c2_lo);

  _FpType __cross_err;
  _FpType __cross = __fpmp_two_sum(__c1_hi, __c2_hi, &__cross_err);

  _FpType __acc_err;
  __r_lo = __fpmp_two_sum(__r_lo, __cross, &__acc_err);

  _FpType __residual = __fpmp_add_rn(__acc_err, __fpmp_add_rn(__cross_err, __fpmp_add_rn(__c1_lo, __c2_lo)));
  __residual         = __fpmp_fma_rn(__x_lo, __y_lo, __residual);
  __residual         = __fpmp_add_rn(__residual, __z_lo);

  __r_lo = __fpmp_add_rn(__r_lo, __residual);

  *__res_hi = __fpmp_fast_two_sum(__r_hi, __r_lo, __res_lo);
} // __fpmp2_high_fma

/*
 * --------------------------------------------------------------------
 * Fused multiply-add with rounding operations
 * --------------------------------------------------------------------
 */
// multiply-add with rounding (default: fast mul + default add)
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_mad(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  const _FpType __z_hi,
  const _FpType __z_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  _FpType __t_hi, __t_lo;
  __fpmp2_low_mul(__x_hi, __x_lo, __y_hi, __y_lo, &__t_hi, &__t_lo);
  __fpmp2_add(__t_hi, __t_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}

// multiply-add fast (fast mul + fast add)
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_low_mad(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  const _FpType __z_hi,
  const _FpType __z_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  _FpType __t_hi, __t_lo;
  __fpmp2_low_mul(__x_hi, __x_lo, __y_hi, __y_lo, &__t_hi, &__t_lo);
  __fpmp2_low_add(__t_hi, __t_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}

// multiply-add accurate (default mul + accurate add)
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_high_mad(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  const _FpType __z_hi,
  const _FpType __z_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  _FpType __t_hi, __t_lo;
  __fpmp2_mul(__x_hi, __x_lo, __y_hi, __y_lo, &__t_hi, &__t_lo);
  __fpmp2_high_add(__t_hi, __t_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}

/*
 * --------------------------------------------------------------------
 * Negation operations
 * --------------------------------------------------------------------
 */
// negation
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void
__fpmp2_neg(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = -__x_hi;
  *__res_lo = -__x_lo;
}

#else // _CCCL_FPMP_USE_LIB

// -- fp32 (single precision) built-in declarations --
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_add(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_mid_add(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_low_add(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_high_add(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_sub(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_mid_sub(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_low_sub(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_high_sub(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_acc(const float __c, float* __acc_hi, float* __acc_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_mid_acc(const float __c, float* __acc_hi, float* __acc_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_low_acc(const float __c, float* __acc_hi, float* __acc_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_high_acc(const float __c, float* __acc_hi, float* __acc_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_mul(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_mid_mul(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_low_mul(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
#  if _CCCL_FPMP_USE_ACCURATE_MUL == 1
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_high_mul(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
#  endif // _CCCL_FPMP_USE_ACCURATE_MUL == 1
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_renormalize(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_mad(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_mid_mad(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_low_mad(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_high_mad(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_fma(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_mid_fma(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_low_fma(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_high_fma(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_neg(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;

// -- fp64 (double precision) built-in declarations --
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_add(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_mid_add(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_low_add(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_high_add(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_sub(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_mid_sub(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_low_sub(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_high_sub(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_acc(const double __c, double* __acc_hi, double* __acc_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_mid_acc(const double __c, double* __acc_hi, double* __acc_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_low_acc(const double __c, double* __acc_hi, double* __acc_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_high_acc(const double __c, double* __acc_hi, double* __acc_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_mul(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_mid_mul(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_low_mul(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
#  if _CCCL_FPMP_USE_ACCURATE_MUL == 1
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_high_mul(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
#  endif // _CCCL_FPMP_USE_ACCURATE_MUL == 1
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_renormalize(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_mad(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_mid_mad(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_low_mad(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_high_mad(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_fma(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_mid_fma(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_low_fma(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_high_fma(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_neg(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;

// -- type-generic template declarations (dispatch to fp32/fp64) --
template <typename _Tp>
_CCCL_API inline void __fpmp2_add(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_mid_add(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_low_add(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_high_add(
  const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_sub(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_mid_sub(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_low_sub(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_high_sub(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_acc(const _Tp __c, _Tp* __acc_hi, _Tp* __acc_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_mid_acc(const _Tp __c, _Tp* __acc_hi, _Tp* __acc_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_low_acc(const _Tp __c, _Tp* __acc_hi, _Tp* __acc_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_high_acc(const _Tp __c, _Tp* __acc_hi, _Tp* __acc_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_mul(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_mid_mul(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_low_mul(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
#  if _CCCL_FPMP_USE_ACCURATE_MUL == 1
template <typename T>
_CCCL_API inline void
__fpmp2_high_mul(const T __x_hi, const T __x_lo, const T __y_hi, const T __y_lo, T* __res_hi, T* __res_lo) noexcept;
#  endif // _CCCL_FPMP_USE_ACCURATE_MUL == 1
template <typename _Tp>
_CCCL_API inline void __fpmp2_renormalize(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_mad(
  const _Tp __x_hi,
  const _Tp __x_lo,
  const _Tp __y_hi,
  const _Tp __y_lo,
  const _Tp __z_hi,
  const _Tp __z_lo,
  _Tp* __res_hi,
  _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_mid_mad(
  const _Tp __x_hi,
  const _Tp __x_lo,
  const _Tp __y_hi,
  const _Tp __y_lo,
  const _Tp __z_hi,
  const _Tp __z_lo,
  _Tp* __res_hi,
  _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_low_mad(
  const _Tp __x_hi,
  const _Tp __x_lo,
  const _Tp __y_hi,
  const _Tp __y_lo,
  const _Tp __z_hi,
  const _Tp __z_lo,
  _Tp* __res_hi,
  _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_high_mad(
  const _Tp __x_hi,
  const _Tp __x_lo,
  const _Tp __y_hi,
  const _Tp __y_lo,
  const _Tp __z_hi,
  const _Tp __z_lo,
  _Tp* __res_hi,
  _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_fma(
  const _Tp __x_hi,
  const _Tp __x_lo,
  const _Tp __y_hi,
  const _Tp __y_lo,
  const _Tp __z_hi,
  const _Tp __z_lo,
  _Tp* __res_hi,
  _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_mid_fma(
  const _Tp __x_hi,
  const _Tp __x_lo,
  const _Tp __y_hi,
  const _Tp __y_lo,
  const _Tp __z_hi,
  const _Tp __z_lo,
  _Tp* __res_hi,
  _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_low_fma(
  const _Tp __x_hi,
  const _Tp __x_lo,
  const _Tp __y_hi,
  const _Tp __y_lo,
  const _Tp __z_hi,
  const _Tp __z_lo,
  _Tp* __res_hi,
  _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_high_fma(
  const _Tp __x_hi,
  const _Tp __x_lo,
  const _Tp __y_hi,
  const _Tp __y_lo,
  const _Tp __z_hi,
  const _Tp __z_lo,
  _Tp* __res_hi,
  _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_fma_exp(
  const _Tp __x_hi,
  const _Tp __x_lo,
  const _Tp __y_hi,
  const _Tp __y_lo,
  const _Tp __z_hi,
  const _Tp __z_lo,
  _Tp* __res_hi,
  _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_neg(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;

// -- fp32 template specializations --
template <>
_CCCL_API inline void __fpmp2_add<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_add(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_add<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_mid_add(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_add<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_low_add(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_high_add<float>(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_high_add(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_sub<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_sub<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_mid_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_sub<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_low_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_high_sub<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_high_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_acc<float>(const float __c, float* __acc_hi, float* __acc_lo) noexcept
{
  __fp32mp2_acc(__c, __acc_hi, __acc_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_acc<float>(const float __c, float* __acc_hi, float* __acc_lo) noexcept
{
  __fp32mp2_mid_acc(__c, __acc_hi, __acc_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_acc<float>(const float __c, float* __acc_hi, float* __acc_lo) noexcept
{
  __fp32mp2_low_acc(__c, __acc_hi, __acc_lo);
}
template <>
_CCCL_API inline void __fpmp2_high_acc<float>(const float __c, float* __acc_hi, float* __acc_lo) noexcept
{
  __fp32mp2_high_acc(__c, __acc_hi, __acc_lo);
}
template <>
_CCCL_API inline void __fpmp2_mul<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_mul<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_mid_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_mul<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_low_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
#  if _CCCL_FPMP_USE_ACCURATE_MUL == 1
template <>
_CCCL_API inline void __fpmp2_high_mul<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_high_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
#  endif // _CCCL_FPMP_USE_ACCURATE_MUL == 1
template <>
_CCCL_API inline void
__fpmp2_renormalize<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_renormalize(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mad<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_mad<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_mid_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_mad<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_low_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_high_mad<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_high_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_fma<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_fma<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_mid_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_fma<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_low_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_high_fma<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  const float __z_hi,
  const float __z_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_high_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_neg<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_neg(__x_hi, __x_lo, __res_hi, __res_lo);
}

// -- fp64 template specializations --
template <>
_CCCL_API inline void __fpmp2_add<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_add(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_add<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_mid_add(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_add<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_low_add(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_high_add<double>(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_high_add(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_sub<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_sub<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_mid_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_sub<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_low_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_high_sub<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_high_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_acc<double>(const double __c, double* __acc_hi, double* __acc_lo) noexcept
{
  __fp64mp2_acc(__c, __acc_hi, __acc_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_acc<double>(const double __c, double* __acc_hi, double* __acc_lo) noexcept
{
  __fp64mp2_mid_acc(__c, __acc_hi, __acc_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_acc<double>(const double __c, double* __acc_hi, double* __acc_lo) noexcept
{
  __fp64mp2_low_acc(__c, __acc_hi, __acc_lo);
}
template <>
_CCCL_API inline void __fpmp2_high_acc<double>(const double __c, double* __acc_hi, double* __acc_lo) noexcept
{
  __fp64mp2_high_acc(__c, __acc_hi, __acc_lo);
}
template <>
_CCCL_API inline void __fpmp2_mul<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_mul<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_mid_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_mul<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_low_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
#  if _CCCL_FPMP_USE_ACCURATE_MUL == 1
template <>
_CCCL_API inline void __fpmp2_high_mul<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_high_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
#  endif // _CCCL_FPMP_USE_ACCURATE_MUL == 1
template <>
_CCCL_API inline void
__fpmp2_renormalize<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_renormalize(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mad<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_mad<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_mid_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_mad<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_low_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_high_mad<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_high_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_fma<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_mid_fma<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_mid_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_low_fma<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_low_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_high_fma<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  const double __z_hi,
  const double __z_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_high_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_neg<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_neg(__x_hi, __x_lo, __res_hi, __res_lo);
}

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_IMPL_MULADD_H
