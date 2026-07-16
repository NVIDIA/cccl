//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_MUL_H
#define _CUDA___FP_FPEMU_IMPL_MUL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! @file fpemu_dmul_impl.hpp
//! @brief Implementation of double-precision multiplication operations for FPEMU floating point emulation library
//!
//! This header provides the implementation of double-precision multiplication operations for the FPEMU library.
//! It includes:
//!
//! - Multiplication functions for fpemu
//! - Multiplication operators for fpemu
//! - Multiplication functions to other types
//!
//! The multiplication functions are designed to work across both host and device code
//! through appropriate decorators and provide bit-exact results matching hardware
//! floating point units.
#define _CCCL_FP64EMU_USE_MUL_UNPACKED      0
#define _CCCL_FP64EMU_DMUL_FP32_FAST_ENABLE 1

#include <cuda/__fp/fpemu_impl.h>
#include <cuda/__fp/fpemu_impl_unpack.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief v1.0 Emulation of double-precision multiplication for FPEMU
//!
//! This function implements the reference (v1.0) emulation of double-precision
//! floating-point multiplication for the FPEMU format. It operates on the
//! internal bitwise representation (__fpbits64) of the operands and produces
//! a bit-exact result matching the IEEE-754 standard for double-precision
//! multiplication.
//!
//! The algorithm performs the following steps:
//!   - Extracts the sign, exponent, and mantissa from both operands.
//!   - Handles special cases such as zero, denormalized numbers, infinities, and NaNs.
//!   - Multiplies the mantissas with full precision, including the implicit leading bit.
//!   - Computes the resulting exponent, taking into account normalization and bias.
//!   - Normalizes the result and applies rounding according to the specified mode.
//!   - Packs the sign, exponent, and mantissa back into the __fpbits64 result.
//!
//! This implementation is designed for correctness and bitwise reproducibility,
//! serving as a reference for optimized or hardware-accelerated versions.
//!
//! @tparam rm   Rounding mode (default: nearest-even)
//! @tparam _Acc Accuracy level (fpemu_accuracy; default: high)
//! @param x     First operand (__fpbits64)
//! @param y     Second operand (__fpbits64)
//! @return      Product as __fpbits64
template <__fpemu_rounding _Rm = __fpemu_rounding::def, fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_high_dmul(__fpbits64 __x, __fpbits64 __y) noexcept
{
  uint64_t __a = __x;
  uint64_t __b = __y;

  __uint32x2 __a_32x2 = __fpemu_bit_cast<__uint32x2>(__a);
  __uint32x2 __b_32x2 = __fpemu_bit_cast<__uint32x2>(__b);

  __uint32x2 __man_a_32x2, __man_b_32x2, __man_c_32x2;
  int32_t __exp_a, __exp_b, __exp_c, __shift;
  bool __is_sign_a, __is_sign_b, __is_sign_c, __is_a_exp_zero, __is_b_exp_zero, __is_impl_bit;
  __fpbits64 __result;

  // Extract sign and exponent for input A
  __exp_a = __unpack_exp<_Acc>(__a_32x2);
  // Extract sign and exponent for input B
  __exp_b = __unpack_exp<_Acc>(__b_32x2);

  // Check if input A is denormal or zero
  __is_a_exp_zero = (__exp_a == 0);
  __is_b_exp_zero = (__exp_b == 0);

  // Extract mantissa for input A
  __man_a_32x2 = __unpack_mant<_Acc>(&__is_sign_a, __a_32x2, __is_a_exp_zero);
  // Extract mantissa for input B
  __man_b_32x2 = __unpack_mant<_Acc>(&__is_sign_b, __b_32x2, __is_b_exp_zero);

  if constexpr (_Acc != fpemu_accuracy::high)
  {
    if (__is_a_exp_zero)
    {
      __man_a_32x2 = {0, 0};
    }
    if (__is_b_exp_zero)
    {
      __man_b_32x2 = {0, 0};
    }
  }

  // Correct exp and mantissa for denormal
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    __exp_a      = (__is_a_exp_zero) ? 12 - __flo_u64(__a_32x2) : __exp_a;
    __exp_b      = (__is_b_exp_zero) ? 12 - __flo_u64(__b_32x2) : __exp_b;
    __man_a_32x2 = (__is_a_exp_zero) ? __shl_64(__man_a_32x2, 1 - __exp_a) : __man_a_32x2;
    __man_b_32x2 = (__is_b_exp_zero) ? __shl_64(__man_b_32x2, 1 - __exp_b) : __man_b_32x2;
  }
  else
  {
    __exp_a = (!__is_a_exp_zero) ? __exp_a : -52;
    __exp_b = (!__is_b_exp_zero) ? __exp_b : -52;
  }

  if constexpr (_Acc == fpemu_accuracy::high)
  {
    __uint32x4 __man_c_32x4 = __mul_128<_Acc>(__man_a_32x2, __man_b_32x2);
    // Rounding: Set least significant bit to "1"
    // if low 64 bits are non-zero
    if ((__man_c_32x4.lo.x[0] | __man_c_32x4.lo.x[1]) != 0)
    {
      __man_c_32x4.hi.x[0] |= 1;
    }
    __man_c_32x2 = __man_c_32x4.hi;
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    __man_c_32x2 = __mul_64<_Acc>(__man_a_32x2, __man_b_32x2);
  }
  else
  {
    __man_c_32x2.x[1] = __mul_32<_Acc>(__man_a_32x2, __man_b_32x2);
    __man_c_32x2.x[0] = 0;
  }

  // Check implicit-bit position
  __is_impl_bit = (__man_c_32x2.x[1] >= 0x08000000);

  // Calculate exponent
  __exp_c = __exp_a + __exp_b - (__fpemu_bias + 1) + __is_impl_bit;

  // Calculate SIGN
  __is_sign_c = __is_sign_a ^ __is_sign_b;

  // Check for negative exponent
  bool __is_exp_c_neg = (__exp_c < 0);

  if constexpr (_Acc == fpemu_accuracy::high)
  {
    if (__is_exp_c_neg)
    {
      // shift is negative, so we need to add it to exp_c
      __shift = -__exp_c;
      __exp_c = __exp_c + __shift;
      // Shift mantissa to the right with rounding
      // in case of negative exponent (DENORM)
      __man_c_32x2 = __sar_64_rnd<_Acc, _Rm>(__man_c_32x2, __shift, __is_sign_c);
    }
  }
  else
  {
    // Flush denormals to zero
    __exp_c = (__is_exp_c_neg) ? 0 : __exp_c;
    if (__is_exp_c_neg)
    {
      __man_c_32x2 = {0, 0};
    }
  }

  if (__is_impl_bit)
  {
    __man_c_32x2 = __round<_Rm>(__man_c_32x2, -2, __is_sign_c);
  }
  else
  {
    __man_c_32x2 = __round<_Rm>(__man_c_32x2, -3, __is_sign_c);
  }

  // Pack sign, exponent and mantissa back to FP64
  // (+checks for NAN,INF,0)
  __result = __pack<_Acc, _Rm>(__is_sign_c, __exp_c, __man_c_32x2);
  return __result;
} // __internal_fp64emu_high_dmul

//! @brief Version 2.0 emulation of the double-precision multiplication function.
//!
//! This implementation provides an emulation of IEEE-754 double-precision (fp64) multiplication,
//! supporting configurable rounding modes, accuracy, and range options.
//!
//! The function operates by:
//!   - Extracting the exponent and mantissa fields from the input operands.
//!   - Computing the result sign as the XOR of the input signs.
//!   - Adding the exponents and subtracting the fp64 bias to obtain the result exponent.
//!   - Multiplying the mantissas by a dedicated helper (__mul_mant), which handles
//!     normalization, carry, and accuracy-specific bit manipulations.
//!   - Adjusting the result for exponent overflow and underflow, and setting the sign and exponent bits.
//!   - Returning the correctly packed fp64 result.
//!
//! This version is designed for improved accuracy and performance, and is suitable for both
//! host and device execution.
template <__fpemu_rounding _Rm = __fpemu_rounding::def, fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_mid_dmul(__fpbits64 __x, __fpbits64 __y) noexcept
{
  __uint32x2 __a_32x2 = __fpemu_bit_cast<__uint32x2>(__x);
  __uint32x2 __b_32x2 = __fpemu_bit_cast<__uint32x2>(__y);
  __fpbits64 __result;

  // Extract exponents
  uint32_t __exp_a    = (__a_32x2.x[1] >> _CCCL_FP64_HI_MANT_SHIFT) & _CCCL_FP64_LO_EXP_MASK;
  uint32_t __exp_b    = (__b_32x2.x[1] >> _CCCL_FP64_HI_MANT_SHIFT) & _CCCL_FP64_LO_EXP_MASK;
  bool __is_exps_zero = ((__exp_a == 0) || (__exp_b == 0));

  // Compute result sign (XOR of input signs)
  uint32_t __result_sign = (__a_32x2.x[1] ^ __b_32x2.x[1]) & _CCCL_FP64_HI_SIGN_MASK;

  // Add exponents and subtract fp64 bias (1023)
  int32_t __result_exp = (int32_t) __exp_a + (int32_t) __exp_b - _CCCL_FP64_BIAS;

  // Multiply the mantissas
  __uint32x2 __result_32x2;

  // Integer based mantissa processing
  {
    uint32_t __carry_bit;

    // Clear the exponent/sign bits
    __a_32x2.x[1] &= _CCCL_FP64_HI_MANT_MASK;
    __b_32x2.x[1] &= _CCCL_FP64_HI_MANT_MASK;

    // Set the implicit 1 bit
    __a_32x2.x[1] |= (1 << _CCCL_FP64_HI_MANT_SHIFT);
    __b_32x2.x[1] |= (1 << _CCCL_FP64_HI_MANT_SHIFT);

    if constexpr (_Acc == fpemu_accuracy::mid)
    {
      // Shift the mantissa to the left by _CCCL_FP64_EXTRA_BITS (9) bits to preserve low bits
      __a_32x2 = __shl_64(__a_32x2, _CCCL_FP64_EXTRA_BITS);
      __b_32x2 = __shl_64(__b_32x2, _CCCL_FP64_EXTRA_BITS);

      // Multiply the mantissas
      __result_32x2 = __mul_64<_Acc>(__a_32x2, __b_32x2);

      // Check if the carry bit is set
      __carry_bit = (__result_32x2.x[1] >= (1 << (_CCCL_FP64_MANT_MUL_CARRY_BIT + _CCCL_FP64_EXTRA_BITS * 2 + 1)));

      // Shift the mantissa back with directed rounding for ru/rd
      const int __mant_shift = (__carry_bit) ? (((_CCCL_FP64_EXTRA_BITS * 2) - _CCCL_FP64_MANT_MUL_SHIFT) + 1)
                                             : ((_CCCL_FP64_EXTRA_BITS * 2) - _CCCL_FP64_MANT_MUL_SHIFT);
      __result_32x2          = __shr_64_rnd<_Rm>(__result_32x2, __mant_shift, __result_sign != 0);
    }
    else
    {
      // Multiply the mantissas
      __result_32x2 = __mul_64<_Acc>(__a_32x2, __b_32x2);

      // Check if the carry bit is set
      __carry_bit = (__result_32x2.x[1] >= (1 << (_CCCL_FP64_MANT_MUL_CARRY_BIT + 1)));

      // Shift the mantissa to the left by the correct amount taking into account the carry bit
      __result_32x2 =
        __shl_64(__result_32x2, (__carry_bit) ? (_CCCL_FP64_MANT_MUL_SHIFT - 1) : _CCCL_FP64_MANT_MUL_SHIFT);
    }

    // Add the carry bit to the exponent
    __result_exp = __result_exp + __carry_bit;

    // Clear the unused high bits
    __result_32x2.x[1] &= _CCCL_FP64_HI_MANT_MASK;
  }

  // Check for exponent overflow
  bool __is_exp_ovfl = (__result_exp > (_CCCL_FP64_BIAS * 2));

  /*
  // Correct exponent:
  - Check for exponent zero
  - Adjust exponent if overflow occurs
  - Check for negative exponent
  - Adjust exponent if negative
  - Check for exponent overflow or zero
  - Adjust exponent if overflow or zero
  - Check for exponent zero
  - Adjust exponent if zero
  */
  const bool __is_sign_c = (__result_sign != 0);

  if (__is_exps_zero)
  {
    __result_exp  = 0;
    __result_32x2 = {0, 0};
  }
  else if (__is_exp_ovfl)
  {
    __fp64_ovfl_sat<_Rm>(__is_sign_c, __result_exp, __result_32x2);
  }
  else
  {
    // Check for negative exponent
    if (__result_exp < 0)
    {
      __result_exp = 0;
    }
  }

  // Set the sign bit
  __result_32x2.x[1] |= (uint64_t) __result_sign;

  // Set the exponent
  __result_32x2.x[1] |= (uint64_t) __result_exp << _CCCL_FP64_HI_MANT_SHIFT;

  __result = __fpemu_bit_cast<uint64_t>(__result_32x2);
  return __result;
} // __internal_fp64emu_mid_dmul

//! @brief Fast double-precision multiplication for FPEMU
//!
//! This function performs double-precision multiplication on two FPEMU floating-point numbers,
//! by single precision multiplication for the mantissa (fast mode).
//! It takes two __fpbits64 structures as input, representing the packed sign, exponent,
//! and mantissa fields of the operands. The multiplication is performed according to the specified rounding mode,
//! accuracy, range, and engine template parameters.
template <__fpemu_rounding _Rm = __fpemu_rounding::def, fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_low_dmul(__fpbits64 __x, __fpbits64 __y) noexcept
{
  __uint32x2 __a_32x2 = __fpemu_bit_cast<__uint32x2>(__x);
  __uint32x2 __b_32x2 = __fpemu_bit_cast<__uint32x2>(__y);
  __fpbits64 __result;

  // Extract exponents
  uint32_t __exp_a    = (__a_32x2.x[1] >> _CCCL_FP64_HI_MANT_SHIFT) & _CCCL_FP64_LO_EXP_MASK;
  uint32_t __exp_b    = (__b_32x2.x[1] >> _CCCL_FP64_HI_MANT_SHIFT) & _CCCL_FP64_LO_EXP_MASK;
  bool __is_exps_zero = ((__exp_a == 0) || (__exp_b == 0));

  // Compute result sign (XOR of input signs)
  uint32_t __result_sign = (__a_32x2.x[1] ^ __b_32x2.x[1]) & _CCCL_FP64_HI_SIGN_MASK;

  // Add exponents and subtract fp64 bias (1023)
  int32_t __result_exp = (int32_t) __exp_a + (int32_t) __exp_b - _CCCL_FP64_BIAS;

  // Multiply the mantissas
  __uint32x2 __result_32x2;

  // Convert mantissas to single precision for multiplication
  // Extract the upper 24 bits of the 53-bit mantissa
  // by shifting the bits left by 3
  uint32_t __mant_a_sp =
    ((__a_32x2.x[0] >> _CCCL_FP64_MANT_TO_FP32_LO_SHIFT) | (__a_32x2.x[1] << _CCCL_FP64_MANT_TO_FP32_HI_SHIFT))
    & _CCCL_FP32_MANT_MASK;
  uint32_t __mant_b_sp =
    ((__b_32x2.x[0] >> _CCCL_FP64_MANT_TO_FP32_LO_SHIFT) | (__b_32x2.x[1] << _CCCL_FP64_MANT_TO_FP32_HI_SHIFT))
    & _CCCL_FP32_MANT_MASK;

  // Normalize to [1.0, 2.0) by adding the 1.0 exponent
  __mant_a_sp = _CCCL_FP32_ONE | __mant_a_sp;
  __mant_b_sp = _CCCL_FP32_ONE | __mant_b_sp;

  // Convert to single precision for multiplication
  float __mant_a_float = __fpemu_bit_cast<float>(__mant_a_sp);
  float __mant_b_float = __fpemu_bit_cast<float>(__mant_b_sp);

  // Perform single precision multiplication with directed rounding on device
  float __mant_product_float = __fmul_dir<_Rm>(__mant_a_float, __mant_b_float);

  // Extract mantissa directly from the float result
  uint32_t __mant_product_bits = __fpemu_bit_cast<uint32_t>(__mant_product_float);

  // Extract 23-bit mantissa
  uint32_t __result_mant = __mant_product_bits & _CCCL_FP32_MANT_MASK;

  // Convert back to fp64 by shifting the mantissa to the right by 3 bits
  __result_32x2.x[0] = __result_mant << _CCCL_FP64_MANT_TO_FP32_LO_SHIFT;
  __result_32x2.x[1] = __result_mant >> _CCCL_FP64_MANT_TO_FP32_HI_SHIFT;

  // Correct exponent field by 1 if the product is >= 2.0
  __result_exp = (__mant_product_bits >= _CCCL_FP32_TWO) ? __result_exp + 1 : __result_exp;

  // Check for exponent overflow
  bool __is_exp_ovfl = (__result_exp > (_CCCL_FP64_BIAS * 2));

  const bool __is_sign_c = (__result_sign != 0);

  if (__is_exps_zero)
  {
    __result_exp  = 0;
    __result_32x2 = {0, 0};
  }
  else if (__is_exp_ovfl)
  {
    __fp64_ovfl_sat<_Rm>(__is_sign_c, __result_exp, __result_32x2);
  }
  else
  {
    if (__result_exp < 0)
    {
      __result_exp = 0;
    }
  }

  // Set the sign bit
  __result_32x2.x[1] |= (uint64_t) __result_sign;

  // Set the exponent
  __result_32x2.x[1] |= (uint64_t) __result_exp << _CCCL_FP64_HI_MANT_SHIFT;

  __result = __fpemu_bit_cast<uint64_t>(__result_32x2);
  return __result;
} // __internal_fp64emu_low_dmul

//! @brief Pure MUL core on the unpacked representation (unified path).
//!
//! Consumes/produces __fpbits64_unpacked exactly as the universal
//! __internal_fp64emu_unpack / __internal_fp64emu_pack. This
//! is the multiply section of the unified FMA core, made standalone: the
//! significand product normalized so the high 64 bits carry the universal
//! mantissa (implicit bit at position 61) with a sticky LSB, and the product
//! exponent in the pack's "+1" convention.
//!
//!   - accurate (cr/full): returns the pre-rounding intermediate; the universal
//!     full pack does the single correct rounding plus inf/denormal/overflow.
//!     inf*0 -> NaN is folded into the exponent band; inf*finite -> inf rides
//!     the huge unpack exponent that pack classifies as inf.
//!   - def (ha/normal): same 128-bit product, but the lean pack does no special
//!     handling, so the core flushes denormal-inputs / underflow to signed zero
//!     and saturates overflow, exactly as the legacy fused def kernel did.
//!   - fast (la/normal): fp32 product of the top significand bits (the legacy
//!     fast kernel), re-expressed on the universal scale, same flush/saturate.
template <fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API __fpbits64_unpacked
__internal_fp64emu_dmul_unpacked(__fpbits64_unpacked __a, __fpbits64_unpacked __b) noexcept
{
  constexpr fpemu_accuracy __acc_forced = fpemu_accuracy::_CCCL_FPEMU_MUL_METHOD;
  constexpr fpemu_accuracy __acc_used   = (__acc_forced != fpemu_accuracy::unset) ? __acc_forced : _Acc;
  // Unpacked cores always run on the fully-accurate full-range unpack/pack
  // boundary (accuracy-independent), so the inf*0 -> NaN exponent fold is
  // always live regardless of the accuracy level's arithmetic precision.
  // The accuracy level selects only the *arithmetic*: high uses the 128-bit
  // product, mid the high-64 only, low the fp32 product. The
  // boundary contract is the SAME for every accuracy level here -- this is the
  // full (high) bit-61 boundary: encode inf*0 -> NaN in the exponent band
  // and defer subnormal emission / inf / overflow / rounding to the (full)
  // pack. So mid/low cores compose correctly with the high pack/unpack
  // (Model 1 subsumption) with no range parameter -- range stays internal to
  // the pack/unpack, never threaded through the core. For mid/low the lean
  // (FTZ/DAZ) boundary handling lives in the mid/low pack/unpack, not here.
  constexpr int32_t __nan_exp = 0x0007ff00;

  const uint32_t __sign_ab = __a.sign ^ __b.sign;
  __fpbits64_unpacked __r;

  if constexpr (__acc_used == fpemu_accuracy::high)
  {
    // ---- accurate: full product -> correctly-rounded by pack ----
    // The low product bits feed the sticky LSB so the universal full pack
    // does a single correct rounding.
    __uint32x2 __a_32x2 = __fpemu_bit_cast<__uint32x2>(__a.mantissa);
    __uint32x2 __b_32x2 = __fpemu_bit_cast<__uint32x2>(__b.mantissa);
    __uint32x4 __prod   = __mul_128<__acc_used>(__a_32x2, __b_32x2);

    int32_t __exponent_ab = (int32_t) __a.exponent + (int32_t) __b.exponent - (int32_t) __fpemu_bias;
    // mul_nzeros == 1 when the leading product bit landed one place low
    // (significand in [1,2) rather than [2,4)); normalize the implicit bit
    // to position 61 in the high word.
    int __mul_nzeros = __prod.hi.x[1] < 0x08000000;
    int32_t __e      = __exponent_ab - __mul_nzeros + 1;
    // Full range: inf*0 -> NaN (inf*finite -> inf rides the exponent band).
    if (__e == (int32_t) __fpemu_inf_zero)
    {
      __e = __nan_exp;
    }

    const int __sh = (11 - EXTRA_BITS) + __mul_nzeros; // 2 or 3
    // 64-bit normalization (no __uint128_t). Shift the high 64 bits up by
    // sh, pulling in the top sh bits of the low word (the guard bits), and
    // fold the remaining low bits into the sticky LSB -- this is how the
    // legacy accurate kernel stays in 64-bit.
    uint64_t __hi     = __fpemu_bit_cast<uint64_t>(__prod.hi);
    uint64_t __lo     = __fpemu_bit_cast<uint64_t>(__prod.lo);
    uint64_t __hi_n   = (__hi << __sh) | (__lo >> (64 - __sh));
    uint64_t __sticky = ((__lo << __sh) != 0) ? 1u : 0u;
    __r.sign          = __sign_ab;
    __r.exponent      = static_cast<uint32_t>(__e);
    __r.mantissa      = __hi_n | __sticky;
    return __r;
  }
  else if constexpr (__acc_used == fpemu_accuracy::mid)
  {
    // ---- def: high-64 product only (no low-64 / sticky) -----------------
    // High-accuracy tolerates dropping the low product, so def uses the
    // cheaper __mul_64 (top 64 bits) exactly like the legacy fused def
    // kernel -- this keeps the GPU multiply lean. The high word carries the
    // implicit bit at position 58/59 (significand in [2,4)/[1,2)); shift it
    // up to position 61 (the universal scale). The exponent arithmetic and
    // special handling are identical to the cr branch (only the multiply is
    // cheaper): inf*0 -> NaN folds into the exponent band, and inf/overflow/
    // subnormal/rounding are all emitted by the full pack. This is what lets
    // a def core run on the accurate pack/unpack (~2 ulp on normals).
    __uint32x2 __a_32x2 = __fpemu_bit_cast<__uint32x2>(__a.mantissa);
    __uint32x2 __b_32x2 = __fpemu_bit_cast<__uint32x2>(__b.mantissa);
    __uint32x2 __hi     = __mul_64<__acc_used>(__a_32x2, __b_32x2);

    int32_t __exponent_ab = (int32_t) __a.exponent + (int32_t) __b.exponent - (int32_t) __fpemu_bias;
    int __mul_nzeros      = __hi.x[1] < 0x08000000;
    int32_t __e           = __exponent_ab - __mul_nzeros + 1;
    // inf*0 -> NaN: dead on the lean def/fast boundary (no INF magic), so
    // gate it out of the hot exponent chain. Kept only for full (cr).
    if (__e == (int32_t) __fpemu_inf_zero)
    {
      __e = __nan_exp;
    }

    uint64_t __m = __fpemu_bit_cast<uint64_t>(__hi) << (11 - EXTRA_BITS + __mul_nzeros);
    __r.sign     = __sign_ab;
    __r.exponent = static_cast<uint32_t>(__e);
    __r.mantissa = __m;
    return __r;
  }
  else
  {
    // ---- fp32 significand product (fast / la) ----
    // The legacy fast kernel multiplies the top 24 significand bits in fp32.
    // At ~half mantissa precision directed rounding of the fp32 product is
    // meaningless, so this uses plain round-to-nearest fp32 (the final
    // directed rounding is the pack's job). The universal mantissa carries
    // the top 23 fraction bits at positions 60..38 (implicit bit at 61), so
    // we rebuild the fp32 [1,2) operands directly and re-expand the product.
    int32_t __exp_a = (int32_t) __a.exponent;
    int32_t __exp_b = (int32_t) __b.exponent;

    uint32_t __mant_a_sp = ((uint32_t) (__a.mantissa >> 38) & _CCCL_FP32_MANT_MASK) | _CCCL_FP32_ONE;
    uint32_t __mant_b_sp = ((uint32_t) (__b.mantissa >> 38) & _CCCL_FP32_MANT_MASK) | _CCCL_FP32_ONE;

    float __fa       = __fpemu_bit_cast<float>(__mant_a_sp);
    float __fb       = __fpemu_bit_cast<float>(__mant_b_sp);
    float __fp       = __fmul_dir<__fpemu_rounding::rn>(__fa, __fb);
    uint32_t __pbits = __fpemu_bit_cast<uint32_t>(__fp);

    // Product of two [1,2) significands lands in [1,4): a binade bump when
    // it reaches [2,4) carries into the result exponent. Express the exponent
    // in the cr convention (E = exponent_ab + binade) so the inf*0 -> NaN
    // fold is identical to cr; inf/overflow/subnormal are deferred to pack.
    int32_t __binade       = (__pbits >= _CCCL_FP32_TWO) ? 1 : 0;
    int32_t __exponent_ab  = __exp_a + __exp_b - (int32_t) __fpemu_bias;
    int32_t __e            = __exponent_ab + __binade;
    uint32_t __result_mant = __pbits & _CCCL_FP32_MANT_MASK;
    // inf*0 -> NaN: dead on the lean def/fast boundary; gate it off the hot
    // path (kept only for full / cr).
    if (__e == (int32_t) __fpemu_inf_zero)
    {
      __e = __nan_exp;
    }

    // Re-expand the 24-bit fp32 significand back to the universal scale.
    uint64_t __sig = ((uint64_t) ((1u << _CCCL_FP32_MANT_BITS) | __result_mant)) << 38;
    // The fp32 rebuild fabricates a 1.0 significand even for a zero operand
    // (on the full boundary zero has a normalized exponent, not exp==0), so a
    // true zero would leak a tiny subnormal. Force the significand to zero; a
    // finite zero then packs to signed zero, while 0*inf already folded E to
    // NAN_EXP above and the pack overrides the mantissa to NaN.
    if (__a.mantissa == 0 || __b.mantissa == 0)
    {
      __sig = 0;
    }
    __r.sign     = __sign_ab;
    __r.exponent = static_cast<uint32_t>(__e);
    __r.mantissa = __sig;
    return __r;
  }
} // __internal_fp64emu_dmul_unpacked
//! @brief Emulation of double-precision floating-point multiplication.
//!
//! This function provides an emulated implementation of the IEEE-754
//! double-precision (fp64) multiplication operation. It operates on the
//! internal bitwise representation of fp64 numbers (__fpbits64) and
//! supports configurable rounding modes, accuracy, and range
//! selection via template parameters.
//!
//! The emulation performs the following steps:
//!   - Extracts the exponent and mantissa fields from the input operands.
//!   - Computes the result sign as the XOR of the input signs.
//!   - Adds the exponents and subtracts the fp64 bias to obtain the result exponent.
//!   - Multiplies the mantissas by a dedicated helper function, handling
//!     normalization and carry propagation.
//!   - Checks for exponent overflow and underflow, saturating or zeroing the
//!     result as appropriate.
//!   - Assembles the final result by setting the sign, exponent, and mantissa
//!     fields in the output bit pattern.
//!
//! This emulation is designed to be bit-accurate and to match the behavior
//! of hardware or reference software implementations, including correct
//! handling of special cases such as zero, infinity, and NaN.
//!
//! The function is intended for use in both host and device code, and is
//! selected as the optimized multiplication path when the appropriate
//! macros are enabled.
template <__fpemu_rounding _Rm = __fpemu_rounding::def, fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_dmul(__fpbits64 __x, __fpbits64 __y) noexcept
{
  // Forced method override for the multiplication operation
  constexpr fpemu_accuracy __acc_forced = fpemu_accuracy::_CCCL_FPEMU_MUL_METHOD;
  constexpr fpemu_accuracy __acc_used   = (__acc_forced != fpemu_accuracy::unset) ? __acc_forced : _Acc;
  {
#if (_CCCL_FPEMU_PACKED_VIA_UNPACKED == 1)
    {
      // Packed-via-unpacked (testing): pack(dmul_unpacked(unpack(x), unpack(y))). One
      // bit-61 core for every accuracy level (high/mid/low); the core
      // selects only the multiply precision and defers subnormal/inf/
      // overflow to the pack. The boundary is the accuracy level's own:
      // high -> full (IEEE), mid/low -> normal (FTZ/DAZ).
      __fpbits64_unpacked __a = __internal_fp64emu_unpack(__x);
      __fpbits64_unpacked __b = __internal_fp64emu_unpack(__y);
      __fpbits64_unpacked __r = __internal_fp64emu_dmul_unpacked<__acc_used>(__a, __b);
      return __internal_fp64emu_pack<_Rm>(__r);
    }
#else
    if constexpr (__acc_used == fpemu_accuracy::high)
    {
      return __internal_fp64emu_high_dmul<_Rm, __acc_used>(__x, __y);
    }
    else if constexpr (__acc_used == fpemu_accuracy::mid)
    {
      return __internal_fp64emu_mid_dmul<_Rm, __acc_used>(__x, __y);
    }
    else if constexpr (__acc_used == fpemu_accuracy::low)
    {
#  if _CCCL_FP64EMU_DMUL_FP32_FAST_ENABLE == 1
      return __internal_fp64emu_low_dmul<_Rm, __acc_used>(__x, __y);
#  else
      return __internal_fp64emu_mid_dmul<_Rm, __acc_used>(__x, __y);
#  endif
    }
    else
    {
      return __internal_fp64emu_mid_dmul<_Rm, __acc_used>(__x, __y);
    }
#endif
  }
} // __internal_fp64emu_dmul

// ============================================================================
// Builtin declarations/implementations for multiplication operations
// ============================================================================
#if defined(_CCCL_FPEMU_INLINE)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dmul_rn(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dmul<__fpemu_rounding::rn, fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dmul_rz(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dmul<__fpemu_rounding::rz, fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dmul_ru(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dmul<__fpemu_rounding::ru, fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dmul_rd(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dmul<__fpemu_rounding::rd, fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_high_dmul_rn(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dmul<__fpemu_rounding::rn, fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dmul_rn(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dmul<__fpemu_rounding::rn, fpemu_accuracy::mid>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dmul_rz(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dmul<__fpemu_rounding::rz, fpemu_accuracy::mid>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dmul_ru(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dmul<__fpemu_rounding::ru, fpemu_accuracy::mid>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dmul_rd(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dmul<__fpemu_rounding::rd, fpemu_accuracy::mid>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dmul_rn(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dmul<__fpemu_rounding::rn, fpemu_accuracy::low>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dmul_rz(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dmul<__fpemu_rounding::rz, fpemu_accuracy::low>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dmul_ru(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dmul<__fpemu_rounding::ru, fpemu_accuracy::low>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dmul_rd(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dmul<__fpemu_rounding::rd, fpemu_accuracy::low>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_dmul(__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
{
  return __internal_fp64emu_dmul_unpacked<fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_high_dmul(__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
{
  return __internal_fp64emu_dmul_unpacked<fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_mid_dmul(__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
{
  return __internal_fp64emu_dmul_unpacked<fpemu_accuracy::mid>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_low_dmul(__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
{
  return __internal_fp64emu_dmul_unpacked<fpemu_accuracy::low>(__x, __y);
}
#else
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dmul_rn(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dmul_rz(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dmul_ru(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dmul_rd(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_high_dmul_rn(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dmul_rn(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dmul_rz(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dmul_ru(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dmul_rd(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dmul_rn(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dmul_rz(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dmul_ru(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dmul_rd(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_dmul(__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_high_dmul(__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_mid_dmul(__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_low_dmul(__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept;
#endif // _CCCL_FPEMU_INLINE
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_MUL_H (builtins)

#if defined(_CCCL_FPEMU_API_CLASSES_DEFINED) && !defined(_CCCL_FPEMU_DMUL_API_MERGED)
#define _CCCL_FPEMU_DMUL_API_MERGED
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// ============================================================================
// API (merged from fp64emu_dmul_api.hpp)
// ============================================================================

// Default API implementation
template <fpemu_accuracy _Acc>
_CCCL_DEVICE_API fpemu<double, _Acc> operator*(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(__fp64emu_high_dmul_rn(__x.__bits_, __y.__bits_));
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(__fp64emu_mid_dmul_rn(__x.__bits_, __y.__bits_));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(__fp64emu_low_dmul_rn(__x.__bits_, __y.__bits_));
  }
  else
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(__fp64emu_dmul_rn(__x.__bits_, __y.__bits_));
  }
} // operator*

template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc> __dmul_rn(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_high_dmul_rn(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_low_dmul_rn(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_mid_dmul_rn(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc> __dmul_rz(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_dmul_rz(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_mid_dmul_rz(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_low_dmul_rz(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_dmul_rz(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc> __dmul_ru(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_dmul_ru(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_mid_dmul_ru(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_low_dmul_ru(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_dmul_ru(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc> __dmul_rd(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_dmul_rd(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_mid_dmul_rd(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_low_dmul_rd(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_dmul_rd(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
}

// Operator* for unpacked multiplication
template <fpemu_accuracy _Acc>
_CCCL_DEVICE_API fpemu_unpacked<double, _Acc>
operator*(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_high_dmul(__x.__bits_, __y.__bits_));
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_mid_dmul(__x.__bits_, __y.__bits_));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_low_dmul(__x.__bits_, __y.__bits_));
  }
  else
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_dmul(__x.__bits_, __y.__bits_));
  }
} // operator*

template <fpemu_accuracy _Acc>
_CCCL_API fpemu_unpacked<double, _Acc>
__dmul_rn(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_high_dmul(
      __fpemu_bit_cast<__fpbits64_unpacked>(__x), __fpemu_bit_cast<__fpbits64_unpacked>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_low_dmul(
      __fpemu_bit_cast<__fpbits64_unpacked>(__x), __fpemu_bit_cast<__fpbits64_unpacked>(__y)));
  }
  else
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_mid_dmul(
      __fpemu_bit_cast<__fpbits64_unpacked>(__x), __fpemu_bit_cast<__fpbits64_unpacked>(__y)));
  }
}

// Mixed-operand promoters (relocated from the class body; formerly hidden
// friends). Enabled only when at least one operand is an fpemu and at least
// one is a built-in arithmetic type: both operands are promoted to the fpemu
// type and the exact-match core above is called. Pure fpemu/fpemu calls bind
// to the cores directly; pure arithmetic calls are left to the language.

_CCCL_TEMPLATE(class _T1, class _T2)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2>)
_CCCL_API __fpemu_pick_t<_T1, _T2> __dmul_rn(const _T1& __x, const _T2& __y) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2>;
  return __dmul_rn(_Fp(__x), _Fp(__y));
}

_CCCL_TEMPLATE(class _T1, class _T2)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2>)
_CCCL_API __fpemu_pick_t<_T1, _T2> __dmul_rz(const _T1& __x, const _T2& __y) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2>;
  return __dmul_rz(_Fp(__x), _Fp(__y));
}

_CCCL_TEMPLATE(class _T1, class _T2)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2>)
_CCCL_API __fpemu_pick_t<_T1, _T2> __dmul_ru(const _T1& __x, const _T2& __y) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2>;
  return __dmul_ru(_Fp(__x), _Fp(__y));
}

_CCCL_TEMPLATE(class _T1, class _T2)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2>)
_CCCL_API __fpemu_pick_t<_T1, _T2> __dmul_rd(const _T1& __x, const _T2& __y) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2>;
  return __dmul_rd(_Fp(__x), _Fp(__y));
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_MUL_H
