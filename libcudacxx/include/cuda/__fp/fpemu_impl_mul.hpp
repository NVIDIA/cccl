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

/** 
 * @file fpemu_dmul_impl.hpp
 * @brief Implementation of double-precision multiplication operations for FPEMU floating point emulation library
 *
 * This header provides the implementation of double-precision multiplication operations for the FPEMU library.
 * It includes:
 *
 * - Multiplication functions for fp64emu_t
 * - Multiplication operators for fp64emu_t
 * - Multiplication functions to other types
 *
 * The multiplication functions are designed to work across both host and device code
 * through appropriate decorators and provide bit-exact results matching hardware
 * floating point units.
 */
#define __FP64EMU_USE_MUL_UNPACKED__ 0
#define __FP64EMU_DMUL_FP32_FAST_ENABLE__ 1


#include <cuda/__fp/fpemu_common.hpp>
#include <cuda/__fp/fpemu_impl_utils.hpp>
#include <cuda/__fp/fpemu_impl_unpack.hpp>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

namespace impl
{
    /**
     * @brief v1.0 Emulation of double-precision multiplication for FPEMU
     *
     * This function implements the reference (v1.0) emulation of double-precision
     * floating-point multiplication for the FPEMU format. It operates on the
     * internal bitwise representation (fpbits64_t) of the operands and produces
     * a bit-exact result matching the IEEE-754 standard for double-precision
     * multiplication.
     *
     * The algorithm performs the following steps:
     *   - Extracts the sign, exponent, and mantissa from both operands.
     *   - Handles special cases such as zero, denormalized numbers, infinities, and NaNs.
     *   - Multiplies the mantissas with full precision, including the implicit leading bit.
     *   - Computes the resulting exponent, taking into account normalization and bias.
     *   - Normalizes the result and applies rounding according to the specified mode.
     *   - Packs the sign, exponent, and mantissa back into the fpbits64_t result.
     *
     * This implementation is designed for correctness and bitwise reproducibility,
     * serving as a reference for optimized or hardware-accelerated versions.
     *
     * @tparam rm   Rounding mode (default: nearest-even)
     * @tparam meth Accuracy level (fp64emu_accuracy; default: high)
     * @param x     First operand (fpbits64_t)
     * @param y     Second operand (fpbits64_t)
     * @return      Product as fpbits64_t
     */
    template<fpemu::rounding rm   = fpemu::rounding::def, 
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_dmul_accurate(fpbits64_t x, 
                                                   fpbits64_t y)
    {
        uint64_t a_ = x;
        uint64_t b_ = y;

        fpemu::uint32x2_t a_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(a_);
        fpemu::uint32x2_t b_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(b_);

        fpemu::uint32x2_t man_a_32x2, man_b_32x2, man_c_32x2;
        int32_t exp_a, exp_b, exp_c, shift;
        bool is_sign_a, is_sign_b, is_sign_c, is_a_exp_zero, is_b_exp_zero, is_impl_bit;
        fpbits64_t result;

        //Extract sign and exponent for input A
        exp_a = fpemu::__unpack_exp<meth>(a_32x2); 
        //Extract sign and exponent for input B
        exp_b = fpemu::__unpack_exp<meth>(b_32x2); 

        //Check if input A is denormal or zero
        is_a_exp_zero = (exp_a == 0);
        is_b_exp_zero = (exp_b == 0);
        
        //Extract mantissa for input A
        man_a_32x2 = fpemu::__unpack_mant<meth>(&is_sign_a, a_32x2, is_a_exp_zero);
        //Extract mantissa for input B
        man_b_32x2 = fpemu::__unpack_mant<meth>(&is_sign_b, b_32x2, is_b_exp_zero);
        
        if constexpr (meth != fp64emu_accuracy::high)
        {
            if (is_a_exp_zero) man_a_32x2 = {0, 0};
            if (is_b_exp_zero) man_b_32x2 = {0, 0};
        }

        // Correct exp and mantissa for denormal
        if constexpr (meth == fp64emu_accuracy::high)
        {
            exp_a       = (is_a_exp_zero)?12 - fpemu::__flo_u64(a_32x2):exp_a;
            exp_b       = (is_b_exp_zero)?12 - fpemu::__flo_u64(b_32x2):exp_b;
            man_a_32x2  = (is_a_exp_zero)?fpemu::__shl_64(man_a_32x2, 1 - exp_a):man_a_32x2;
            man_b_32x2  = (is_b_exp_zero)?fpemu::__shl_64(man_b_32x2, 1 - exp_b):man_b_32x2;                    
        }
        else
        {
            exp_a      = (!is_a_exp_zero)?exp_a:-52;
            exp_b      = (!is_b_exp_zero)?exp_b:-52;
        }

        if constexpr (meth == fp64emu_accuracy::high)
        {
            fpemu::uint32x4_t man_c_32x4 = fpemu::__mul_128<meth>(man_a_32x2, man_b_32x2);
            // Rounding: Set least significant bit to "1" 
            // if low 64 bits are non-zero
            if ((man_c_32x4.lo.x[0] | man_c_32x4.lo.x[1]) != 0) man_c_32x4.hi.x[0] |= 1;
            man_c_32x2 = man_c_32x4.hi;
        }
        else if constexpr (meth == fp64emu_accuracy::mid)
        {
            man_c_32x2 = fpemu::__mul_64<meth>(man_a_32x2, man_b_32x2);
        }
        else
        {
            man_c_32x2.x[1] = fpemu::__mul_32<meth>(man_a_32x2, man_b_32x2);
            man_c_32x2.x[0] = 0;
        }

        //Check implicit-bit position
        is_impl_bit = (man_c_32x2.x[1] >= 0x08000000);

        //Calculate exponent
        exp_c = exp_a + exp_b - (fpemu::BIAS+1) + is_impl_bit;

        //Calculate SIGN
        is_sign_c = is_sign_a ^ is_sign_b;

        // Check for negative exponent
        bool is_exp_c_neg = (exp_c < 0);

        if constexpr (meth == fp64emu_accuracy::high)
        {   
            if (is_exp_c_neg)
            {            
                // shift is negative, so we need to add it to exp_c
                shift = -exp_c;
                exp_c = exp_c + shift;
                // Shift mantissa to the right with rounding 
                // in case of negative exponent (DENORM)
                man_c_32x2 = fpemu::__sar_64_rnd<meth, rm>(man_c_32x2, shift, is_sign_c);
            }
        }
        else
        {
            // Flush denormals to zero
            exp_c = (is_exp_c_neg)?0:exp_c;
            if (is_exp_c_neg) man_c_32x2 = {0, 0};
        }
                                        
        if (is_impl_bit) man_c_32x2 = fpemu::__round<rm>(man_c_32x2, -2, is_sign_c);
        else             man_c_32x2 = fpemu::__round<rm>(man_c_32x2, -3, is_sign_c);

        // Pack sign, exponent and mantissa back to FP64 
        // (+checks for NAN,INF,0)
        result = fpemu::__pack<meth, rm>(is_sign_c, exp_c, man_c_32x2);
        return result;
    } // __nv_internal_fp64emu_dmul_accurate



    /**
     * @brief Version 2.0 emulation of the double-precision multiplication function.
     *
     * This implementation provides an emulation of IEEE-754 double-precision (fp64) multiplication,
     * supporting configurable rounding modes, accuracy, and range options.
     *
     * The function operates by:
     *   - Extracting the exponent and mantissa fields from the input operands.
     *   - Computing the result sign as the XOR of the input signs.
     *   - Adding the exponents and subtracting the fp64 bias to obtain the result exponent.
     *   - Multiplying the mantissas by a dedicated helper (__mul_mant), which handles
     *     normalization, carry, and accuracy-specific bit manipulations.
     *   - Adjusting the result for exponent overflow and underflow, and setting the sign and exponent bits.
     *   - Returning the correctly packed fp64 result.
     *
     * This version is designed for improved accuracy and performance, and is suitable for both
     * host and device execution.
     */
    template<fpemu::rounding rm   = fpemu::rounding::def, 
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_dmul_def(fpbits64_t x, 
                                              fpbits64_t y)
    {
        fpemu::uint32x2_t a_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(x);
        fpemu::uint32x2_t b_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(y);
        fpbits64_t result;
            
        // Extract exponents
        uint32_t exp_a  = (a_32x2.x[1] >> FP64_HI_MANT_SHIFT) & FP64_LO_EXP_MASK;
        uint32_t exp_b  = (b_32x2.x[1] >> FP64_HI_MANT_SHIFT) & FP64_LO_EXP_MASK;
        bool is_exps_zero = ((exp_a == 0) || (exp_b == 0));
                
        // Compute result sign (XOR of input signs)
        uint32_t result_sign = (a_32x2.x[1] ^ b_32x2.x[1]) & FP64_HI_SIGN_MASK;
        
        // Add exponents and subtract fp64 bias (1023)
        int32_t result_exp = (int32_t)exp_a + (int32_t)exp_b - FP64_BIAS;
        
        // Multiply the mantissas
        fpemu::uint32x2_t result_32x2;

        // Integer based mantissa processing
        {
            uint32_t carry_bit;

            // Clear the exponent/sign bits
            a_32x2.x[1] &= FP64_HI_MANT_MASK;
            b_32x2.x[1] &= FP64_HI_MANT_MASK;

            // Set the implicit 1 bit
            a_32x2.x[1] |= ( 1 << FP64_HI_MANT_SHIFT );
            b_32x2.x[1] |= ( 1 << FP64_HI_MANT_SHIFT );  
        
            if constexpr (meth == fp64emu_accuracy::mid)
            {
                // Shift the mantissa to the left by FP64_EXTRA_BITS (9) bits to preserve low bits
                a_32x2 = fpemu::__shl_64(a_32x2, FP64_EXTRA_BITS);
                b_32x2 = fpemu::__shl_64(b_32x2, FP64_EXTRA_BITS);

                // Multiply the mantissas
                result_32x2 = fpemu::__mul_64<meth>(a_32x2, b_32x2);

                // Check if the carry bit is set
                carry_bit   = (result_32x2.x[1] >= (1 << (FP64_MANT_MUL_CARRY_BIT + FP64_EXTRA_BITS*2 + 1)));

                // Shift the mantissa back with directed rounding for ru/rd
                const int mant_shift = (carry_bit)?(((FP64_EXTRA_BITS*2) - FP64_MANT_MUL_SHIFT) + 1):
                                                  ((FP64_EXTRA_BITS*2) - FP64_MANT_MUL_SHIFT);
                result_32x2 = fpemu::__shr_64_rnd<rm>(result_32x2, mant_shift, result_sign != 0);
            }
            else
            {
                // Multiply the mantissas
                result_32x2 = fpemu::__mul_64<meth>(a_32x2, b_32x2);

                // Check if the carry bit is set
                carry_bit   = (result_32x2.x[1] >= (1 << (FP64_MANT_MUL_CARRY_BIT + 1)));

                // Shift the mantissa to the left by the correct amount taking into account the carry bit
                result_32x2 = fpemu::__shl_64(result_32x2, (carry_bit)?(FP64_MANT_MUL_SHIFT - 1):
                                                                 FP64_MANT_MUL_SHIFT);
            }

            // Add the carry bit to the exponent
            result_exp = result_exp + carry_bit;

            // Clear the unused high bits
            result_32x2.x[1] &= FP64_HI_MANT_MASK;
        }

        // Check for exponent overflow
        bool is_exp_ovfl = (result_exp > (FP64_BIAS*2));

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
        const bool is_sign_c = (result_sign != 0);

        if (is_exps_zero)
        {
            result_exp = 0;
            result_32x2 = {0, 0};
        }
        else if (is_exp_ovfl)
        {
            fpemu::__fp64_ovfl_sat<rm>(is_sign_c, result_exp, result_32x2);
        }
        else
        {
            // Check for negative exponent
            if (result_exp < 0) result_exp = 0;
        }

        // Set the sign bit
        result_32x2.x[1] |= (uint64_t)result_sign;

        // Set the exponent
        result_32x2.x[1] |= (uint64_t)result_exp << FP64_HI_MANT_SHIFT;

        result = fpemu::bit_cast<uint64_t>(result_32x2);
        return result;
    } // __nv_internal_fp64emu_dmul_def

    /**
     * @brief Fast double-precision multiplication for FPEMU
     * 
     * This function performs double-precision multiplication on two FPEMU floating-point numbers,
     * by single precision multiplication for the mantissa (fast mode).
     * It takes two fpbits64_t structures as input, representing the packed sign, exponent,
     * and mantissa fields of the operands. The multiplication is performed according to the specified rounding mode,
     * accuracy, range, and engine template parameters.
     */
    template<fpemu::rounding rm   = fpemu::rounding::def, 
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_dmul_fast(fpbits64_t x, 
                                               fpbits64_t y)
    {
        fpemu::uint32x2_t a_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(x);
        fpemu::uint32x2_t b_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(y);
        fpbits64_t result;
            
        // Extract exponents
        uint32_t exp_a  = (a_32x2.x[1] >> FP64_HI_MANT_SHIFT) & FP64_LO_EXP_MASK;
        uint32_t exp_b  = (b_32x2.x[1] >> FP64_HI_MANT_SHIFT) & FP64_LO_EXP_MASK;
        bool is_exps_zero = ((exp_a == 0) || (exp_b == 0));
                
        // Compute result sign (XOR of input signs)
        uint32_t result_sign = (a_32x2.x[1] ^ b_32x2.x[1]) & FP64_HI_SIGN_MASK;
        
        // Add exponents and subtract fp64 bias (1023)
        int32_t result_exp = (int32_t)exp_a + (int32_t)exp_b - FP64_BIAS;
        
        // Multiply the mantissas
        fpemu::uint32x2_t result_32x2;

        // Convert mantissas to single precision for multiplication
        // Extract the upper 24 bits of the 53-bit mantissa
        // by shifting the bits left by 3
        uint32_t mant_a_sp = ((a_32x2.x[0] >> FP64_MANT_TO_FP32_LO_SHIFT) | 
                              (a_32x2.x[1] << FP64_MANT_TO_FP32_HI_SHIFT)) & FP32_MANT_MASK;
        uint32_t mant_b_sp = ((b_32x2.x[0] >> FP64_MANT_TO_FP32_LO_SHIFT) | 
                              (b_32x2.x[1] << FP64_MANT_TO_FP32_HI_SHIFT)) & FP32_MANT_MASK;

        // Normalize to [1.0, 2.0) by adding the 1.0 exponent
        mant_a_sp = FP32_ONE | mant_a_sp;
        mant_b_sp = FP32_ONE | mant_b_sp;

        // Convert to single precision for multiplication
        float mant_a_float = fpemu::bit_cast<float>(mant_a_sp);  
        float mant_b_float = fpemu::bit_cast<float>(mant_b_sp);

        // Perform single precision multiplication with directed rounding on device
        float mant_product_float = fpemu::__fmul_dir<rm>(mant_a_float, mant_b_float);

        // Extract mantissa directly from the float result
        uint32_t mant_product_bits = fpemu::bit_cast<uint32_t>(mant_product_float);

        // Extract 23-bit mantissa
        uint32_t result_mant = mant_product_bits & FP32_MANT_MASK;  

        // Convert back to fp64 by shifting the mantissa to the right by 3 bits
        result_32x2.x[0] = result_mant << FP64_MANT_TO_FP32_LO_SHIFT;
        result_32x2.x[1] = result_mant >> FP64_MANT_TO_FP32_HI_SHIFT;

        // Correct exponent field by 1 if the product is >= 2.0
        result_exp = (mant_product_bits >= FP32_TWO)? result_exp + 1:result_exp;

        // Check for exponent overflow
        bool is_exp_ovfl = (result_exp > (FP64_BIAS*2));

        const bool is_sign_c = (result_sign != 0);

        if (is_exps_zero)
        {
            result_exp = 0;
            result_32x2 = {0, 0};
        }
        else if (is_exp_ovfl)
        {
            fpemu::__fp64_ovfl_sat<rm>(is_sign_c, result_exp, result_32x2);
        }
        else
        {
            if (result_exp < 0) result_exp = 0;
        }

        // Set the sign bit
        result_32x2.x[1] |= (uint64_t)result_sign;

        // Set the exponent
        result_32x2.x[1] |= (uint64_t)result_exp << FP64_HI_MANT_SHIFT;

        result = fpemu::bit_cast<uint64_t>(result_32x2);
        return result;
    } // __nv_internal_fp64emu_dmul_fast

    /**
     * @brief Pure MUL core on the unpacked representation (unified path).
     *
     * Consumes/produces fpbits64_unpacked_t exactly as the universal
     * impl::__nv_internal_fp64emu_unpack / impl::__nv_internal_fp64emu_pack. This
     * is the multiply section of the unified FMA core, made standalone: the
     * significand product normalized so the high 64 bits carry the universal
     * mantissa (implicit bit at position 61) with a sticky LSB, and the product
     * exponent in the pack's "+1" convention.
     *
     *   - accurate (cr/full): returns the pre-rounding intermediate; the universal
     *     full pack does the single correct rounding plus inf/denormal/overflow.
     *     inf*0 -> NaN is folded into the exponent band; inf*finite -> inf rides
     *     the huge unpack exponent that pack classifies as inf.
     *   - def (ha/normal): same 128-bit product, but the lean pack does no special
     *     handling, so the core flushes denormal-inputs / underflow to signed zero
     *     and saturates overflow, exactly as the legacy fused def kernel did.
     *   - fast (la/normal): fp32 product of the top significand bits (the legacy
     *     fast kernel), re-expressed on the universal scale, same flush/saturate.
     */
    template<fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_unpacked_t __nv_internal_fp64emu_dmul_unpacked(fpbits64_unpacked_t a,
                                                        fpbits64_unpacked_t b)
    {
        constexpr fp64emu_accuracy   meth_forced = fp64emu_accuracy::__FPEMU_MUL_METHOD__;
        constexpr fp64emu_accuracy   meth_used   = (meth_forced != fp64emu_accuracy::unset) ? meth_forced : meth;
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
        constexpr int32_t NAN_EXP = 0x0007ff00;

        const uint32_t sign_ab = a.sign ^ b.sign;
        fpbits64_unpacked_t r;

        if constexpr (meth_used == fp64emu_accuracy::high)
        {
            // ---- accurate: full product -> correctly-rounded by pack ----
            // The low product bits feed the sticky LSB so the universal full pack
            // does a single correct rounding.
            fpemu::uint32x2_t a_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(a.mantissa);
            fpemu::uint32x2_t b_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(b.mantissa);
            fpemu::uint32x4_t prod   = fpemu::__mul_128<meth_used>(a_32x2, b_32x2);

            int32_t exponent_ab = (int32_t)a.exponent + (int32_t)b.exponent - (int32_t)fpemu::BIAS;
            // mul_nzeros == 1 when the leading product bit landed one place low
            // (significand in [1,2) rather than [2,4)); normalize the implicit bit
            // to position 61 in the high word.
            int mul_nzeros = prod.hi.x[1] < 0x08000000;
            int32_t E = exponent_ab - mul_nzeros + 1;
            // Full range: inf*0 -> NaN (inf*finite -> inf rides the exponent band).
            if (E == (int32_t)fpemu::INF_ZERO) E = NAN_EXP;

            const int sh = (11 - EXTRA_BITS) + mul_nzeros;  // 2 or 3
            // 64-bit normalization (no __uint128_t). Shift the high 64 bits up by
            // sh, pulling in the top sh bits of the low word (the guard bits), and
            // fold the remaining low bits into the sticky LSB -- this is how the
            // legacy accurate kernel stays in 64-bit.
            uint64_t hi     = fpemu::bit_cast<uint64_t>(prod.hi);
            uint64_t lo     = fpemu::bit_cast<uint64_t>(prod.lo);
            uint64_t hi_n   = (hi << sh) | (lo >> (64 - sh));
            uint64_t sticky = ((lo << sh) != 0) ? 1u : 0u;
            r.sign     = sign_ab;
            r.exponent = static_cast<uint32_t>(E);
            r.mantissa = hi_n | sticky;
            return r;
        }
        else if constexpr (meth_used == fp64emu_accuracy::mid)
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
            fpemu::uint32x2_t a_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(a.mantissa);
            fpemu::uint32x2_t b_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(b.mantissa);
            fpemu::uint32x2_t hi = fpemu::__mul_64<meth_used>(a_32x2, b_32x2);

            int32_t exponent_ab = (int32_t)a.exponent + (int32_t)b.exponent - (int32_t)fpemu::BIAS;
            int mul_nzeros = hi.x[1] < 0x08000000;
            int32_t E = exponent_ab - mul_nzeros + 1;
            // inf*0 -> NaN: dead on the lean def/fast boundary (no INF magic), so
            // gate it out of the hot exponent chain. Kept only for full (cr).
            if (E == (int32_t)fpemu::INF_ZERO) E = NAN_EXP;

            uint64_t m = fpemu::bit_cast<uint64_t>(hi) << (11 - EXTRA_BITS + mul_nzeros);
            r.sign     = sign_ab;
            r.exponent = static_cast<uint32_t>(E);
            r.mantissa = m;
            return r;
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
            int32_t exp_a = (int32_t)a.exponent;
            int32_t exp_b = (int32_t)b.exponent;

            uint32_t mant_a_sp = ((uint32_t)(a.mantissa >> 38) & FP32_MANT_MASK) | FP32_ONE;
            uint32_t mant_b_sp = ((uint32_t)(b.mantissa >> 38) & FP32_MANT_MASK) | FP32_ONE;

            float fa = fpemu::bit_cast<float>(mant_a_sp);
            float fb = fpemu::bit_cast<float>(mant_b_sp);
            float fp = fpemu::__fmul_dir<fpemu::rounding::rn>(fa, fb);
            uint32_t pbits = fpemu::bit_cast<uint32_t>(fp);

            // Product of two [1,2) significands lands in [1,4): a binade bump when
            // it reaches [2,4) carries into the result exponent. Express the exponent
            // in the cr convention (E = exponent_ab + binade) so the inf*0 -> NaN
            // fold is identical to cr; inf/overflow/subnormal are deferred to pack.
            int32_t  binade      = (pbits >= FP32_TWO) ? 1 : 0;
            int32_t  exponent_ab = exp_a + exp_b - (int32_t)fpemu::BIAS;
            int32_t  E           = exponent_ab + binade;
            uint32_t result_mant = pbits & FP32_MANT_MASK;
            // inf*0 -> NaN: dead on the lean def/fast boundary; gate it off the hot
            // path (kept only for full / cr).
            if (E == (int32_t)fpemu::INF_ZERO) E = NAN_EXP;

            // Re-expand the 24-bit fp32 significand back to the universal scale.
            uint64_t sig = ((uint64_t)((1u << FP32_MANT_BITS) | result_mant)) << 38;
            // The fp32 rebuild fabricates a 1.0 significand even for a zero operand
            // (on the full boundary zero has a normalized exponent, not exp==0), so a
            // true zero would leak a tiny subnormal. Force the significand to zero; a
            // finite zero then packs to signed zero, while 0*inf already folded E to
            // NAN_EXP above and the pack overrides the mantissa to NaN.
            if (a.mantissa == 0 || b.mantissa == 0) sig = 0;
            r.sign     = sign_ab;
            r.exponent = static_cast<uint32_t>(E);
            r.mantissa = sig;
            return r;
        }
    } // __nv_internal_fp64emu_dmul_unpacked
    /**
     * @brief Emulation of double-precision floating-point multiplication.
     *
     * This function provides an emulated implementation of the IEEE-754
     * double-precision (fp64) multiplication operation. It operates on the
     * internal bitwise representation of fp64 numbers (fpbits64_t) and
     * supports configurable rounding modes, accuracy, and range
     * selection via template parameters.
     *
     * The emulation performs the following steps:
     *   - Extracts the exponent and mantissa fields from the input operands.
     *   - Computes the result sign as the XOR of the input signs.
     *   - Adds the exponents and subtracts the fp64 bias to obtain the result exponent.
     *   - Multiplies the mantissas by a dedicated helper function, handling
     *     normalization and carry propagation.
     *   - Checks for exponent overflow and underflow, saturating or zeroing the
     *     result as appropriate.
     *   - Assembles the final result by setting the sign, exponent, and mantissa
     *     fields in the output bit pattern.
     *
     * This emulation is designed to be bit-accurate and to match the behavior
     * of hardware or reference software implementations, including correct
     * handling of special cases such as zero, infinity, and NaN.
     *
     * The function is intended for use in both host and device code, and is
     * selected as the optimized multiplication path when the appropriate
     * macros are enabled.
     */
    template<fpemu::rounding rm   = fpemu::rounding::def, 
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_dmul(fpbits64_t x, 
                                          fpbits64_t y)
    {

        // Forced method override for the multiplication operation
        constexpr fp64emu_accuracy   meth_forced = fp64emu_accuracy::__FPEMU_MUL_METHOD__;
        constexpr fp64emu_accuracy   meth_used   = (meth_forced != fp64emu_accuracy::unset) ? meth_forced : meth;
        {
              #if (__FPEMU_PACKED_VIA_UNPACKED__ == 1)
                {
                    // Packed-via-unpacked (testing): pack(dmul_unpacked(unpack(x), unpack(y))). One
                    // bit-61 core for every accuracy level (high/mid/low); the core
                    // selects only the multiply precision and defers subnormal/inf/
                    // overflow to the pack. The boundary is the accuracy level's own:
                    // high -> full (IEEE), mid/low -> normal (FTZ/DAZ).
                    fpbits64_unpacked_t a = __nv_internal_fp64emu_unpack(x);
                    fpbits64_unpacked_t b = __nv_internal_fp64emu_unpack(y);
                    fpbits64_unpacked_t r = __nv_internal_fp64emu_dmul_unpacked<meth_used>(a, b);
                    return __nv_internal_fp64emu_pack<rm>(r);
                }
              #else
                if constexpr (meth_used == fp64emu_accuracy::high)
                {
                    return __nv_internal_fp64emu_dmul_accurate<rm, meth_used>(x, y);
                }
                else if constexpr (meth_used == fp64emu_accuracy::mid)
                {
                    return __nv_internal_fp64emu_dmul_def<rm, meth_used>(x, y);
                }
                else if constexpr (meth_used == fp64emu_accuracy::low)
                {
    #if __FP64EMU_DMUL_FP32_FAST_ENABLE__ == 1
                    return __nv_internal_fp64emu_dmul_fast<rm, meth_used>(x, y);
    #else
                    return __nv_internal_fp64emu_dmul_def<rm, meth_used>(x, y);
    #endif
                }
                else
                {
                    return __nv_internal_fp64emu_dmul_def<rm, meth_used>(x, y);
                }
              #endif
        }
    } // __nv_internal_fp64emu_dmul
} // namespace impl

// ============================================================================
// Builtin declarations/implementations for multiplication operations
// ============================================================================
#if defined(__FPEMU_INLINE__)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dmul_rn (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dmul<fpemu::rounding::rn, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dmul_rz (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dmul<fpemu::rounding::rz, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dmul_ru (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dmul<fpemu::rounding::ru, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dmul_rd (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dmul<fpemu::rounding::rd, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_dmul_rn (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dmul<fpemu::rounding::rn, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dmul_rn      (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dmul<fpemu::rounding::rn, fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dmul_rz      (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dmul<fpemu::rounding::rz, fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dmul_ru      (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dmul<fpemu::rounding::ru, fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dmul_rd      (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dmul<fpemu::rounding::rd, fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dmul_rn     (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dmul<fpemu::rounding::rn, fp64emu_accuracy::low>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dmul_rz     (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dmul<fpemu::rounding::rz, fp64emu_accuracy::low>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dmul_ru     (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dmul<fpemu::rounding::ru, fp64emu_accuracy::low>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dmul_rd     (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dmul<fpemu::rounding::rd, fp64emu_accuracy::low>(x, y); }
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dmul          (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_dmul_unpacked<fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dmul (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_dmul_unpacked<fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dmul      (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_dmul_unpacked<fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dmul     (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_dmul_unpacked<fp64emu_accuracy::low>(x, y); }
#endif
#else
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dmul_rn (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dmul_rz (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dmul_ru (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dmul_rd (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_dmul_rn (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dmul_rn      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dmul_rz      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dmul_ru      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dmul_rd      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dmul_rn     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dmul_rz     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dmul_ru     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dmul_rd     (fpbits64_t x, fpbits64_t y);
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dmul          (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dmul (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dmul      (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dmul     (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
#endif
#endif // __FPEMU_INLINE__

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_MUL_HPP__ (builtins)

#if defined(__FPEMU_API_CLASSES_DEFINED__) && !defined(__FPEMU_DMUL_API_MERGED__)
#define __FPEMU_DMUL_API_MERGED__
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_dmul_api.hpp)
// ============================================================================

    // Default API implementation
    template<fp64emu_accuracy m>  
    __FPEMU_DEVICE_DECL__ static fp64emu_t<m> operator* (const fp64emu_t<m>& x, 
                                                                   const fp64emu_t<m>& y)
    {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_high_dmul_rn(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dmul_rn(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dmul_rn(x.bits, y.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dmul_rn(x.bits, y.bits)); }
    } // operator*


    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dmul_rn (const fp64emu_t<m>& x, const fp64emu_t<m>& y) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_high_dmul_rn(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dmul_rn(x.bits, y.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dmul_rn(x.bits, y.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dmul_rz (const fp64emu_t<m>& x, const fp64emu_t<m>& y) {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dmul_rz(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dmul_rz(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dmul_rz(x.bits, y.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dmul_rz(x.bits, y.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dmul_ru (const fp64emu_t<m>& x, const fp64emu_t<m>& y) {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dmul_ru(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dmul_ru(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dmul_ru(x.bits, y.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dmul_ru(x.bits, y.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dmul_rd (const fp64emu_t<m>& x, const fp64emu_t<m>& y) {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dmul_rd(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dmul_rd(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dmul_rd(x.bits, y.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dmul_rd(x.bits, y.bits)); }
    }

#if __FPEMU_UNPACKED__ == 1

    // Operator* for unpacked multiplication
    template<fp64emu_accuracy m>
    __FPEMU_DEVICE_DECL__ static fp64emu_unpacked_t<m> operator* (const fp64emu_unpacked_t<m>& x, 
                                                                            const fp64emu_unpacked_t<m>& y)
    {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_high_dmul(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_mid_dmul(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_low_dmul(x.bits, y.bits)); }
        else                                             { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_dmul(x.bits, y.bits)); }
    } // operator*


    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<m> __dmul_rn (const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_high_dmul(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_low_dmul(x.bits, y.bits)); }
        else                                             { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_mid_dmul(x.bits, y.bits)); }
    }


#endif // __FPEMU_UNPACKED__ == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_MUL_HPP__
