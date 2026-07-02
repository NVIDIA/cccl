//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_ADD_H
#define _CUDA___FP_FPEMU_IMPL_ADD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/** 
 * @file fpemu_dadd_impl.hpp
 * @brief Implementation of double-precision addition operations for FPEMU floating point emulation library
 *
 * This header provides the implementation of double-precision addition operations for the FPEMU library.
 * It includes:
 *
 * - Addition functions for fp64emu_t
 * - Addition operators for fp64emu_t
 * - Addition functions to other types
 *
 * The addition functions are designed to work across both host and device code
 * through appropriate decorators and provide bit-exact results matching hardware
 * floating point units.
 */

//#define __FP64EMU_ADD_OUTPUT_INF__         0
//#define __FP64EMU_ADD_USE_CARRY_BIT__      0
#define __FP64EMU_DADD_FTZ__              0
#define __FP64EMU_DADD_OUTPUT_INF__       0
#define __FP64EMU_DADD_USE_CARRY_BIT__    0
#define __FP64EMU_DADD_V2_EXTRA_BITS__    9
#define __FP64EMU_DADD_FP32_FAST_ENABLE__ 1

#include <cuda/__fp/fpemu_common.hpp>
#include <cuda/__fp/fpemu_impl_utils.hpp>
#include <cuda/__fp/fpemu_impl_unpack.hpp>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

namespace impl
{
    /**
     * @brief Unpacked double-precision addition for FPEMU
     *
     * This function performs double-precision addition on two unpacked FPEMU floating-point numbers.
     * It takes two fpbits64_unpacked_t structures as input, representing the unpacked sign, exponent,
     * and mantissa fields of the operands. The addition is performed according to the specified rounding mode,
     * accuracy, range, and engine template parameters.
     *
     * The function handles normalization, exponent alignment, sign management, and special cases such as
     * denormals and zeroes, producing an unpacked result in fpbits64_unpacked_t form.
     *
     * @tparam rm    Rounding mode (rounding)
     * @tparam meth  Accuracy level (fp64emu_accuracy)
     * @param x      First operand (unpacked)
     * @param y      Second operand (unpacked)
     * @return       Result of addition in unpacked form (fpbits64_unpacked_t)
     */
    template<fpemu::rounding rm     = fpemu::rounding::def, 
             fp64emu_accuracy   meth   = fp64emu_accuracy::def,
             bool     is_sub = false>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_dadd_accurate( fpbits64_t x, 
                                                    fpbits64_t y)
    {
        uint64_t a_ = x;
        uint64_t b_ = y;
        fpemu::uint32x2_t a_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(a_);
        fpemu::uint32x2_t b_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(b_);

        fpemu::uint32x2_t man_a_32x2, man_b_32x2, man_c_32x2;
        int32_t exp_a, exp_b, exp_c, shift, delta_a, delta_b, nzeros;
        bool is_sign_a, is_sign_b, is_sign_c, is_a_exp_zero, is_b_exp_zero;
        fpbits64_t result;

        //Extract exponent for input A. a_den_zero true if denorm or zero
        exp_a = fpemu::__unpack_exp<meth>(a_32x2);
        //Extract exponent for input B. a_den_zero true if denorm or zero
        exp_b = fpemu::__unpack_exp<meth>(b_32x2);

        //Check if input A is denormal or zero  
        is_a_exp_zero = (exp_a == 0);
        is_b_exp_zero = (exp_b == 0);

        //Extract mantissa for input A
        man_a_32x2 = fpemu::__unpack_mant<meth>(&is_sign_a, a_32x2, is_a_exp_zero);
        //Extract mantissa for input B
        man_b_32x2 = fpemu::__unpack_mant<meth>(&is_sign_b, b_32x2, is_b_exp_zero);

        // If subtracting, invert the sign of B
        is_sign_b ^= is_sub;

        // 2's complement for A and B if negative
        if (is_sign_a)  man_a_32x2 = fpemu::__two_comp(man_a_32x2);
        if (is_sign_b)  man_b_32x2 = fpemu::__two_comp(man_b_32x2);

        // Denormals processing
        if constexpr (meth == fp64emu_accuracy::high)
        {
            //Adjust exponent A for denorm
            if (is_a_exp_zero) exp_a = exp_a + 1;
            //Adjust exponent B for denorm
            if (is_b_exp_zero) exp_b = exp_b + 1;
        }                
        else
        {
            // Flush denormals to zero
            if (is_a_exp_zero) man_a_32x2 = {0, 0};
            if (is_b_exp_zero) man_b_32x2 = {0, 0};
        }

        //Find maximum from input exponents
        exp_c = (exp_a > exp_b) ? exp_a : exp_b;

        //Find shifts to equlize mantissa A
        delta_a = exp_c - exp_a;

        //Find shifts to equlize mantissa B
        delta_b = exp_c - exp_b;

        // Shift Mantissa A, B(2x ALU)
        // Alignment must preserve the sticky bit UNCONDITIONALLY (jam, rn-style):
        // the discarded low bits feed the single directed rounding step performed
        // later by __round<rm>. Passing rm here let rz drop the sticky entirely,
        // which truncated the aligned operand and produced a 1-ulp-too-large
        // (away-from-zero) rz result on roughly a quarter of inputs.
        man_a_32x2 = fpemu::__sar_64_rnd<meth, fpemu::rounding::rn>(man_a_32x2, delta_a);
        man_b_32x2 = fpemu::__sar_64_rnd<meth, fpemu::rounding::rn>(man_b_32x2, delta_b);

        // Add up mantissas A and B (2x IADD)
        man_c_32x2 = fpemu::__iadd_u64(man_a_32x2, man_b_32x2);

        // Check for sign of result
        is_sign_c = (man_c_32x2.x[1] & 0x80000000);

        // 2's complement for C
        if (is_sign_c) man_c_32x2 = fpemu::__two_comp(man_c_32x2);

        // Check first significant bit
        nzeros = fpemu::__flo_s64(man_c_32x2);

        // Check for exact zero result and set correct sign (IEEE-754 6.3): an
        // exact cancellation is +0 in every rounding mode except round-toward-
        // negative (rd), where it is -0 unless both addends were positive.
        if (nzeros >= 63)
        {
            if constexpr (rm == fpemu::rounding::rd)
                is_sign_c = is_sign_a || is_sign_b;
            else
                is_sign_c = is_sign_a && is_sign_b;
        }

        //Correct exponent
        exp_c = exp_c - nzeros;
        
        // Shift first significant bit to implicit-bit position
        man_c_32x2 = fpemu::__shl_64(man_c_32x2, nzeros);

        // Shift mantissa to the right
        if constexpr (meth != fp64emu_accuracy::high)
        {
            if constexpr (meth == fp64emu_accuracy::low)
            {
                // Zero out lower 32 bits of mantissa (LA)
                man_c_32x2.x[0] = man_c_32x2.x[1] << (31 - EXTRA_BITS);
                man_c_32x2.x[1] = man_c_32x2.x[1] >> (EXTRA_BITS + 1);
            }
            else
            {
                // Shift mantissa to the right with directed rounding (HA)
                man_c_32x2 = fpemu::__shr_64_rnd<rm>(man_c_32x2, EXTRA_BITS+1, is_sign_c);
            }
        }
        else
        {
            // Shift and round mantissa (CR)
            man_c_32x2 = fpemu::__round<rm>(man_c_32x2, 1, is_sign_c);
        }
                                    
        // Check for negative exponent
        bool is_exp_c_neg = (exp_c < 0);
    
        // Denormals processing
        if constexpr (meth == fp64emu_accuracy::high)
        {
            // shift is negative, so we need to add it to exp_c
            shift = -exp_c;
            exp_c = (is_exp_c_neg)?exp_c + shift:exp_c;

            // Shift mantissa to the right with rounding 
            // in case of negative exponent (DENORM)
            man_c_32x2 = (is_exp_c_neg)?
                 fpemu::__sar_64_rnd<meth, rm>(man_c_32x2, shift, is_sign_c):man_c_32x2;              
        }
        else
        {
            // Flush denormals to zero
            exp_c = (is_exp_c_neg)?0:exp_c;
            if (is_exp_c_neg) man_c_32x2 = {0, 0};
        }

        // Pack sign, exponent and mantissa back to FP64 
        // (+checks for NAN,INF,0)
        result = fpemu::__pack<meth, rm>(is_sign_c, exp_c, man_c_32x2);
        return result;
    } // __nv_internal_fp64emu_dadd_accurate

    /**
     * @brief Packed double-precision addition for FPEMU
     *
     * This function performs double-precision addition on two packed FPEMU floating-point numbers.
     * It takes two fpbits64_t structures as input, representing the packed sign, exponent,
     * and mantissa fields of the operands. The addition is performed according to the specified rounding mode,
     * accuracy, range, and engine template parameters.
     *
     * The function handles normalization, exponent alignment, sign management, and special cases such as
     * denormals and zeroes, producing a packed result in fpbits64_t form.
     *
     * @tparam rm    Rounding mode (rounding)
     * @tparam meth  Accuracy level (fp64emu_accuracy)
     * @param x      First operand (packed)
     * @param y      Second operand (packed)
     * @return       Result of addition in packed form (fpbits64_t)
     */
    template<fpemu::rounding rm     = fpemu::rounding::def, 
             fp64emu_accuracy   meth   = fp64emu_accuracy::def,
             bool     is_sub = false>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_dadd_def(fpbits64_t x, 
                                              fpbits64_t y)
    {
        fpemu::uint32x2_t a_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(x);
        fpemu::uint32x2_t b_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(y);
        constexpr int32_t extra_bits = 
            (meth == fp64emu_accuracy::low)?\
                 0:__FP64EMU_DADD_V2_EXTRA_BITS__;

        // Extract exponents by integer operations
        uint32_t exp_a = (a_32x2.x[1] >> FP64_HI_MANT_SHIFT)
                                       & FP64_LO_EXP_MASK;
        uint32_t exp_b = (b_32x2.x[1] >> FP64_HI_MANT_SHIFT)
                                       & FP64_LO_EXP_MASK;

        // Extract signs by integer operations
        uint32_t sign_a = a_32x2.x[1] & FP64_HI_SIGN_MASK;
        uint32_t sign_b = b_32x2.x[1] & FP64_HI_SIGN_MASK;

        // If subtracting, invert the sign of B
        if constexpr (is_sub) 
            { sign_b ^= (((uint32_t)(is_sub))<<31); }

        // Clear the exponent/sign bits
        a_32x2.x[1] &= FP64_HI_MANT_MASK;
        b_32x2.x[1] &= FP64_HI_MANT_MASK;

        constexpr fpemu::uint32x2_t zero_32x2 = {0, 0}; 
#if  __FP64EMU_DADD_FTZ__
        // Set the implicit 1 bit
        a_32x2.x[1] |= ( 1 << FP64_HI_MANT_SHIFT );
        b_32x2.x[1] |= ( 1 << FP64_HI_MANT_SHIFT );
        // Flush denormals to zero
        if (exp_a == 0) a_32x2 = zero_32x2;
        if (exp_b == 0) b_32x2 = zero_32x2;
#else
        // Set the implicit 1 bit
        if (exp_a != 0) a_32x2.x[1] |= ( 1 << FP64_HI_MANT_SHIFT );
        if (exp_b != 0) b_32x2.x[1] |= ( 1 << FP64_HI_MANT_SHIFT );
#endif
        // Preserve extra mantissa bits for HA accuracy
        if (meth != fp64emu_accuracy::low)
        {
            a_32x2 = fpemu::__shl_64(a_32x2, extra_bits);
            b_32x2 = fpemu::__shl_64(b_32x2, extra_bits);
        }
        // 2's complement for A and B if negative
        if (sign_a)  a_32x2 = fpemu::__two_comp(a_32x2);
        if (sign_b)  b_32x2 = fpemu::__two_comp(b_32x2);

        //Find maximum from input exponents
        int32_t exp_c = (exp_a > exp_b) ? exp_a : exp_b;
        //Find shifts to equlize mantissa A
        int32_t delta_a = exp_c - exp_a;
        //Find shifts to equlize mantissa B
        int32_t delta_b = exp_c - exp_b;

        // Shift Mantissas
        a_32x2 = fpemu::__sar_64(a_32x2, delta_a);
        b_32x2 = fpemu::__sar_64(b_32x2, delta_b);

        // Add up mantissas
        fpemu::uint32x2_t c_32x2 = fpemu::__iadd_u64(a_32x2, b_32x2);

        // Check for sign of result
        uint32_t sign_c = c_32x2.x[1] & 0x80000000;

        // 2's complement for C
        if (sign_c) c_32x2 = fpemu::__two_comp(c_32x2);

        // Check first significant bit after sign bit
        int32_t nzeros = 
            fpemu::__internal_clzll(fpemu::bit_cast<int64_t>(c_32x2));

        // Correct exponent by nzeros
        int32_t exp_corr = (nzeros - (11-1-extra_bits));
        exp_c = exp_c - exp_corr;

        // Check for negative exponent then set exponent to zero
        if ((exp_c < 0)) { exp_c = 0; c_32x2 = zero_32x2; }
        // Check for finite exponent then set exponent to zero
        if ((exp_c < 0x000007ff) && (nzeros == 64)) exp_c = 0;

        // Shift first significant bit to implicit-bit position
        c_32x2 = fpemu::__shl_64(c_32x2, exp_corr);

        const bool is_sign_c = (sign_c != 0);

// Branch by carry bit to handle exponent overflow/underflow
#if __FP64EMU_DADD_USE_CARRY_BIT__
        // Carry bit handling
        uint32_t carry_bit = 
            (c_32x2.x[1] & (1 << (20+1+extra_bits))) != 0;
        // Add carry bit to exponent
        exp_c += carry_bit;
    #if __FP64EMU_DADD_OUTPUT_INF__
        // Check for infinite result
        bool is_infinite = (exp_c >= 0x000007ff);
    #endif
        // Shift exponent to the fp64 exponent position
        exp_c <<= 20;
        // Shift mantissa to the right 
        // at the fp64 precision position
        c_32x2 = fpemu::__shr_64_rnd<rm>(c_32x2, extra_bits+1, is_sign_c);
        // Clear the unused high bits
        c_32x2.x[1] &= 0x000fffff;
        // Set exponent 
        c_32x2.x[1] |= (exp_c);
    #if __FP64EMU_DADD_OUTPUT_INF__
        // Set infinite result
        if (is_infinite ) c_32x2 = {0, 0x7ff00000};
    #endif
// Branch by addition of mantissa to exponent
#else
        const bool is_exp_ovfl = (exp_c > (FP64_BIAS*2));

        // Shift mantissa to the right at the fp64 precision position
        c_32x2 = fpemu::__shr_64_rnd<rm>(c_32x2, extra_bits+1, is_sign_c);

        if (is_exp_ovfl)
        {
            fpemu::__fp64_ovfl_sat<rm>(is_sign_c, exp_c, c_32x2);
            c_32x2.x[1] |= (uint32_t)exp_c << FP64_HI_MANT_SHIFT;
        }
        else
        {
            // Set exponent
            c_32x2.x[1] += (exp_c<<20);
    #if __FP64EMU_DADD_OUTPUT_INF__
            // Check for infinite result
            if (c_32x2.x[1] >= 0x7ff00000 ) c_32x2 = {0, 0x7ff00000};
    #endif
        }
#endif
        // Sign bit at bit 31
        c_32x2.x[1] |= sign_c;

        // Final result
        fpbits64_t result = fpemu::bit_cast<fpbits64_t>(c_32x2);
        return result;
    } // __nv_internal_fp64emu_dadd_def

    /**
     * @brief Fast double-precision addition for FPEMU
     * 
     * This function performs double-precision addition on two FPEMU floating-point numbers,
     * by single precision addition for the mantissa (fast mode).
     * It takes two fpbits64_t structures as input, representing the packed sign, exponent,
     * and mantissa fields of the operands. The addition is performed according to the specified rounding mode,
     * accuracy, range, and engine template parameters.
     */
    template<fpemu::rounding rm     = fpemu::rounding::def, 
             fp64emu_accuracy   meth   = fp64emu_accuracy::def,
             bool     is_sub = false>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_dadd_fast(fpbits64_t x, 
                                               fpbits64_t y)
    {
        fpemu::uint32x2_t a_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(x);
        fpemu::uint32x2_t b_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(y);
        fpbits64_t result;
        
        // Extract exponents by integer operations
        uint32_t exp_a = (a_32x2.x[1] >> FP64_HI_MANT_SHIFT) & FP64_LO_EXP_MASK;
        uint32_t exp_b = (b_32x2.x[1] >> FP64_HI_MANT_SHIFT) & FP64_LO_EXP_MASK;

        // Extract signs by integer operations
        uint32_t sign_a = a_32x2.x[1] & FP64_HI_SIGN_MASK;
        uint32_t sign_b = b_32x2.x[1] & FP64_HI_SIGN_MASK;

        // If subtracting, invert the sign of B
        if constexpr (is_sub) sign_b ^= (((uint32_t)(is_sub))<<31);

        // Integer operations for exponent handling
        int32_t max_exp = (exp_a > exp_b) ? exp_a : exp_b;
        int32_t exp_diff = max_exp - ((exp_a > exp_b) ? exp_b : exp_a);

        // Convert mantissas to single precision for addition
        // Extract the upper 24 bits of the 53-bit mantissa (including implicit leading 1)
        uint32_t mant_a_sp = ((a_32x2.x[0] >> FP64_MANT_TO_FP32_LO_SHIFT) | 
                              (a_32x2.x[1] << FP64_MANT_TO_FP32_HI_SHIFT)) & FP32_MANT_MASK;
        uint32_t mant_b_sp = ((b_32x2.x[0] >> FP64_MANT_TO_FP32_LO_SHIFT) | 
                              (b_32x2.x[1] << FP64_MANT_TO_FP32_HI_SHIFT)) & FP32_MANT_MASK;

        // Set unbiased exponents to 0 to scale the mantissas to [1.0, 2.0)   
        int32_t exp_a_sp = FP32_BIAS;
        int32_t exp_b_sp = FP32_BIAS;

        // Adjust exponents for mantissas
        if (exp_a < exp_b) exp_a_sp = exp_a_sp - exp_diff;
        if (exp_b < exp_a) exp_b_sp = exp_b_sp - exp_diff;

        // Flush denormals to zero
        if (exp_a_sp < 1) { exp_a_sp = 0; mant_a_sp = 0; }
        if (exp_b_sp < 1) { exp_b_sp = 0; mant_b_sp = 0; }

        // Normalize to [1.0, 2.0) by single precision construction
        mant_a_sp |= (exp_a_sp << FP32_MANT_BITS) | sign_a;
        mant_b_sp |= (exp_b_sp << FP32_MANT_BITS) | sign_b;

        // Cast to single precision floats
        // and perform single precision addition with directed rounding on device
        float mant_a_float = fpemu::bit_cast<float>(mant_a_sp);  
        float mant_b_float = fpemu::bit_cast<float>(mant_b_sp);  
        float mant_sum_float = fpemu::__fadd_dir<rm>(mant_a_float, mant_b_float);

        // Cast single precision result to integer
        int32_t mant_sum = fpemu::bit_cast<int32_t>(mant_sum_float);

        // Correct resulted exponent and subtract fp32 bias
        int32_t exp_adjust = ((mant_sum >> FP32_MANT_BITS) & FP32_LO_EXP_MASK) - FP32_BIAS;
        int32_t result_exp = max_exp + exp_adjust;

        // Extract mantissa from normalized single precision result
        uint32_t result_mant = mant_sum & FP32_MANT_MASK;

        // Determine result sign by single precision
        uint32_t result_sign = (mant_sum < 0) ? FP64_HI_SIGN_MASK : 0;
        const bool is_sign_c = (result_sign != 0);
        const bool is_exps_zero = ((exp_a == 0) && (exp_b == 0));

        // Handle exponent overflow/underflow with mode-aware saturation
        fpemu::uint32x2_t result_32x2;
        const bool is_exp_ovfl = (result_exp > (FP64_BIAS*2));

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
            result_32x2.x[0] = result_mant << FP64_MANT_TO_FP32_LO_SHIFT;
            result_32x2.x[1] = result_mant >> FP64_MANT_TO_FP32_HI_SHIFT;
        }

        // Sign bit at bit 31
        result_32x2.x[1] |= result_sign;  
        // Exponent at bits 20-30
        result_32x2.x[1] |= (uint32_t)result_exp << FP64_HI_MANT_SHIFT;  

        result = fpemu::bit_cast<uint64_t>(result_32x2);
        return result;
    }

    /**
     * @brief Pure ADD core operating on the unpacked representation (unified).
     *
     * Consumes/produces fpbits64_unpacked_t exactly as produced by the universal
     * impl::__nv_internal_fp64emu_unpack and consumed by impl::__nv_internal_fp64emu_pack
     * (normalized mantissa with the implicit bit set, inf/nan encoded in the
     * exponent band). is_sub negates b. The result is the pre-rounding
     * intermediate that pack rounds once.
     *
     * All three accuracy levels are the legacy kernels with their prologue/epilogue
     * replaced by the universal unpack/pack, then combined here. Because the
     * universal unpack/pack already do everything the legacy prologues/epilogues
     * did (extract, set implicit bit, shift to the EXTRA_BITS scale, normalize,
     * round, repack), the per-accuracy bodies collapse to just their distinct core
     * arithmetic:
     *   - high: integer add with sticky-jam alignment -> correctly
     *     rounded (rounding deferred to the universal pack);
     *   - mid: the same integer add with truncating alignment;
     *   - low: the fp32 add of the top 24 significand bits.
     * inf/nan and over/underflow all flow through the universal full-range pack.
     */
    template<fp64emu_accuracy   meth   = fp64emu_accuracy::def,
             bool     is_sub = false>
    __FPEMU_INTERNAL_DECL__
    fpbits64_unpacked_t __nv_internal_fp64emu_dadd_unpacked(fpbits64_unpacked_t a,
                                                            fpbits64_unpacked_t b)
    {
        constexpr fp64emu_accuracy   meth_forced = fp64emu_accuracy::__FPEMU_ADD_METHOD__;
        constexpr fp64emu_accuracy   meth_used   = (meth_forced != fp64emu_accuracy::unset) ? meth_forced : meth;

        // Inf/Nan exponent magics produced by the universal unpack.
        constexpr uint32_t INF_EXP = 0x00007ff0u;
        constexpr int32_t  NAN_EXP = 0x0007ff00;

        if constexpr (meth_used == fp64emu_accuracy::high ||
                      meth_used == fp64emu_accuracy::mid)
        {
            // ---- Lean 64-bit integer add core (accurate + def share this) ----
            // The legacy accurate (`dadd_accurate`) and def (`dadd_def`) kernels
            // are the same 64-bit uint32x2 add; their prologue/epilogue (extract,
            // implicit-bit, <<EXTRA_BITS, 2's-comp / round, pack) are exactly what
            // the universal unpack/pack now provide. The universal unpack delivers
            // a normalized significand with the implicit bit at position 61
            // (== 52 + EXTRA_BITS) and inf/nan in the exponent band -- the same
            // internal scale both legacy cores used -- so nothing else changes.
            //
            // The ONLY method-dependent step is the alignment shift, which is the
            // accuracy knob itself:
            //   - cr (accurate): jam the sticky bit -> correctly rounded;
            //   - ha (def):      truncate (legacy `__sar_64`) -> high accuracy.
            // The single rounding step is deferred to the universal pack (this
            // also fixes the legacy accurate round-toward-zero cancellation bug).
            int32_t exp_a = (int32_t)a.exponent;
            int32_t exp_b = (int32_t)b.exponent;
            bool is_sign_a = (a.sign != 0);
            bool is_sign_b = ((b.sign != 0) != (bool)is_sub);

            fpemu::uint32x2_t man_a = fpemu::bit_cast<fpemu::uint32x2_t>(a.mantissa);
            fpemu::uint32x2_t man_b = fpemu::bit_cast<fpemu::uint32x2_t>(b.mantissa);

            // inf +/- inf with opposite effective signs -> NaN: poison one
            // operand's exponent into the NaN band so the result lands there. Only
            // the full-range (accurate) unpack encodes the INF_EXP magic; the lean
            // def/fast unpack leaves inf at the raw 0x7ff exponent (out of contract,
            // WARN), so this check is dead there -- gate it out of the hot path.
            // Unpacked cores always run on the fully-accurate full-range
            // unpack/pack boundary, so the inf+/-inf -> NaN fold is always live.
            // Unpacked cores always run on the fully-accurate full-range boundary,
            // so the inf+/-inf -> NaN fold is always live.
            if (a.exponent == INF_EXP && b.exponent == INF_EXP && (is_sign_a != is_sign_b))
            {
                exp_a = NAN_EXP;
            }

            // Operand sign handling: negate each operand by its OWN sign, then read the
            // result sign straight off the summed bits (neg_c). This mirrors the legacy
            // fused def/fast kernels, which the compiler lowers to branchless selp on
            // both cr and ha. The earlier ha-only "sign-magnitude, 2's-complement B
            // alone" trick did one fewer negate but compiled to a data-dependent BRANCH
            // (and extended is_sign_a's live range), a net loss vs legacy -- so both
            // accuracies now use the same branchless classic form.
            if (is_sign_a) man_a = fpemu::__two_comp(man_a);
            if (is_sign_b) man_b = fpemu::__two_comp(man_b);

            int32_t exp_max = (exp_a > exp_b) ? exp_a : exp_b;
            int32_t delta_a = exp_max - exp_a;
            int32_t delta_b = exp_max - exp_b;

            if constexpr (meth_used == fp64emu_accuracy::high)
            {
                // Sticky-preserving alignment (jam) -> correctly rounded. Both
                // operands are jammed unconditionally: the larger one has delta==0
                // (a cheap no-op shift), and keeping this branchless is faster on
                // GPU than selecting which operand to shift (a data-dependent branch
                // measured *slower* despite doing one fewer shift).
                man_a = fpemu::__sar_64_rnd<fp64emu_accuracy::high, fpemu::rounding::rn>(man_a, delta_a);
                man_b = fpemu::__sar_64_rnd<fp64emu_accuracy::high, fpemu::rounding::rn>(man_b, delta_b);
            }
            else
            {
                // Truncating alignment -> legacy def (high-accuracy) behavior.
                man_a = fpemu::__sar_64(man_a, delta_a);
                man_b = fpemu::__sar_64(man_b, delta_b);
            }

            fpemu::uint32x2_t man_c = fpemu::__iadd_u64(man_a, man_b);
            const bool neg_c = (man_c.x[1] & 0x80000000u) != 0;
            if (neg_c) man_c = fpemu::__two_comp(man_c);
            // Both operands were negated by their own sign, so the summed sign bit *is*
            // the result sign.
            bool is_sign_c = neg_c;

            // Leading-one search via a plain count-leading-zeros (man_c is positive
            // after the 2's-complement, so bit 63 is clear): the leading set bit sits
            // at position (63 - nz). nz == 1 means a carry out of bit 61 (leading at
            // bit 62); nz == 64 means exact cancellation to zero. IEEE-754 6.3 would
            // make an exact-cancellation sum -0 under round-toward-negative (rd); that
            // rounding-dependent zero sign is intentionally NOT honored here (the core
            // is rounding-independent), so an exactly-zero sum is -0 only when both
            // addends are negative -- a tolerated deviation for directed modes.
            int32_t nz = fpemu::__internal_clzll(fpemu::bit_cast<int64_t>(man_c));
            if (nz >= 64) is_sign_c = (is_sign_a && is_sign_b);

            // Branchless normalization, mirroring the legacy fused def kernel
            // (clz -> single shift). Legacy lands the leading bit at position 62
            // via __shl_64(c, nzeros-1): the shift is ALWAYS >= 0 (nz in [1,64]),
            // so the carry case nz==1 is just a no-op shift -- no data-dependent
            // branch. We then drop one more bit unconditionally to reach the shared
            // pack's universal scale (implicit bit at 61). The only bit lost by that
            // >>1 is a zero-fill except in the carry case, where CR jams it back as
            // the sticky bit; HA tolerates the truncation. This replaces the
            // `if (nz==1) {...} else {...}` branch with straight-line shifts.
            uint64_t m64   = fpemu::bit_cast<uint64_t>(man_c);
            uint64_t norm  = m64 << (nz - 1);          // leading bit -> 62
            uint64_t out64 = norm >> 1;                // -> 61 (universal scale)
            if constexpr (meth_used == fp64emu_accuracy::high)
            {
                out64 |= (norm & 1ULL);                // carry-case sticky (else 0)
            }
            fpemu::uint32x2_t out = fpemu::bit_cast<fpemu::uint32x2_t>(out64);

            int32_t res_exp = exp_max - nz + 2;

            fpbits64_unpacked_t r;
            r.sign = is_sign_c ? (1u << 31) : 0u;
            // +1 matches the universal pack's "mask the implicit bit" convention
            // (pack reconstitutes via exp-1); the +1/-1 cancel.
            r.exponent = static_cast<uint32_t>(res_exp);
            r.mantissa = fpemu::bit_cast<uint64_t>(out);

            // Full-range boundary: no FTZ -- underflow flows to the pack, which
            // emits the correct subnormal for every method.
            return r;
        }
        else
        {
            // ---- Single-precision add core (fast / la) -----------------------
            // The legacy fast kernel (`dadd_fast`) adds the top 24 significand
            // bits in fp32 with directed rounding; this IS the low-accuracy core.
            // Its prologue/epilogue (extract, scale to [1,2), pack) are replaced
            // by the universal unpack/pack: the universal mantissa already carries
            // the top 23 fraction bits at positions 60..38 (implicit bit at 61),
            // and the fp32 sum is re-expressed as a universal intermediate
            // (implicit bit at 61) that the universal pack finalizes.
            (void)NAN_EXP;
            int32_t  exp_a  = (int32_t)a.exponent;
            int32_t  exp_b  = (int32_t)b.exponent;
            uint32_t sign_a = a.sign;
            uint32_t sign_b = b.sign ^ (is_sub ? FP64_HI_SIGN_MASK : 0u);

            int32_t max_exp  = (exp_a > exp_b) ? exp_a : exp_b;
            int32_t exp_diff = max_exp - ((exp_a > exp_b) ? exp_b : exp_a);

            // Top 23 fraction bits of each operand (implicit bit at 61 dropped).
            uint32_t mant_a_sp = (uint32_t)(a.mantissa >> 38) & FP32_MANT_MASK;
            uint32_t mant_b_sp = (uint32_t)(b.mantissa >> 38) & FP32_MANT_MASK;

            // Scale both significands into [1.0, 2.0) at the fp32 bias, dropping
            // the relative exponent into the smaller operand's fp32 exponent.
            int32_t exp_a_sp = FP32_BIAS;
            int32_t exp_b_sp = FP32_BIAS;
            if (exp_a < exp_b) exp_a_sp -= exp_diff;
            if (exp_b < exp_a) exp_b_sp -= exp_diff;

            // Flush zero operands and underflowed alignments to +/-0.0f.
            if (exp_a_sp < 1 || a.mantissa == 0) { exp_a_sp = 0; mant_a_sp = 0; }
            if (exp_b_sp < 1 || b.mantissa == 0) { exp_b_sp = 0; mant_b_sp = 0; }

            mant_a_sp |= ((uint32_t)exp_a_sp << FP32_MANT_BITS) | sign_a;
            mant_b_sp |= ((uint32_t)exp_b_sp << FP32_MANT_BITS) | sign_b;

            float fa   = fpemu::bit_cast<float>(mant_a_sp);
            float fb   = fpemu::bit_cast<float>(mant_b_sp);
            // ~half-mantissa result: directed fp32 rounding is meaningless, so use
            // plain round-to-nearest fp32 (final rounding is the pack's job).
            float fsum = fpemu::__fadd_dir<fpemu::rounding::rn>(fa, fb);
            int32_t msum = fpemu::bit_cast<int32_t>(fsum);

            fpbits64_unpacked_t r;
            // Exact cancellation -> signed zero (mantissa 0 makes pack emit +/-0).
            if ((msum & 0x7fffffff) == 0)
            {
                r.sign     = (msum < 0) ? (1u << 31) : 0u;
                r.exponent = 0u;
                r.mantissa = 0u;
                return r;
            }

            int32_t  exp_adjust  = ((msum >> FP32_MANT_BITS) & FP32_LO_EXP_MASK) - FP32_BIAS;
            int32_t  result_exp  = max_exp + exp_adjust;
            uint32_t result_mant = (uint32_t)msum & FP32_MANT_MASK;

            // Full-range boundary: no FTZ -- underflow flows to the pack, which
            // emits the correct subnormal for every method.

            // Re-expand the 24-bit fp32 significand (implicit bit + 23 fraction)
            // back to the universal scale with the implicit bit at position 61.
            uint64_t sig = ((uint64_t)((1u << FP32_MANT_BITS) | result_mant)) << 38;

            r.sign     = (msum < 0) ? (1u << 31) : 0u;
            // result_exp is already the result's biased exponent (the fp32
            // exp_adjust carried the binade change); pack maps exp_field == this.
            r.exponent = (uint32_t)result_exp;
            r.mantissa = sig;
            return r;
        }
    } // __nv_internal_fp64emu_dadd_unpacked

    /**
     * @brief Double-precision addition for FPEMU by native operations
    *
     * This function performs double-precision addition on two FPEMU floating-point numbers.
     * It takes two fpbits64_t structures as input, representing the packed sign, exponent,
     * and mantissa fields of the operands. The addition is performed according to the specified rounding mode,
     * accuracy, range, and engine template parameters.
     *
     * The function handles normalization, exponent alignment, sign management, and special cases such as
     * This function performs double-precision addition on two FPEMU floating-point numbers.
     * It takes two fpbits64_t structures as input, representing the packed sign, exponent,
     * and mantissa fields of the operands. The addition is performed according to the specified rounding mode,
     * accuracy, range, and engine template parameters.
     *
     * The function handles normalization, exponent alignment, sign management, and special cases such as
     * denormals and zeroes, producing a result in fpbits64_t form.
     *
     * @tparam rm    Rounding mode (rounding)
     * @tparam meth  Accuracy level (fp64emu_accuracy)
     * @param x      First operand (packed)
     * @param y      Second operand (packed)
     * @return       Result of addition in packed form (fpbits64_t)
     */
    template<fpemu::rounding rm   = fpemu::rounding::def, 
             fp64emu_accuracy   meth = fp64emu_accuracy::def,
             bool        is_sub = false>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_dadd(fpbits64_t x, 
                                        fpbits64_t y)
    {
        // Forced parameters for the addition operation
        constexpr fp64emu_accuracy   meth_forced = fp64emu_accuracy::__FPEMU_ADD_METHOD__;
        constexpr fp64emu_accuracy   meth_used   = (meth_forced != fp64emu_accuracy::unset) ? meth_forced : meth;
        {
    #if (__FPEMU_PACKED_VIA_UNPACKED__ == 1)
            {
                // Packed-via-unpacked (testing): pack(dadd_unpacked(unpack(x), unpack(y))). The
                // dadd_unpacked core selects accurate/def/fast internally; the
                // universal unpack/pack are the shared prologue/epilogue.
                fpbits64_unpacked_t a = __nv_internal_fp64emu_unpack(x);
                fpbits64_unpacked_t b = __nv_internal_fp64emu_unpack(y);
                fpbits64_unpacked_t r = __nv_internal_fp64emu_dadd_unpacked<meth_used, is_sub>(a, b);
                return __nv_internal_fp64emu_pack<rm>(r);
            }
    #else
            if constexpr (meth_used == fp64emu_accuracy::high)
            {
                return __nv_internal_fp64emu_dadd_accurate<rm, meth_used, is_sub>(x, y);
            }
            else if constexpr (meth_used == fp64emu_accuracy::mid)
            {
                return __nv_internal_fp64emu_dadd_def<rm, meth_used, is_sub>(x, y);
            }
            else if constexpr (meth_used == fp64emu_accuracy::low)
            {
    #if __FP64EMU_DADD_FP32_FAST_ENABLE__ == 1
                return __nv_internal_fp64emu_dadd_fast<rm, meth_used, is_sub>(x, y);
    #else
                return __nv_internal_fp64emu_dadd_def<rm, meth_used, is_sub>(x, y);
    #endif
            }
            else
            {
                return __nv_internal_fp64emu_dadd_def<rm, meth_used, is_sub>(x, y);
            }
    #endif
        }
    } // __nv_internal_fp64emu_dadd
} // namespace impl

// ============================================================================
// Builtin declarations/implementations for addition operations
// ============================================================================
#if defined(__FPEMU_INLINE__)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dadd_rn (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dadd<fpemu::rounding::rn, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dadd_rz (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dadd<fpemu::rounding::rz, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dadd_ru (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dadd<fpemu::rounding::ru, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dadd_rd (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dadd<fpemu::rounding::rd, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_dadd_rn (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dadd<fpemu::rounding::rn, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dadd_rn      (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dadd<fpemu::rounding::rn, fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dadd_rz      (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dadd<fpemu::rounding::rz, fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dadd_ru      (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dadd<fpemu::rounding::ru, fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dadd_rd      (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dadd<fpemu::rounding::rd, fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dadd_rn     (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dadd<fpemu::rounding::rn, fp64emu_accuracy::low>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dadd_rz     (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dadd<fpemu::rounding::rz, fp64emu_accuracy::low>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dadd_ru     (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dadd<fpemu::rounding::ru, fp64emu_accuracy::low>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dadd_rd     (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dadd<fpemu::rounding::rd, fp64emu_accuracy::low>(x, y); }
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dadd          (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_dadd_unpacked<fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dadd (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_dadd_unpacked<fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dadd      (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_dadd_unpacked<fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dadd     (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_dadd_unpacked<fp64emu_accuracy::low>(x, y); }
#endif
#else
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dadd_rn (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dadd_rz (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dadd_ru (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dadd_rd (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_dadd_rn (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dadd_rn      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dadd_rz      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dadd_ru      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dadd_rd      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dadd_rn     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dadd_rz     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dadd_ru     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dadd_rd     (fpbits64_t x, fpbits64_t y);
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dadd          (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dadd (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dadd      (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dadd     (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
#endif
#endif // __FPEMU_INLINE__

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_ADD_HPP__ (builtins)

#if defined(__FPEMU_API_CLASSES_DEFINED__) && !defined(__FPEMU_DADD_API_MERGED__)
#define __FPEMU_DADD_API_MERGED__
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

 


// ============================================================================
// API (merged from fp64emu_dadd_api.hpp)
// ============================================================================
    // Default API implementation
    template<fp64emu_accuracy m>  
    __FPEMU_HOST_DEVICE_DECL__ static fp64emu_t<m> operator+ (const fp64emu_t<m>& x, 
                                                                     const fp64emu_t<m>& y)
    {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_high_dadd_rn(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dadd_rn(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dadd_rn(x.bits, y.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dadd_rn(x.bits, y.bits)); }
    } // operator+


    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dadd_rn (const fp64emu_t<m>& x, const fp64emu_t<m>& y) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_high_dadd_rn(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dadd_rn(x.bits, y.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dadd_rn(x.bits, y.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dadd_rz (const fp64emu_t<m>& x, const fp64emu_t<m>& y) {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dadd_rz(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dadd_rz(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dadd_rz(x.bits, y.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dadd_rz(x.bits, y.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dadd_ru (const fp64emu_t<m>& x, const fp64emu_t<m>& y) {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dadd_ru(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dadd_ru(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dadd_ru(x.bits, y.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dadd_ru(x.bits, y.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dadd_rd (const fp64emu_t<m>& x, const fp64emu_t<m>& y) {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dadd_rd(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dadd_rd(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dadd_rd(x.bits, y.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dadd_rd(x.bits, y.bits)); }
    }

#if __FPEMU_UNPACKED__ == 1

    // Operator+ for unpacked addition
    template<fp64emu_accuracy m>  
    __FPEMU_DEVICE_DECL__ static fp64emu_unpacked_t<m> operator+ (const fp64emu_unpacked_t<m>& x, 
                                                                            const fp64emu_unpacked_t<m>& y)
    {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_high_dadd(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_mid_dadd(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_low_dadd(x.bits, y.bits)); }
        else                                             { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_dadd(x.bits, y.bits)); }
    } // operator+


    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<m> __dadd_rn (const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_high_dadd(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_low_dadd(x.bits, y.bits)); }
        else                                             { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_mid_dadd(x.bits, y.bits)); }
    }

#endif // __FPEMU_UNPACKED__ == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_ADD_HPP__
