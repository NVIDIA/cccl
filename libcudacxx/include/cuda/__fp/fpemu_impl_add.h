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
 * - Addition functions for fpemu
 * - Addition operators for fpemu
 * - Addition functions to other types
 *
 * The addition functions are designed to work across both host and device code
 * through appropriate decorators and provide bit-exact results matching hardware
 * floating point units.
 */

//#define __FP64EMU_ADD_OUTPUT_INF__         0
//#define __FP64EMU_ADD_USE_CARRY_BIT__      0
#define _CCCL_FP64EMU_DADD_FTZ              0
#define _CCCL_FP64EMU_DADD_OUTPUT_INF       0
#define _CCCL_FP64EMU_DADD_USE_CARRY_BIT    0
#define _CCCL_FP64EMU_DADD_V2_EXTRA_BITS    9
#define _CCCL_FP64EMU_DADD_FP32_FAST_ENABLE 1

#include <cuda/__fp/fpemu_impl.h>
#include <cuda/__fp/fpemu_impl_unpack.h>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


    /**
     * @brief Unpacked double-precision addition for FPEMU
     *
     * This function performs double-precision addition on two unpacked FPEMU floating-point numbers.
     * It takes two __fpbits64_unpacked structures as input, representing the unpacked sign, exponent,
     * and mantissa fields of the operands. The addition is performed according to the specified rounding mode,
     * accuracy, range, and engine template parameters.
     *
     * The function handles normalization, exponent alignment, sign management, and special cases such as
     * denormals and zeroes, producing an unpacked result in __fpbits64_unpacked form.
     *
     * @tparam rm    Rounding mode (rounding)
     * @tparam _Acc  Accuracy level (fpemu_accuracy)
     * @param x      First operand (unpacked)
     * @param y      Second operand (unpacked)
     * @return       Result of addition in unpacked form (__fpbits64_unpacked)
     */
    template<__fpemu_rounding    _Rm    = __fpemu_rounding::def, 
             fpemu_accuracy      _Acc   = fpemu_accuracy::def,
             bool                _IsSub = false>
    _CCCL_TRIVIAL_API
    __fpbits64 __internal_fp64emu_dadd_accurate( __fpbits64 __x, 
                                                 __fpbits64 __y) noexcept
    {
        uint64_t __a = __x;
        uint64_t __b = __y;
        __uint32x2 __a_32x2 = __fpemu_bit_cast<__uint32x2>(__a);
        __uint32x2 __b_32x2 = __fpemu_bit_cast<__uint32x2>(__b);

        __uint32x2 __man_a_32x2, __man_b_32x2, __man_c_32x2;
        int32_t __exp_a, __exp_b, __exp_c, __shift, __delta_a, __delta_b, __nzeros;
        bool __is_sign_a, __is_sign_b, __is_sign_c, __is_a_exp_zero, __is_b_exp_zero;
        __fpbits64 __result;

        //Extract exponent for input A. a_den_zero true if denorm or zero
        __exp_a = __unpack_exp<_Acc>(__a_32x2);
        //Extract exponent for input B. a_den_zero true if denorm or zero
        __exp_b = __unpack_exp<_Acc>(__b_32x2);

        //Check if input A is denormal or zero  
        __is_a_exp_zero = (__exp_a == 0);
        __is_b_exp_zero = (__exp_b == 0);

        //Extract mantissa for input A
        __man_a_32x2 = __unpack_mant<_Acc>(&__is_sign_a, __a_32x2, __is_a_exp_zero);
        //Extract mantissa for input B
        __man_b_32x2 = __unpack_mant<_Acc>(&__is_sign_b, __b_32x2, __is_b_exp_zero);

        // If subtracting, invert the sign of B
        __is_sign_b ^= _IsSub;

        // 2's complement for A and B if negative
        if (__is_sign_a)  __man_a_32x2 = __two_comp(__man_a_32x2);
        if (__is_sign_b)  __man_b_32x2 = __two_comp(__man_b_32x2);

        // Denormals processing
        if constexpr (_Acc == fpemu_accuracy::high)
        {
            //Adjust exponent A for denorm
            if (__is_a_exp_zero) __exp_a = __exp_a + 1;
            //Adjust exponent B for denorm
            if (__is_b_exp_zero) __exp_b = __exp_b + 1;
        }                
        else
        {
            // Flush denormals to zero
            if (__is_a_exp_zero) __man_a_32x2 = {0, 0};
            if (__is_b_exp_zero) __man_b_32x2 = {0, 0};
        }

        //Find maximum from input exponents
        __exp_c = (__exp_a > __exp_b) ? __exp_a : __exp_b;

        //Find shifts to equlize mantissa A
        __delta_a = __exp_c - __exp_a;

        //Find shifts to equlize mantissa B
        __delta_b = __exp_c - __exp_b;

        // Shift Mantissa A, B(2x ALU)
        // Alignment must preserve the sticky bit UNCONDITIONALLY (jam, rn-style):
        // the discarded low bits feed the single directed rounding step performed
        // later by __round<rm>. Passing rm here let rz drop the sticky entirely,
        // which truncated the aligned operand and produced a 1-ulp-too-large
        // (away-from-zero) rz result on roughly a quarter of inputs.
        __man_a_32x2 = __sar_64_rnd<_Acc, __fpemu_rounding::rn>(__man_a_32x2, __delta_a);
        __man_b_32x2 = __sar_64_rnd<_Acc, __fpemu_rounding::rn>(__man_b_32x2, __delta_b);

        // Add up mantissas A and B (2x IADD)
        __man_c_32x2 = __iadd_u64(__man_a_32x2, __man_b_32x2);

        // Check for sign of result
        __is_sign_c = (__man_c_32x2.x[1] & 0x80000000);

        // 2's complement for C
        if (__is_sign_c) __man_c_32x2 = __two_comp(__man_c_32x2);

        // Check first significant bit
        __nzeros = __flo_s64(__man_c_32x2);

        // Check for exact zero result and set correct sign (IEEE-754 6.3): an
        // exact cancellation is +0 in every rounding mode except round-toward-
        // negative (rd), where it is -0 unless both addends were positive.
        if (__nzeros >= 63)
        {
            if constexpr (_Rm == __fpemu_rounding::rd)
                __is_sign_c = __is_sign_a || __is_sign_b;
            else
                __is_sign_c = __is_sign_a && __is_sign_b;
        }

        //Correct exponent
        __exp_c = __exp_c - __nzeros;
        
        // Shift first significant bit to implicit-bit position
        __man_c_32x2 = __shl_64(__man_c_32x2, __nzeros);

        // Shift mantissa to the right
        if constexpr (_Acc != fpemu_accuracy::high)
        {
            if constexpr (_Acc == fpemu_accuracy::low)
            {
                // Zero out lower 32 bits of mantissa (LA)
                __man_c_32x2.x[0] = __man_c_32x2.x[1] << (31 - EXTRA_BITS);
                __man_c_32x2.x[1] = __man_c_32x2.x[1] >> (EXTRA_BITS + 1);
            }
            else
            {
                // Shift mantissa to the right with directed rounding (HA)
                __man_c_32x2 = __shr_64_rnd<_Rm>(__man_c_32x2, EXTRA_BITS+1, __is_sign_c);
            }
        }
        else
        {
            // Shift and round mantissa (CR)
            __man_c_32x2 = __round<_Rm>(__man_c_32x2, 1, __is_sign_c);
        }
                                    
        // Check for negative exponent
        bool __is_exp_c_neg = (__exp_c < 0);
    
        // Denormals processing
        if constexpr (_Acc == fpemu_accuracy::high)
        {
            // shift is negative, so we need to add it to exp_c
            __shift = -__exp_c;
            __exp_c = (__is_exp_c_neg)?__exp_c + __shift:__exp_c;

            // Shift mantissa to the right with rounding 
            // in case of negative exponent (DENORM)
            __man_c_32x2 = (__is_exp_c_neg)?
                 __sar_64_rnd<_Acc, _Rm>(__man_c_32x2, __shift, __is_sign_c):__man_c_32x2;              
        }
        else
        {
            // Flush denormals to zero
            __exp_c = (__is_exp_c_neg)?0:__exp_c;
            if (__is_exp_c_neg) __man_c_32x2 = {0, 0};
        }

        // Pack sign, exponent and mantissa back to FP64 
        // (+checks for NAN,INF,0)
        __result = __pack<_Acc, _Rm>(__is_sign_c, __exp_c, __man_c_32x2);
        return __result;
    } // __internal_fp64emu_dadd_accurate

    /**
     * @brief Packed double-precision addition for FPEMU
     *
     * This function performs double-precision addition on two packed FPEMU floating-point numbers.
     * It takes two __fpbits64 structures as input, representing the packed sign, exponent,
     * and mantissa fields of the operands. The addition is performed according to the specified rounding mode,
     * accuracy, range, and engine template parameters.
     *
     * The function handles normalization, exponent alignment, sign management, and special cases such as
     * denormals and zeroes, producing a packed result in __fpbits64 form.
     *
     * @tparam rm    Rounding mode (rounding)
     * @tparam _Acc  Accuracy level (fpemu_accuracy)
     * @param x      First operand (packed)
     * @param y      Second operand (packed)
     * @return       Result of addition in packed form (__fpbits64)
     */
    template<__fpemu_rounding    _Rm    = __fpemu_rounding::def, 
             fpemu_accuracy      _Acc   = fpemu_accuracy::def,
             bool                _IsSub = false>
    _CCCL_TRIVIAL_API
    __fpbits64 __internal_fp64emu_dadd_def(__fpbits64 __x, 
                                           __fpbits64 __y) noexcept
    {
        __uint32x2 __a_32x2 = __fpemu_bit_cast<__uint32x2>(__x);
        __uint32x2 __b_32x2 = __fpemu_bit_cast<__uint32x2>(__y);
        constexpr int32_t __extra_bits = 
            (_Acc == fpemu_accuracy::low)?\
                 0:_CCCL_FP64EMU_DADD_V2_EXTRA_BITS;

        // Extract exponents by integer operations
        uint32_t __exp_a = (__a_32x2.x[1] >> _CCCL_FP64_HI_MANT_SHIFT)
                                       & _CCCL_FP64_LO_EXP_MASK;
        uint32_t __exp_b = (__b_32x2.x[1] >> _CCCL_FP64_HI_MANT_SHIFT)
                                       & _CCCL_FP64_LO_EXP_MASK;

        // Extract signs by integer operations
        uint32_t __sign_a = __a_32x2.x[1] & _CCCL_FP64_HI_SIGN_MASK;
        uint32_t __sign_b = __b_32x2.x[1] & _CCCL_FP64_HI_SIGN_MASK;

        // If subtracting, invert the sign of B
        if constexpr (_IsSub) 
            { __sign_b ^= (((uint32_t)(_IsSub))<<31); }

        // Clear the exponent/sign bits
        __a_32x2.x[1] &= _CCCL_FP64_HI_MANT_MASK;
        __b_32x2.x[1] &= _CCCL_FP64_HI_MANT_MASK;

        constexpr __uint32x2 __zero_32x2 = {0, 0}; 
#if  _CCCL_FP64EMU_DADD_FTZ
        // Set the implicit 1 bit
        a_32x2.x[1] |= ( 1 << _CCCL_FP64_HI_MANT_SHIFT );
        b_32x2.x[1] |= ( 1 << _CCCL_FP64_HI_MANT_SHIFT );
        // Flush denormals to zero
        if (exp_a == 0) a_32x2 = zero_32x2;
        if (exp_b == 0) b_32x2 = zero_32x2;
#else
        // Set the implicit 1 bit
        if (__exp_a != 0) __a_32x2.x[1] |= ( 1 << _CCCL_FP64_HI_MANT_SHIFT );
        if (__exp_b != 0) __b_32x2.x[1] |= ( 1 << _CCCL_FP64_HI_MANT_SHIFT );
#endif
        // Preserve extra mantissa bits for HA accuracy
        if (_Acc != fpemu_accuracy::low)
        {
            __a_32x2 = __shl_64(__a_32x2, __extra_bits);
            __b_32x2 = __shl_64(__b_32x2, __extra_bits);
        }
        // 2's complement for A and B if negative
        if (__sign_a)  __a_32x2 = __two_comp(__a_32x2);
        if (__sign_b)  __b_32x2 = __two_comp(__b_32x2);

        //Find maximum from input exponents
        int32_t __exp_c = (__exp_a > __exp_b) ? __exp_a : __exp_b;
        //Find shifts to equlize mantissa A
        int32_t __delta_a = __exp_c - __exp_a;
        //Find shifts to equlize mantissa B
        int32_t __delta_b = __exp_c - __exp_b;

        // Shift Mantissas
        __a_32x2 = __sar_64(__a_32x2, __delta_a);
        __b_32x2 = __sar_64(__b_32x2, __delta_b);

        // Add up mantissas
        __uint32x2 __c_32x2 = __iadd_u64(__a_32x2, __b_32x2);

        // Check for sign of result
        uint32_t __sign_c = __c_32x2.x[1] & 0x80000000;

        // 2's complement for C
        if (__sign_c) __c_32x2 = __two_comp(__c_32x2);

        // Check first significant bit after sign bit
        int32_t __nzeros = 
            __internal_clzll(__fpemu_bit_cast<int64_t>(__c_32x2));

        // Correct exponent by nzeros
        int32_t __exp_corr = (__nzeros - (11-1-__extra_bits));
        __exp_c = __exp_c - __exp_corr;

        // Check for negative exponent then set exponent to zero
        if ((__exp_c < 0)) { __exp_c = 0; __c_32x2 = __zero_32x2; }
        // Check for finite exponent then set exponent to zero
        if ((__exp_c < 0x000007ff) && (__nzeros == 64)) __exp_c = 0;

        // Shift first significant bit to implicit-bit position
        __c_32x2 = __shl_64(__c_32x2, __exp_corr);

        const bool __is_sign_c = (__sign_c != 0);

// Branch by carry bit to handle exponent overflow/underflow
#if _CCCL_FP64EMU_DADD_USE_CARRY_BIT
        // Carry bit handling
        uint32_t __carry_bit = 
            (__c_32x2.x[1] & (1 << (20+1+__extra_bits))) != 0;
        // Add carry bit to exponent
        __exp_c += __carry_bit;
    #if _CCCL_FP64EMU_DADD_OUTPUT_INF
        // Check for infinite result
        bool __is_infinite = (__exp_c >= 0x000007ff);
    #endif
        // Shift exponent to the fp64 exponent position
        __exp_c <<= 20;
        // Shift mantissa to the right 
        // at the fp64 precision position
        __c_32x2 = __shr_64_rnd<_Rm>(__c_32x2, __extra_bits+1, __is_sign_c);
        // Clear the unused high bits
        __c_32x2.x[1] &= 0x000fffff;
        // Set exponent 
        __c_32x2.x[1] |= (__exp_c);
    #if _CCCL_FP64EMU_DADD_OUTPUT_INF
        // Set infinite result
        if (__is_infinite ) __c_32x2 = {0, 0x7ff00000};
    #endif
// Branch by addition of mantissa to exponent
#else
        const bool __is_exp_ovfl = (__exp_c > (_CCCL_FP64_BIAS*2));

        // Shift mantissa to the right at the fp64 precision position
        __c_32x2 = __shr_64_rnd<_Rm>(__c_32x2, __extra_bits+1, __is_sign_c);

        if (__is_exp_ovfl)
        {
            __fp64_ovfl_sat<_Rm>(__is_sign_c, __exp_c, __c_32x2);
            __c_32x2.x[1] |= (uint32_t)__exp_c << _CCCL_FP64_HI_MANT_SHIFT;
        }
        else
        {
            // Set exponent
            __c_32x2.x[1] += (__exp_c<<20);
    #if _CCCL_FP64EMU_DADD_OUTPUT_INF
            // Check for infinite result
            if (c_32x2.x[1] >= 0x7ff00000 ) c_32x2 = {0, 0x7ff00000};
    #endif
        }
#endif
        // Sign bit at bit 31
        __c_32x2.x[1] |= __sign_c;

        // Final result
        __fpbits64 __result = __fpemu_bit_cast<__fpbits64>(__c_32x2);
        return __result;
    } // __internal_fp64emu_dadd_def

    /**
     * @brief Fast double-precision addition for FPEMU
     * 
     * This function performs double-precision addition on two FPEMU floating-point numbers,
     * by single precision addition for the mantissa (fast mode).
     * It takes two __fpbits64 structures as input, representing the packed sign, exponent,
     * and mantissa fields of the operands. The addition is performed according to the specified rounding mode,
     * accuracy, range, and engine template parameters.
     */
    template<__fpemu_rounding    _Rm    = __fpemu_rounding::def, 
             fpemu_accuracy      _Acc   = fpemu_accuracy::def,
             bool                _IsSub = false>
    _CCCL_TRIVIAL_API
    __fpbits64 __internal_fp64emu_dadd_fast(__fpbits64 __x, 
                                            __fpbits64 __y) noexcept
    {
        __uint32x2 __a_32x2 = __fpemu_bit_cast<__uint32x2>(__x);
        __uint32x2 __b_32x2 = __fpemu_bit_cast<__uint32x2>(__y);
        __fpbits64 __result;
        
        // Extract exponents by integer operations
        uint32_t __exp_a = (__a_32x2.x[1] >> _CCCL_FP64_HI_MANT_SHIFT) & _CCCL_FP64_LO_EXP_MASK;
        uint32_t __exp_b = (__b_32x2.x[1] >> _CCCL_FP64_HI_MANT_SHIFT) & _CCCL_FP64_LO_EXP_MASK;

        // Extract signs by integer operations
        uint32_t __sign_a = __a_32x2.x[1] & _CCCL_FP64_HI_SIGN_MASK;
        uint32_t __sign_b = __b_32x2.x[1] & _CCCL_FP64_HI_SIGN_MASK;

        // If subtracting, invert the sign of B
        if constexpr (_IsSub) __sign_b ^= (((uint32_t)(_IsSub))<<31);

        // Integer operations for exponent handling
        int32_t __max_exp = (__exp_a > __exp_b) ? __exp_a : __exp_b;
        int32_t __exp_diff = __max_exp - ((__exp_a > __exp_b) ? __exp_b : __exp_a);

        // Convert mantissas to single precision for addition
        // Extract the upper 24 bits of the 53-bit mantissa (including implicit leading 1)
        uint32_t __mant_a_sp = ((__a_32x2.x[0] >> _CCCL_FP64_MANT_TO_FP32_LO_SHIFT) | 
                              (__a_32x2.x[1] << _CCCL_FP64_MANT_TO_FP32_HI_SHIFT)) & _CCCL_FP32_MANT_MASK;
        uint32_t __mant_b_sp = ((__b_32x2.x[0] >> _CCCL_FP64_MANT_TO_FP32_LO_SHIFT) | 
                              (__b_32x2.x[1] << _CCCL_FP64_MANT_TO_FP32_HI_SHIFT)) & _CCCL_FP32_MANT_MASK;

        // Set unbiased exponents to 0 to scale the mantissas to [1.0, 2.0)   
        int32_t __exp_a_sp = _CCCL_FP32_BIAS;
        int32_t __exp_b_sp = _CCCL_FP32_BIAS;

        // Adjust exponents for mantissas
        if (__exp_a < __exp_b) __exp_a_sp = __exp_a_sp - __exp_diff;
        if (__exp_b < __exp_a) __exp_b_sp = __exp_b_sp - __exp_diff;

        // Flush denormals to zero
        if (__exp_a_sp < 1) { __exp_a_sp = 0; __mant_a_sp = 0; }
        if (__exp_b_sp < 1) { __exp_b_sp = 0; __mant_b_sp = 0; }

        // Normalize to [1.0, 2.0) by single precision construction
        __mant_a_sp |= (__exp_a_sp << _CCCL_FP32_MANT_BITS) | __sign_a;
        __mant_b_sp |= (__exp_b_sp << _CCCL_FP32_MANT_BITS) | __sign_b;

        // Cast to single precision floats
        // and perform single precision addition with directed rounding on device
        float __mant_a_float = __fpemu_bit_cast<float>(__mant_a_sp);  
        float __mant_b_float = __fpemu_bit_cast<float>(__mant_b_sp);  
        float __mant_sum_float = __fadd_dir<_Rm>(__mant_a_float, __mant_b_float);

        // Cast single precision result to integer
        int32_t __mant_sum = __fpemu_bit_cast<int32_t>(__mant_sum_float);

        // Correct resulted exponent and subtract fp32 bias
        int32_t __exp_adjust = ((__mant_sum >> _CCCL_FP32_MANT_BITS) & _CCCL_FP32_LO_EXP_MASK) - _CCCL_FP32_BIAS;
        int32_t __result_exp = __max_exp + __exp_adjust;

        // Extract mantissa from normalized single precision result
        uint32_t __result_mant = __mant_sum & _CCCL_FP32_MANT_MASK;

        // Determine result sign by single precision
        uint32_t __result_sign = (__mant_sum < 0) ? _CCCL_FP64_HI_SIGN_MASK : 0;
        const bool __is_sign_c = (__result_sign != 0);
        const bool __is_exps_zero = ((__exp_a == 0) && (__exp_b == 0));

        // Handle exponent overflow/underflow with mode-aware saturation
        __uint32x2 __result_32x2;
        const bool __is_exp_ovfl = (__result_exp > (_CCCL_FP64_BIAS*2));

        if (__is_exps_zero)
        {
            __result_exp = 0;
            __result_32x2 = {0, 0};
        }
        else if (__is_exp_ovfl)
        {
            __fp64_ovfl_sat<_Rm>(__is_sign_c, __result_exp, __result_32x2);
        }
        else
        {
            if (__result_exp < 0) __result_exp = 0;
            __result_32x2.x[0] = __result_mant << _CCCL_FP64_MANT_TO_FP32_LO_SHIFT;
            __result_32x2.x[1] = __result_mant >> _CCCL_FP64_MANT_TO_FP32_HI_SHIFT;
        }

        // Sign bit at bit 31
        __result_32x2.x[1] |= __result_sign;  
        // Exponent at bits 20-30
        __result_32x2.x[1] |= (uint32_t)__result_exp << _CCCL_FP64_HI_MANT_SHIFT;  

        __result = __fpemu_bit_cast<uint64_t>(__result_32x2);
        return __result;
    }

    /**
     * @brief Pure ADD core operating on the unpacked representation (unified).
     *
     * Consumes/produces __fpbits64_unpacked exactly as produced by the universal
     * __internal_fp64emu_unpack and consumed by __internal_fp64emu_pack
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
    template<fpemu_accuracy   _Acc   = fpemu_accuracy::def,
             bool               _IsSub = false>
    _CCCL_TRIVIAL_API
    __fpbits64_unpacked __internal_fp64emu_dadd_unpacked(__fpbits64_unpacked __a,
                                                         __fpbits64_unpacked __b) noexcept
    {
        constexpr fpemu_accuracy   __acc_forced = fpemu_accuracy::_CCCL_FPEMU_ADD_METHOD;
        constexpr fpemu_accuracy   __acc_used   = (__acc_forced != fpemu_accuracy::unset) ? __acc_forced : _Acc;

        // Inf/Nan exponent magics produced by the universal unpack.
        constexpr uint32_t __INF_EXP = 0x00007ff0u;
        constexpr int32_t  __NAN_EXP = 0x0007ff00;

        if constexpr (__acc_used == fpemu_accuracy::high ||
                      __acc_used == fpemu_accuracy::mid)
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
            int32_t __exp_a = (int32_t)__a.exponent;
            int32_t __exp_b = (int32_t)__b.exponent;
            bool __is_sign_a = (__a.sign != 0);
            bool __is_sign_b = ((__b.sign != 0) != (bool)_IsSub);

            __uint32x2 __man_a = __fpemu_bit_cast<__uint32x2>(__a.mantissa);
            __uint32x2 __man_b = __fpemu_bit_cast<__uint32x2>(__b.mantissa);

            // inf +/- inf with opposite effective signs -> NaN: poison one
            // operand's exponent into the NaN band so the result lands there. Only
            // the full-range (accurate) unpack encodes the INF_EXP magic; the lean
            // def/fast unpack leaves inf at the raw 0x7ff exponent (out of contract,
            // WARN), so this check is dead there -- gate it out of the hot path.
            // Unpacked cores always run on the fully-accurate full-range
            // unpack/pack boundary, so the inf+/-inf -> NaN fold is always live.
            // Unpacked cores always run on the fully-accurate full-range boundary,
            // so the inf+/-inf -> NaN fold is always live.
            if (__a.exponent == __INF_EXP && __b.exponent == __INF_EXP && (__is_sign_a != __is_sign_b))
            {
                __exp_a = __NAN_EXP;
            }

            // Operand sign handling: negate each operand by its OWN sign, then read the
            // result sign straight off the summed bits (neg_c). This mirrors the legacy
            // fused def/fast kernels, which the compiler lowers to branchless selp on
            // both cr and ha. The earlier ha-only "sign-magnitude, 2's-complement B
            // alone" trick did one fewer negate but compiled to a data-dependent BRANCH
            // (and extended is_sign_a's live range), a net loss vs legacy -- so both
            // accuracies now use the same branchless classic form.
            if (__is_sign_a) __man_a = __two_comp(__man_a);
            if (__is_sign_b) __man_b = __two_comp(__man_b);

            int32_t __exp_max = (__exp_a > __exp_b) ? __exp_a : __exp_b;
            int32_t __delta_a = __exp_max - __exp_a;
            int32_t __delta_b = __exp_max - __exp_b;

            if constexpr (__acc_used == fpemu_accuracy::high)
            {
                // Sticky-preserving alignment (jam) -> correctly rounded. Both
                // operands are jammed unconditionally: the larger one has delta==0
                // (a cheap no-op shift), and keeping this branchless is faster on
                // GPU than selecting which operand to shift (a data-dependent branch
                // measured *slower* despite doing one fewer shift).
                __man_a = __sar_64_rnd<fpemu_accuracy::high, __fpemu_rounding::rn>(__man_a, __delta_a);
                __man_b = __sar_64_rnd<fpemu_accuracy::high, __fpemu_rounding::rn>(__man_b, __delta_b);
            }
            else
            {
                // Truncating alignment -> legacy def (high-accuracy) behavior.
                __man_a = __sar_64(__man_a, __delta_a);
                __man_b = __sar_64(__man_b, __delta_b);
            }

            __uint32x2 __man_c = __iadd_u64(__man_a, __man_b);
            const bool __neg_c = (__man_c.x[1] & 0x80000000u) != 0;
            if (__neg_c) __man_c = __two_comp(__man_c);
            // Both operands were negated by their own sign, so the summed sign bit *is*
            // the result sign.
            bool __is_sign_c = __neg_c;

            // Leading-one search via a plain count-leading-zeros (man_c is positive
            // after the 2's-complement, so bit 63 is clear): the leading set bit sits
            // at position (63 - nz). nz == 1 means a carry out of bit 61 (leading at
            // bit 62); nz == 64 means exact cancellation to zero. IEEE-754 6.3 would
            // make an exact-cancellation sum -0 under round-toward-negative (rd); that
            // rounding-dependent zero sign is intentionally NOT honored here (the core
            // is rounding-independent), so an exactly-zero sum is -0 only when both
            // addends are negative -- a tolerated deviation for directed modes.
            int32_t __nz = __internal_clzll(__fpemu_bit_cast<int64_t>(__man_c));
            if (__nz >= 64) __is_sign_c = (__is_sign_a && __is_sign_b);

            // Branchless normalization, mirroring the legacy fused def kernel
            // (clz -> single shift). Legacy lands the leading bit at position 62
            // via __shl_64(c, nzeros-1): the shift is ALWAYS >= 0 (nz in [1,64]),
            // so the carry case nz==1 is just a no-op shift -- no data-dependent
            // branch. We then drop one more bit unconditionally to reach the shared
            // pack's universal scale (implicit bit at 61). The only bit lost by that
            // >>1 is a zero-fill except in the carry case, where CR jams it back as
            // the sticky bit; HA tolerates the truncation. This replaces the
            // `if (nz==1) {...} else {...}` branch with straight-line shifts.
            uint64_t __m64   = __fpemu_bit_cast<uint64_t>(__man_c);
            uint64_t __norm  = __m64 << (__nz - 1);          // leading bit -> 62
            uint64_t __out64 = __norm >> 1;                // -> 61 (universal scale)
            if constexpr (__acc_used == fpemu_accuracy::high)
            {
                __out64 |= (__norm & 1ULL);                // carry-case sticky (else 0)
            }
            __uint32x2 __out = __fpemu_bit_cast<__uint32x2>(__out64);

            int32_t __res_exp = __exp_max - __nz + 2;

            __fpbits64_unpacked __r;
            __r.sign = __is_sign_c ? (1u << 31) : 0u;
            // +1 matches the universal pack's "mask the implicit bit" convention
            // (pack reconstitutes via exp-1); the +1/-1 cancel.
            __r.exponent = static_cast<uint32_t>(__res_exp);
            __r.mantissa = __fpemu_bit_cast<uint64_t>(__out);

            // Full-range boundary: no FTZ -- underflow flows to the pack, which
            // emits the correct subnormal for every method.
            return __r;
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
            (void)__NAN_EXP;
            int32_t  __exp_a  = (int32_t)__a.exponent;
            int32_t  __exp_b  = (int32_t)__b.exponent;
            uint32_t __sign_a = __a.sign;
            uint32_t __sign_b = __b.sign ^ (_IsSub ? _CCCL_FP64_HI_SIGN_MASK : 0u);

            int32_t __max_exp  = (__exp_a > __exp_b) ? __exp_a : __exp_b;
            int32_t __exp_diff = __max_exp - ((__exp_a > __exp_b) ? __exp_b : __exp_a);

            // Top 23 fraction bits of each operand (implicit bit at 61 dropped).
            uint32_t __mant_a_sp = (uint32_t)(__a.mantissa >> 38) & _CCCL_FP32_MANT_MASK;
            uint32_t __mant_b_sp = (uint32_t)(__b.mantissa >> 38) & _CCCL_FP32_MANT_MASK;

            // Scale both significands into [1.0, 2.0) at the fp32 bias, dropping
            // the relative exponent into the smaller operand's fp32 exponent.
            int32_t __exp_a_sp = _CCCL_FP32_BIAS;
            int32_t __exp_b_sp = _CCCL_FP32_BIAS;
            if (__exp_a < __exp_b) __exp_a_sp -= __exp_diff;
            if (__exp_b < __exp_a) __exp_b_sp -= __exp_diff;

            // Flush zero operands and underflowed alignments to +/-0.0f.
            if (__exp_a_sp < 1 || __a.mantissa == 0) { __exp_a_sp = 0; __mant_a_sp = 0; }
            if (__exp_b_sp < 1 || __b.mantissa == 0) { __exp_b_sp = 0; __mant_b_sp = 0; }

            __mant_a_sp |= ((uint32_t)__exp_a_sp << _CCCL_FP32_MANT_BITS) | __sign_a;
            __mant_b_sp |= ((uint32_t)__exp_b_sp << _CCCL_FP32_MANT_BITS) | __sign_b;

            float __fa   = __fpemu_bit_cast<float>(__mant_a_sp);
            float __fb   = __fpemu_bit_cast<float>(__mant_b_sp);
            // ~half-mantissa result: directed fp32 rounding is meaningless, so use
            // plain round-to-nearest fp32 (final rounding is the pack's job).
            float __fsum = __fadd_dir<__fpemu_rounding::rn>(__fa, __fb);
            int32_t __msum = __fpemu_bit_cast<int32_t>(__fsum);

            __fpbits64_unpacked __r;
            // Exact cancellation -> signed zero (mantissa 0 makes pack emit +/-0).
            if ((__msum & 0x7fffffff) == 0)
            {
                __r.sign     = (__msum < 0) ? (1u << 31) : 0u;
                __r.exponent = 0u;
                __r.mantissa = 0u;
                return __r;
            }

            int32_t  __exp_adjust  = ((__msum >> _CCCL_FP32_MANT_BITS) & _CCCL_FP32_LO_EXP_MASK) - _CCCL_FP32_BIAS;
            int32_t  __result_exp  = __max_exp + __exp_adjust;
            uint32_t __result_mant = (uint32_t)__msum & _CCCL_FP32_MANT_MASK;

            // Full-range boundary: no FTZ -- underflow flows to the pack, which
            // emits the correct subnormal for every method.

            // Re-expand the 24-bit fp32 significand (implicit bit + 23 fraction)
            // back to the universal scale with the implicit bit at position 61.
            uint64_t __sig = ((uint64_t)((1u << _CCCL_FP32_MANT_BITS) | __result_mant)) << 38;

            __r.sign     = (__msum < 0) ? (1u << 31) : 0u;
            // result_exp is already the result's biased exponent (the fp32
            // exp_adjust carried the binade change); pack maps exp_field == this.
            __r.exponent = (uint32_t)__result_exp;
            __r.mantissa = __sig;
            return __r;
        }
    } // __internal_fp64emu_dadd_unpacked

    /**
     * @brief Double-precision addition for FPEMU by native operations
    *
     * This function performs double-precision addition on two FPEMU floating-point numbers.
     * It takes two __fpbits64 structures as input, representing the packed sign, exponent,
     * and mantissa fields of the operands. The addition is performed according to the specified rounding mode,
     * accuracy, range, and engine template parameters.
     *
     * The function handles normalization, exponent alignment, sign management, and special cases such as
     * This function performs double-precision addition on two FPEMU floating-point numbers.
     * It takes two __fpbits64 structures as input, representing the packed sign, exponent,
     * and mantissa fields of the operands. The addition is performed according to the specified rounding mode,
     * accuracy, range, and engine template parameters.
     *
     * The function handles normalization, exponent alignment, sign management, and special cases such as
     * denormals and zeroes, producing a result in __fpbits64 form.
     *
     * @tparam rm    Rounding mode (rounding)
     * @tparam _Acc  Accuracy level (fpemu_accuracy)
     * @param x      First operand (packed)
     * @param y      Second operand (packed)
     * @return       Result of addition in packed form (__fpbits64)
     */
    template<__fpemu_rounding    _Rm    = __fpemu_rounding::def, 
             fpemu_accuracy      _Acc   = fpemu_accuracy::def,
             bool                _IsSub = false>
    _CCCL_TRIVIAL_API
    __fpbits64 __internal_fp64emu_dadd(__fpbits64 __x, 
                                          __fpbits64 __y) noexcept
    {
        // Forced parameters for the addition operation
        constexpr fpemu_accuracy   __acc_forced = fpemu_accuracy::_CCCL_FPEMU_ADD_METHOD;
        constexpr fpemu_accuracy   __acc_used   = (__acc_forced != fpemu_accuracy::unset) ? __acc_forced : _Acc;
        {
    #if (_CCCL_FPEMU_PACKED_VIA_UNPACKED == 1)
            {
                // Packed-via-unpacked (testing): pack(dadd_unpacked(unpack(x), unpack(y))). The
                // dadd_unpacked core selects accurate/def/fast internally; the
                // universal unpack/pack are the shared prologue/epilogue.
                __fpbits64_unpacked __a = __internal_fp64emu_unpack(__x);
                __fpbits64_unpacked __b = __internal_fp64emu_unpack(__y);
                __fpbits64_unpacked __r = __internal_fp64emu_dadd_unpacked<__acc_used, _IsSub>(__a, __b);
                return __internal_fp64emu_pack<_Rm>(__r);
            }
    #else
            if constexpr (__acc_used == fpemu_accuracy::high)
            {
                return __internal_fp64emu_dadd_accurate<_Rm, __acc_used, _IsSub>(__x, __y);
            }
            else if constexpr (__acc_used == fpemu_accuracy::mid)
            {
                return __internal_fp64emu_dadd_def<_Rm, __acc_used, _IsSub>(__x, __y);
            }
            else if constexpr (__acc_used == fpemu_accuracy::low)
            {
    #if _CCCL_FP64EMU_DADD_FP32_FAST_ENABLE == 1
                return __internal_fp64emu_dadd_fast<_Rm, __acc_used, _IsSub>(__x, __y);
    #else
                return __internal_fp64emu_dadd_def<_Rm, __acc_used, _IsSub>(__x, __y);
    #endif
            }
            else
            {
                return __internal_fp64emu_dadd_def<_Rm, __acc_used, _IsSub>(__x, __y);
            }
    #endif
        }
    } // __internal_fp64emu_dadd


// ============================================================================
// Builtin declarations/implementations for addition operations
// ============================================================================
#if defined(_CCCL_FPEMU_INLINE)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dadd_rn (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dadd<__fpemu_rounding::rn, fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dadd_rz (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dadd<__fpemu_rounding::rz, fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dadd_ru (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dadd<__fpemu_rounding::ru, fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dadd_rd (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dadd<__fpemu_rounding::rd, fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_high_dadd_rn (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dadd<__fpemu_rounding::rn, fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dadd_rn  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dadd<__fpemu_rounding::rn, fpemu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dadd_rz  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dadd<__fpemu_rounding::rz, fpemu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dadd_ru  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dadd<__fpemu_rounding::ru, fpemu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dadd_rd  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dadd<__fpemu_rounding::rd, fpemu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dadd_rn  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dadd<__fpemu_rounding::rn, fpemu_accuracy::low>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dadd_rz  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dadd<__fpemu_rounding::rz, fpemu_accuracy::low>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dadd_ru  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dadd<__fpemu_rounding::ru, fpemu_accuracy::low>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dadd_rd  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dadd<__fpemu_rounding::rd, fpemu_accuracy::low>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_dadd      (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_dadd_unpacked<fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_high_dadd (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_dadd_unpacked<fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_mid_dadd  (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_dadd_unpacked<fpemu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_low_dadd  (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_dadd_unpacked<fpemu_accuracy::low>(__x, __y); }
#else
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dadd_rn (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dadd_rz (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dadd_ru (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dadd_rd (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_high_dadd_rn (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dadd_rn  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dadd_rz  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dadd_ru  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dadd_rd  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dadd_rn  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dadd_rz  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dadd_ru  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dadd_rd  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_dadd      (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_high_dadd (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_mid_dadd  (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_low_dadd  (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
#endif // _CCCL_FPEMU_INLINE

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_ADD_H (builtins)

#if defined(_CCCL_FPEMU_API_CLASSES_DEFINED) && !defined(_CCCL_FPEMU_DADD_API_MERGED)
#define _CCCL_FPEMU_DADD_API_MERGED
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

 


// ============================================================================
// API (merged from fp64emu_dadd_api.hpp)
// ============================================================================
    // Default API implementation
    template<fpemu_accuracy _Acc>  
    _CCCL_API fpemu<double, _Acc> operator+ (const fpemu<double, _Acc>& __x, 
                                                    const fpemu<double, _Acc>& __y) noexcept
    {
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_high_dadd_rn(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::mid)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_mid_dadd_rn(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_low_dadd_rn(__x.bits, __y.bits)); }
        else                                             { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dadd_rn(__x.bits, __y.bits)); }
    } // operator+

    template<fpemu_accuracy _Acc>
    _CCCL_API fpemu<double, _Acc> __dadd_rn (const fpemu<double, _Acc>& __x, 
                                             const fpemu<double, _Acc>& __y) noexcept { 
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_high_dadd_rn(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_low_dadd_rn(__x.bits, __y.bits)); }
        else                                             { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_mid_dadd_rn(__x.bits, __y.bits)); }
    }
    template<fpemu_accuracy _Acc>
    _CCCL_API fpemu<double, _Acc> __dadd_rz (const fpemu<double, _Acc>& __x, 
                                             const fpemu<double, _Acc>& __y) noexcept {
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dadd_rz(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::mid)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_mid_dadd_rz(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_low_dadd_rz(__x.bits, __y.bits)); }
        else                                             { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dadd_rz(__x.bits, __y.bits)); }
    }
    template<fpemu_accuracy _Acc>
    _CCCL_API fpemu<double, _Acc> __dadd_ru (const fpemu<double, _Acc>& __x, 
                                             const fpemu<double, _Acc>& __y) noexcept {
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dadd_ru(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::mid)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_mid_dadd_ru(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_low_dadd_ru(__x.bits, __y.bits)); }
        else                                             { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dadd_ru(__x.bits, __y.bits)); }
    }
    template<fpemu_accuracy _Acc>
    _CCCL_API fpemu<double, _Acc> __dadd_rd (const fpemu<double, _Acc>& __x, 
                                             const fpemu<double, _Acc>& __y) noexcept {
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dadd_rd(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::mid)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_mid_dadd_rd(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_low_dadd_rd(__x.bits, __y.bits)); }
        else                                             { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dadd_rd(__x.bits, __y.bits)); }
    }


    // Operator+ for unpacked addition
    template<fpemu_accuracy _Acc>  
    _CCCL_DEVICE_API fpemu_unpacked<double, _Acc> operator+ (const fpemu_unpacked<double, _Acc>& __x, 
                                                                    const fpemu_unpacked<double, _Acc>& __y) noexcept
    {
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_high_dadd(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::mid)  { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_mid_dadd(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_low_dadd(__x.bits, __y.bits)); }
        else                                             { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_dadd(__x.bits, __y.bits)); }
    } // operator+


    template<fpemu_accuracy _Acc>
    _CCCL_API fpemu_unpacked<double, _Acc> __dadd_rn (const fpemu_unpacked<double, _Acc>& __x, 
                                                      const fpemu_unpacked<double, _Acc>& __y) noexcept { 
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_high_dadd(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_low_dadd(__x.bits, __y.bits)); }
        else                                             { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_mid_dadd(__x.bits, __y.bits)); }
    }


} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_ADD_H
