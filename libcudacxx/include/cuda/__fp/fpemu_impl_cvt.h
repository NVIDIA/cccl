//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_CVT_H
#define _CUDA___FP_FPEMU_IMPL_CVT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file fpemu_impl_cvt.h
 * @brief Implementation of type conversion operations for FPEMU floating point emulation library
 *
 * This header provides the implementation of type conversion operations for the FPEMU library.
 * It includes:
 *
 * - Conversion functions from fpemu to other types
 * - Conversion operators to other types
 * - Conversion functions to other types
 *
 * The conversion functions are designed to work across both host and device code
 * through appropriate decorators and provide bit-exact results matching hardware
 * floating point units.    
 */

#include <cuda/__fp/fpemu_impl.h>
#include <cuda/__fp/fpemu_impl_unpack.h>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


    // ========================================================================
    // fp64 -> integer conversions (self-contained; no SoftFloat dependency).
    //
    // The rounding core mirrors SoftFloat-3e (significand jam-shift + a
    // round-increment selected by mode/sign), while the out-of-range and NaN
    // handling matches CUDA hardware saturating conversions
    // (__double2int_rd, __double2ll_ru, ...):
    //   NaN              -> integer indefinite (sign bit only):
    //                       0x80000000 (32-bit) / 0x8000000000000000 (64-bit)
    //   signed   +ovf    -> INT_MAX ;  -ovf -> INT_MIN
    //   unsigned  ovf    -> UINT_MAX;  any negative -> 0
    // ========================================================================

    // Right shift keeping a sticky (jam) bit: __internal_fp64emu_shr_jam64,
    // shared with the divide/sqrt cores (see fpemu_impl.h).

    /// @brief Round a 64-bit fixed-point significand (12 fractional bits) to int32 (CUDA saturation).
    template<__fpemu_rounding _Rm>
    _CCCL_TRIVIAL_API int32_t __internal_fp64emu_round_to_i32 (bool __sign, uint64_t __sig) noexcept
    {
        const int32_t __sat = __sign ? (int32_t)0x80000000 : (int32_t)0x7FFFFFFF;
        uint32_t __round_increment = 0x800;
        if constexpr (_Rm != __fpemu_rounding::rn)
        {
            __round_increment = 0;
            const bool __toward_inf = __sign ? (_Rm == __fpemu_rounding::rd)
                                         : (_Rm == __fpemu_rounding::ru);
            if (__toward_inf) __round_increment = 0xFFF;
        }
        uint32_t __round_bits = (uint32_t)(__sig & 0xFFF);
        __sig += __round_increment;
        if (__sig & 0xFFFFF00000000000ULL) return __sat;

        uint32_t __sig32 = (uint32_t)(__sig >> 12);
        if constexpr (_Rm == __fpemu_rounding::rn)
        {
            if (__round_bits == 0x800) __sig32 &= ~(uint32_t)1;
        }
        uint32_t __uz = __sign ? (uint32_t)(0u - __sig32) : __sig32;
        int32_t __z = (int32_t)__uz;
        if (__z && ((__z < 0) != __sign)) return __sat;
        return __z;
    }

    /// @brief Round a 64-bit fixed-point significand (12 fractional bits) to uint32 (CUDA saturation).
    template<__fpemu_rounding _Rm>
    _CCCL_TRIVIAL_API uint32_t __internal_fp64emu_round_to_ui32 (bool __sign, uint64_t __sig) noexcept
    {
        if (__sign) return 0; // any negative real saturates to 0

        uint32_t __round_increment = (_Rm == __fpemu_rounding::rn) ? 0x800
                                : (_Rm == __fpemu_rounding::ru) ? 0xFFF : 0;
        uint32_t __round_bits = (uint32_t)(__sig & 0xFFF);
        __sig += __round_increment;
        if (__sig & 0xFFFFF00000000000ULL) return 0xFFFFFFFFu;

        uint32_t __z = (uint32_t)(__sig >> 12);
        if constexpr (_Rm == __fpemu_rounding::rn)
        {
            if (__round_bits == 0x800) __z &= ~(uint32_t)1;
        }
        return __z;
    }

    /// @brief Round (sig : sigExtra) to int64 (CUDA saturation).
    template<__fpemu_rounding _Rm>
    _CCCL_TRIVIAL_API int64_t __internal_fp64emu_round_to_i64 (bool __sign, uint64_t __sig, uint64_t __sig_extra) noexcept
    {
        const int64_t __sat = __sign ? (int64_t)0x8000000000000000ULL
                                     : (int64_t)0x7FFFFFFFFFFFFFFFULL;
        bool __increment;
        if constexpr (_Rm == __fpemu_rounding::rn)
        {
            __increment = (__sig_extra >= 0x8000000000000000ULL);
        }
        else
        {
            const bool __toward_inf = __sign ? (_Rm == __fpemu_rounding::rd)
                                         : (_Rm == __fpemu_rounding::ru);
            __increment = (__sig_extra != 0) && __toward_inf;
        }
        if (__increment)
        {
            ++__sig;
            if (!__sig) return __sat;
            if constexpr (_Rm == __fpemu_rounding::rn)
            {
                if (__sig_extra == 0x8000000000000000ULL) __sig &= ~(uint64_t)1;
            }
        }
        uint64_t __uz = __sign ? (uint64_t)(0ULL - __sig) : __sig;
        int64_t __z = (int64_t)__uz;
        if (__z && ((__z < 0) != __sign)) return __sat;
        return __z;
    }

    /// @brief Round (sig : sigExtra) to uint64 (CUDA saturation).
    template<__fpemu_rounding _Rm>
    _CCCL_TRIVIAL_API uint64_t __internal_fp64emu_round_to_ui64 (bool __sign, uint64_t __sig, uint64_t __sig_extra) noexcept
    {
        if (__sign) return 0; // any negative real saturates to 0

        bool __increment;
        if constexpr (_Rm == __fpemu_rounding::rn)
        {
            __increment = (__sig_extra >= 0x8000000000000000ULL);
        }
        else
        {
            __increment = (_Rm == __fpemu_rounding::ru) && (__sig_extra != 0);
        }
        if (__increment)
        {
            ++__sig;
            if (!__sig) return 0xFFFFFFFFFFFFFFFFULL;
            if constexpr (_Rm == __fpemu_rounding::rn)
            {
                if (__sig_extra == 0x8000000000000000ULL) __sig &= ~(uint64_t)1;
            }
        }
        return __sig;
    }

    /// @brief Convert a fp64 to an int32
    template<__fpemu_rounding _Rm  = __fpemu_rounding::rz>
    _CCCL_TRIVIAL_API  int32_t __internal_fp64emu_fpbits64_to_int (__fpbits64 __x) noexcept
    {
        const bool    __sign = ((uint64_t)__x >> 63) != 0;
        const int32_t __exp  = (int32_t)(((uint64_t)__x >> _CCCL_FP64_MANT_BITS) & 0x7FF);
        uint64_t      __sig  = (uint64_t)__x & __fpemu_mantissa_mask;

        if (__exp == 0x7FF && __sig) return (int32_t)0x80000000; // NaN -> integer indefinite

        if (__exp) __sig |= _CCCL_FPEMU_HIDDEN_64;
        int32_t __shift_dist = 0x427 - __exp;
        if (__shift_dist > 0) __sig = __internal_fp64emu_shr_jam64(__sig, (uint32_t)__shift_dist);
        return __internal_fp64emu_round_to_i32<_Rm>(__sign, __sig);
    } // __internal_fp64emu_fpbits64_to_int

    /// @brief Convert a fp64 to an uint32
    template<__fpemu_rounding _Rm  = __fpemu_rounding::rz>
    _CCCL_TRIVIAL_API  uint32_t __internal_fp64emu_fpbits64_to_uint (__fpbits64 __x) noexcept
    {
        const bool    __sign = ((uint64_t)__x >> 63) != 0;
        const int32_t __exp  = (int32_t)(((uint64_t)__x >> _CCCL_FP64_MANT_BITS) & 0x7FF);
        uint64_t      __sig  = (uint64_t)__x & __fpemu_mantissa_mask;

        if (__exp == 0x7FF && __sig) return 0x80000000u; // NaN -> integer indefinite

        if (__exp) __sig |= _CCCL_FPEMU_HIDDEN_64;
        int32_t __shift_dist = 0x427 - __exp;
        if (__shift_dist > 0) __sig = __internal_fp64emu_shr_jam64(__sig, (uint32_t)__shift_dist);
        return __internal_fp64emu_round_to_ui32<_Rm>(__sign, __sig);
    } // __internal_fp64emu_fpbits64_to_uint

    /// @brief Convert a fp64 to an int64
    template<__fpemu_rounding _Rm  = __fpemu_rounding::rz>
    _CCCL_TRIVIAL_API  int64_t __internal_fp64emu_fpbits64_to_ll (__fpbits64 __x) noexcept
    {
        const bool    __sign = ((uint64_t)__x >> 63) != 0;
        const int32_t __exp  = (int32_t)(((uint64_t)__x >> _CCCL_FP64_MANT_BITS) & 0x7FF);
        uint64_t      __sig  = (uint64_t)__x & __fpemu_mantissa_mask;

        if (__exp == 0x7FF && __sig) return (int64_t)0x8000000000000000ULL; // NaN -> integer indefinite

        if (__exp) __sig |= _CCCL_FPEMU_HIDDEN_64;
        int32_t __shift_dist = 0x433 - __exp;
        uint64_t __sig_int, __sig_extra;
        if (__shift_dist <= 0)
        {
            if (__shift_dist < -11) return __sign ? (int64_t)0x8000000000000000ULL
                                             : (int64_t)0x7FFFFFFFFFFFFFFFULL;
            __sig_int   = __sig << (-__shift_dist);
            __sig_extra = 0;
        }
        else if (__shift_dist < 64)
        {
            __sig_int   = __sig >> __shift_dist;
            __sig_extra = __sig << (-__shift_dist & 63);
        }
        else
        {
            __sig_int   = 0;
            __sig_extra = (__shift_dist == 64) ? __sig : (uint64_t)(__sig != 0);
        }
        return __internal_fp64emu_round_to_i64<_Rm>(__sign, __sig_int, __sig_extra);
    } // __internal_fp64emu_fpbits64_to_ll

    /// @brief Convert a fp64 to an uint64
    template<__fpemu_rounding _Rm  = __fpemu_rounding::rz>
    _CCCL_TRIVIAL_API  uint64_t __internal_fp64emu_fpbits64_to_ull (__fpbits64 __x) noexcept
    {
        const bool    __sign = ((uint64_t)__x >> 63) != 0;
        const int32_t __exp  = (int32_t)(((uint64_t)__x >> _CCCL_FP64_MANT_BITS) & 0x7FF);
        uint64_t      __sig  = (uint64_t)__x & __fpemu_mantissa_mask;

        if (__exp == 0x7FF && __sig) return 0x8000000000000000ULL; // NaN -> integer indefinite

        if (__exp) __sig |= _CCCL_FPEMU_HIDDEN_64;
        int32_t __shift_dist = 0x433 - __exp;
        uint64_t __sig_int, __sig_extra;
        if (__shift_dist <= 0)
        {
            // Negative saturates to 0; positive out-of-range to UINT64_MAX.
            if (__shift_dist < -11) return __sign ? 0ULL : 0xFFFFFFFFFFFFFFFFULL;
            __sig_int   = __sig << (-__shift_dist);
            __sig_extra = 0;
        }
        else if (__shift_dist < 64)
        {
            __sig_int   = __sig >> __shift_dist;
            __sig_extra = __sig << (-__shift_dist & 63);
        }
        else
        {
            __sig_int   = 0;
            __sig_extra = (__shift_dist == 64) ? __sig : (uint64_t)(__sig != 0);
        }
        return __internal_fp64emu_round_to_ui64<_Rm>(__sign, __sig_int, __sig_extra);
    } // __internal_fp64emu_fpbits64_to_ull

    /// @brief Convert a fp64 to a float
    _CCCL_TRIVIAL_API  float __internal_fp64emu_fpbits64_to_float (__fpbits64 __x) noexcept
    {
        uint64_t __bits = (uint64_t)__x;
        uint32_t __sign = (uint32_t)(__bits >> 63) << 31;
        int32_t  __exp  = (int32_t)((__bits >> _CCCL_FP64_MANT_BITS) & 0x7FF);
        uint64_t __frac = __bits & __fpemu_mantissa_mask;

        if (__exp == 0x7FF) 
        {
            if (__frac == 0)
                return __fpemu_bit_cast<float>(__sign | 0x7F800000u);
            // NaN: preserve sign, set quiet bit, keep upper payload
            uint32_t __frac32 = (uint32_t)(__frac >> 29) | 0x00400000u;
            return __fpemu_bit_cast<float>(__sign | 0x7F800000u | __frac32);
        }

        // Zero or subnormal double (below float range) → ±0
        if (__exp == 0)
            return __fpemu_bit_cast<float>(__sign);

        int32_t  __e_f = __exp - (_CCCL_FP64_BIAS - _CCCL_FP32_BIAS);
        uint64_t __sig = (1ULL << _CCCL_FP64_MANT_BITS) | __frac;

        if (__e_f >= 0xFF)
            return __fpemu_bit_cast<float>(__sign | 0x7F800000u);

        // Number of mantissa bits to discard: 52 - 23 = 29
        constexpr int32_t __drop = _CCCL_FP64_MANT_BITS - _CCCL_FP32_MANT_BITS;

        if (__e_f > 0) 
        {
            uint64_t __half  = 1ULL << (__drop - 1);
            uint64_t __trail = __sig & ((1ULL << __drop) - 1);
            uint32_t __sig24 = (uint32_t)(__sig >> __drop);

            if (__trail > __half || (__trail == __half && (__sig24 & 1))) 
            {
                __sig24++;
                if (__sig24 >> (_CCCL_FP32_MANT_BITS + 1)) 
                {
                    __sig24 >>= 1;
                    __e_f++;
                    if (__e_f >= 0xFF)
                        return __fpemu_bit_cast<float>(__sign | 0x7F800000u);
                }
            }

            uint32_t __frac_f = __sig24 & ((1u << _CCCL_FP32_MANT_BITS) - 1);
            return __fpemu_bit_cast<float>(__sign | ((uint32_t)__e_f << _CCCL_FP32_MANT_BITS) | __frac_f);
        } // if (e_f > 0)

        // Subnormal float output (e_f <= 0)
        int32_t __total_shift = __drop + 1 - __e_f;

        if (__total_shift >= 54)
            return __fpemu_bit_cast<float>(__sign);

        uint64_t __half  = 1ULL << (__total_shift - 1);
        uint64_t __trail = __sig & ((1ULL << __total_shift) - 1);
        uint32_t __sig_sub = (uint32_t)(__sig >> __total_shift);

        if (__trail > __half || (__trail == __half && (__sig_sub & 1)))
            __sig_sub++;

        // Overflow to 2^23 naturally becomes exponent=1, frac=0 (min normal)
        return __fpemu_bit_cast<float>(__sign | __sig_sub);
    } // __internal_fp64emu_fpbits64_to_float

    /// @brief Convert a float to a fp64
    _CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_float_to_fpbits64  (float __x) noexcept
    {
        uint32_t __bits = __fpemu_bit_cast<uint32_t>(__x);
        uint64_t __sign = (uint64_t)(__bits >> 31) << 63;
        int32_t  __exp  = (int32_t)((__bits >> _CCCL_FP32_MANT_BITS) & 0xFF);
        uint32_t __frac = __bits & ((1u << _CCCL_FP32_MANT_BITS) - 1);

        if (__exp == 0xFF) 
        {
            if (__frac == 0)
                return (__fpbits64)(__sign | ((uint64_t)0x7FF << _CCCL_FP64_MANT_BITS));
            // NaN: preserve sign, set quiet bit, widen payload
            uint64_t __d_frac = ((uint64_t)__frac << 29) | ((uint64_t)1 << (_CCCL_FP64_MANT_BITS - 1));
            return (__fpbits64)(__sign | ((uint64_t)0x7FF << _CCCL_FP64_MANT_BITS) | __d_frac);
        }

        if (__exp == 0 && __frac == 0)
            return (__fpbits64)__sign;

        // Subnormal float → normalize
        if (__exp == 0) 
        {
            int32_t __nz = __internal_clz((int)__frac) - (32 - _CCCL_FP32_MANT_BITS - 1);
            __frac = (__frac << __nz) & ((1u << _CCCL_FP32_MANT_BITS) - 1);
            __exp  = 1 - __nz;
        }

        // Exact widening conversion
        uint64_t __d_exp  = (uint64_t)(__exp + (_CCCL_FP64_BIAS - _CCCL_FP32_BIAS));
        uint64_t __d_frac = (uint64_t)__frac << (_CCCL_FP64_MANT_BITS - _CCCL_FP32_MANT_BITS);
        return (__fpbits64)(__sign | (__d_exp << _CCCL_FP64_MANT_BITS) | __d_frac);
    } // __internal_fp64emu_float_to_fpbits64

    /// @brief Convert a int32 to a fp64
    _CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_int_to_fpbits64  (int32_t __x) noexcept  
    { 
        if (__x == 0) return (__fpbits64)0;

        uint64_t __sign  = (__x < 0) ? (1ULL << 63) : 0ULL;
        uint32_t __abs_x = (uint32_t)((__x < 0) ? -(int64_t)__x : (int64_t)__x);

        int32_t __nz        = __internal_clz((int)__abs_x);
        uint64_t __exp      = (uint64_t)(_CCCL_FP64_BIAS + 31 - __nz);
        uint64_t __mantissa = ((uint64_t)__abs_x << (21 + __nz)) & __fpemu_mantissa_mask;

        return (__fpbits64)(__sign | (__exp << _CCCL_FP64_MANT_BITS) | __mantissa);
    } // __internal_fp64emu_int_to_fpbits64

    /// @brief Convert a uint32 to a fp64
    _CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_uint_to_fpbits64 (uint32_t __x) noexcept 
    { 
        if (__x == 0) return (__fpbits64)0;

        int32_t __nz        = __internal_clz((int)__x);
        uint64_t __exp      = (uint64_t)(_CCCL_FP64_BIAS + 31 - __nz);
        uint64_t __mantissa = ((uint64_t)__x << (21 + __nz)) & __fpemu_mantissa_mask;

        return (__fpbits64)((__exp << _CCCL_FP64_MANT_BITS) | __mantissa);
    } // __internal_fp64emu_uint_to_fpbits64

    /// @brief Convert a int64 to a fp64
    _CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_ll_to_fpbits64   (int64_t __x) noexcept  
    { 
        if (__x == 0) return (__fpbits64)0;

        uint64_t __sign = (__x < 0) ? (1ULL << 63) : 0ULL;
        uint64_t __abs_a = (__x < 0) ? -(uint64_t)__x : (uint64_t)__x;

        int32_t __nz = __internal_clzll((int64_t)__abs_a);
        int32_t __exp = _CCCL_FP64_BIAS + 63 - __nz;

        if (__nz >= 11) 
        {
            // <= 53 significant bits: exact
            uint64_t __mantissa = (__abs_a << (__nz - 11)) & __fpemu_mantissa_mask;
            return (__fpbits64)(__sign | ((uint64_t)__exp << _CCCL_FP64_MANT_BITS) | __mantissa);
        }

        // > 53 significant bits: round to nearest even
        int32_t __shift  = 11 - __nz;
        uint64_t __half  = 1ULL << (__shift - 1);
        uint64_t __trail = __abs_a & ((1ULL << __shift) - 1);
        uint64_t __sig53 = __abs_a >> __shift;

        if (__trail > __half || (__trail == __half && (__sig53 & 1))) 
        {
            __sig53++;
            if (__sig53 >> 53) { __sig53 >>= 1; __exp++; }
        }

        uint64_t __mantissa = __sig53 & __fpemu_mantissa_mask;
        return (__fpbits64)(__sign | ((uint64_t)__exp << _CCCL_FP64_MANT_BITS) | __mantissa);
    } // __internal_fp64emu_ll_to_fpbits64

    /// @brief Convert a uint64 to a fp64
    _CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_ull_to_fpbits64  (uint64_t __x) noexcept 
    {
        if (__x == 0) return (__fpbits64)0;

        int32_t __nz = __internal_clzll((int64_t)__x);
        int32_t __exp = _CCCL_FP64_BIAS + 63 - __nz;

        if (__nz >= 11) 
        {
            // <= 53 significant bits: exact
            uint64_t __mantissa = (__x << (__nz - 11)) & __fpemu_mantissa_mask;
            return (__fpbits64)(((uint64_t)__exp << _CCCL_FP64_MANT_BITS) | __mantissa);
        }

        // > 53 significant bits: round to nearest even
        int32_t __shift  = 11 - __nz;
        uint64_t __half  = 1ULL << (__shift - 1);
        uint64_t __trail = __x & ((1ULL << __shift) - 1);
        uint64_t __sig53 = __x >> __shift;

        if (__trail > __half || (__trail == __half && (__sig53 & 1))) 
        {
            __sig53++;
            if (__sig53 >> 53) { __sig53 >>= 1; __exp++; }
        }

        uint64_t __mantissa = __sig53 & __fpemu_mantissa_mask;
        return (__fpbits64)(((uint64_t)__exp << _CCCL_FP64_MANT_BITS) | __mantissa);
    } // __internal_fp64emu_ull_to_fpbits64


    // __fpbits64<->uint64 casts
    _CCCL_TRIVIAL_API uint64_t   __internal_fp64emu_fpbits64_cast_ull (__fpbits64 __x) noexcept { return __x; }
    _CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_ull_cast_fpbits64 (uint64_t __x) noexcept   { return __fpbits64{__x}; }

    // double<->__fpbits64 conversions
    _CCCL_TRIVIAL_API double     __internal_fp64emu_fpbits64_to_double (__fpbits64 __x) noexcept { return __fpemu_bit_cast<double>(__x); }
    _CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_double_to_fpbits64 (double __x) noexcept
    { 
        return __fpemu_bit_cast<__fpbits64>(__x);
    }

    _CCCL_TRIVIAL_API uint64_t   __internal_fp64emu_fpbits64_unpacked_cast_ull (__fpbits64_unpacked __x) noexcept 
    { 
        __fpbits64 __x_packed = __internal_fp64emu_pack(__x);
        return __internal_fp64emu_fpbits64_cast_ull(__x_packed);
    }

    _CCCL_TRIVIAL_API __fpbits64_unpacked __internal_fp64emu_ull_cast_fpbits64_unpacked (uint64_t __x) noexcept
    { 
        __fpbits64 __x_packed = __internal_fp64emu_ull_cast_fpbits64(__x);
        return __internal_fp64emu_unpack(__x_packed);
    }

    /**
     * @brief Convert a __fpbits64_unpacked to a double
     * 
     * This function converts a __fpbits64_unpacked to a double.
     * 
     * @param x The __fpbits64_unpacked to convert
     * @return The converted double
     */
    template<__fpemu_rounding _Rm   = __fpemu_rounding::def,
             fpemu_accuracy   _Acc  = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API double __internal_fp64emu_fpbits64_unpacked_to_double (__fpbits64_unpacked __x) noexcept 
    { 
        __fpbits64 __x_packed = __internal_fp64emu_pack<_Rm>(__x);
        return __internal_fp64emu_fpbits64_to_double(__x_packed); 
    }

    /**
     * @brief Convert a double to a __fpbits64_unpacked
     * 
     * This function converts a double to a __fpbits64_unpacked.
     * 
     * @param x The double to convert
     * @return The converted __fpbits64_unpacked
     */
    template<__fpemu_rounding _Rm   = __fpemu_rounding::def,
             fpemu_accuracy   _Acc  = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API __fpbits64_unpacked __internal_fp64emu_double_to_fpbits64_unpacked (double __x) noexcept
    { 
        __fpbits64 __x_packed = __internal_fp64emu_double_to_fpbits64(__x);
        return __internal_fp64emu_unpack(__x_packed);
    }

    // ------------------------------------------------------------------------
    // True unpacked -> integer conversions. Operate directly on the fully-accurate
    // unpacked fields (no operand pack): the full unpack already normalized
    // denormals (implicit bit at 61) and encoded inf/nan in the exponent band, so
    // the 53-bit significand is mantissa>>EXTRA_BITS and the exponent is the same
    // IEEE-biased value the packed converters consume. The shift/round/saturate
    // cores (shr_jam64 + round_to_*) are shared with the packed path, so results
    // match bit-for-bit (incl. NaN->indefinite and inf/overflow saturation).
    // ------------------------------------------------------------------------
    static constexpr int32_t __fp64emu_cvt_nan_exp = 0x0007ff00;
    static constexpr int32_t __fp64emu_cvt_inf_exp = 0x00007ff0;

    template<__fpemu_rounding _Rm   = __fpemu_rounding::def,
             fpemu_accuracy   _Acc  = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API  int32_t __internal_fp64emu_fpbits64_unpacked_to_int (__fpbits64_unpacked __x) noexcept
    {
        const bool    __sign = (__x.sign != 0);
        const int32_t __exp  = (int32_t)__x.exponent;
        if (__exp == __fp64emu_cvt_nan_exp) return (int32_t)0x80000000;                       // NaN -> indefinite
        if (__exp == __fp64emu_cvt_inf_exp) return __sign ? (int32_t)0x80000000 : (int32_t)0x7FFFFFFF;
        uint64_t __sig = __x.mantissa >> EXTRA_BITS;   // 53-bit significand (implicit at 52), 0 for zero
        int32_t  __shift_dist = 0x427 - __exp;
        if (__shift_dist > 0) __sig = __internal_fp64emu_shr_jam64(__sig, (uint32_t)__shift_dist);
        return __internal_fp64emu_round_to_i32<_Rm>(__sign, __sig);
    }
    template<__fpemu_rounding _Rm   = __fpemu_rounding::def,
             fpemu_accuracy   _Acc  = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API  uint32_t __internal_fp64emu_fpbits64_unpacked_to_uint (__fpbits64_unpacked __x) noexcept
    {
        const bool    __sign = (__x.sign != 0);
        const int32_t __exp  = (int32_t)__x.exponent;
        if (__exp == __fp64emu_cvt_nan_exp) return 0x80000000u;                               // NaN -> indefinite
        if (__exp == __fp64emu_cvt_inf_exp) return __sign ? 0u : 0xFFFFFFFFu;
        uint64_t __sig = __x.mantissa >> EXTRA_BITS;
        int32_t  __shift_dist = 0x427 - __exp;
        if (__shift_dist > 0) __sig = __internal_fp64emu_shr_jam64(__sig, (uint32_t)__shift_dist);
        return __internal_fp64emu_round_to_ui32<_Rm>(__sign, __sig);
    }

    template<__fpemu_rounding _Rm   = __fpemu_rounding::def,
             fpemu_accuracy   _Acc  = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API  int64_t __internal_fp64emu_fpbits64_unpacked_to_ll (__fpbits64_unpacked __x) noexcept
    {
        const bool    __sign = (__x.sign != 0);
        const int32_t __exp  = (int32_t)__x.exponent;
        if (__exp == __fp64emu_cvt_nan_exp) return (int64_t)0x8000000000000000ULL;            // NaN -> indefinite
        if (__exp == __fp64emu_cvt_inf_exp) return __sign ? (int64_t)0x8000000000000000ULL
                                                      : (int64_t)0x7FFFFFFFFFFFFFFFULL;
        uint64_t __sig = __x.mantissa >> EXTRA_BITS;
        int32_t  __shift_dist = 0x433 - __exp;
        uint64_t __sig_int, __sig_extra;
        if (__shift_dist <= 0)
        {
            if (__shift_dist < -11) return __sign ? (int64_t)0x8000000000000000ULL
                                             : (int64_t)0x7FFFFFFFFFFFFFFFULL;
            __sig_int   = __sig << (-__shift_dist);
            __sig_extra = 0;
        }
        else if (__shift_dist < 64)
        {
            __sig_int   = __sig >> __shift_dist;
            __sig_extra = __sig << (-__shift_dist & 63);
        }
        else
        {
            __sig_int   = 0;
            __sig_extra = (__shift_dist == 64) ? __sig : (uint64_t)(__sig != 0);
        }
        return __internal_fp64emu_round_to_i64<_Rm>(__sign, __sig_int, __sig_extra);
    }

    template<__fpemu_rounding _Rm   = __fpemu_rounding::def,
             fpemu_accuracy   _Acc  = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API  uint64_t __internal_fp64emu_fpbits64_unpacked_to_ull (__fpbits64_unpacked __x) noexcept
    {
        const bool    __sign = (__x.sign != 0);
        const int32_t __exp  = (int32_t)__x.exponent;
        if (__exp == __fp64emu_cvt_nan_exp) return 0x8000000000000000ULL;                     // NaN -> indefinite
        if (__exp == __fp64emu_cvt_inf_exp) return __sign ? 0ULL : 0xFFFFFFFFFFFFFFFFULL;
        uint64_t __sig = __x.mantissa >> EXTRA_BITS;
        int32_t  __shift_dist = 0x433 - __exp;
        uint64_t __sig_int, __sig_extra;
        if (__shift_dist <= 0)
        {
            if (__shift_dist < -11) return __sign ? 0ULL : 0xFFFFFFFFFFFFFFFFULL;
            __sig_int   = __sig << (-__shift_dist);
            __sig_extra = 0;
        }
        else if (__shift_dist < 64)
        {
            __sig_int   = __sig >> __shift_dist;
            __sig_extra = __sig << (-__shift_dist & 63);
        }
        else
        {
            __sig_int   = 0;
            __sig_extra = (__shift_dist == 64) ? __sig : (uint64_t)(__sig != 0);
        }
        return __internal_fp64emu_round_to_ui64<_Rm>(__sign, __sig_int, __sig_extra);
    }

    template<__fpemu_rounding _Rm   = __fpemu_rounding::def,
             fpemu_accuracy   _Acc  = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API  float __internal_fp64emu_fpbits64_unpacked_to_float (__fpbits64_unpacked __x) noexcept
    {
        __fpbits64 __x_packed = __internal_fp64emu_pack<_Rm>(__x);
        return __internal_fp64emu_fpbits64_to_float(__x_packed);
    }

    template<__fpemu_rounding _Rm   = __fpemu_rounding::def,
             fpemu_accuracy   _Acc  = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API __fpbits64_unpacked __internal_fp64emu_float_to_fpbits64_unpacked  (float __x) noexcept     
    { 
        __fpbits64 __x_packed = __internal_fp64emu_float_to_fpbits64(__x);
        return __internal_fp64emu_unpack(__x_packed);
    }

    template<__fpemu_rounding _Rm   = __fpemu_rounding::def,
             fpemu_accuracy   _Acc  = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API __fpbits64_unpacked __internal_fp64emu_int_to_fpbits64_unpacked  (int32_t __x) noexcept     
    { 
        __fpbits64 __x_packed = __internal_fp64emu_int_to_fpbits64(__x);
        return __internal_fp64emu_unpack(__x_packed);
    }

    template<__fpemu_rounding _Rm   = __fpemu_rounding::def,
             fpemu_accuracy   _Acc  = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API __fpbits64_unpacked __internal_fp64emu_uint_to_fpbits64_unpacked  (uint32_t __x) noexcept     
    { 
        __fpbits64 __x_packed = __internal_fp64emu_uint_to_fpbits64(__x);
        return __internal_fp64emu_unpack(__x_packed);
    }   

    template<__fpemu_rounding _Rm   = __fpemu_rounding::def,
             fpemu_accuracy   _Acc  = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API __fpbits64_unpacked __internal_fp64emu_ull_to_fpbits64_unpacked  (uint64_t __x) noexcept     
    { 
        __fpbits64 __x_packed = __internal_fp64emu_ull_to_fpbits64(__x);
        return __internal_fp64emu_unpack(__x_packed);
    }

    template<__fpemu_rounding _Rm   = __fpemu_rounding::def,
             fpemu_accuracy   _Acc  = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API __fpbits64_unpacked __internal_fp64emu_ll_to_fpbits64_unpacked  (int64_t __x) noexcept     
    { 
        __fpbits64 __x_packed = __internal_fp64emu_ll_to_fpbits64(__x);
        return __internal_fp64emu_unpack(__x_packed);
    }

// ============================================================================
// Builtin declarations/implementations for conversion operations
// ============================================================================
#if defined(_CCCL_FPEMU_INLINE)
#if (_CCCL_FPEMU_PACKED_VIA_UNPACKED == 1)
// Packed-via-unpacked (testing): route the packed conversion builtins through the
// unpacked cores. fp->int unpack(x) then the rounding-aware unpacked core; fp->fp
// goes through the universal unpack/pack; int/fp->fp builds the unpacked value and
// packs (rn -- the integer/widening conversions are exact or already rounded).
// The pure bit-reinterpret casts (__fpbits64<->ull) are NOT rerouted: they must
// preserve the exact bit pattern and have no unpacked-core equivalent.
_CCCL_FPEMU_BUILTIN_DECL double     __fp64emu_to_double (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_double (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL float      __fp64emu_to_float  (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_float (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL int32_t    __fp64emu_to_int_rn (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_int<__fpemu_rounding::rn> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL int32_t    __fp64emu_to_int_rz (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_int<__fpemu_rounding::rz> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL int32_t    __fp64emu_to_int_ru (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_int<__fpemu_rounding::ru> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL int32_t    __fp64emu_to_int_rd (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_int<__fpemu_rounding::rd> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL uint32_t   __fp64emu_to_uint_rn (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_uint<__fpemu_rounding::rn> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL uint32_t   __fp64emu_to_uint_rz (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_uint<__fpemu_rounding::rz> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL uint32_t   __fp64emu_to_uint_ru (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_uint<__fpemu_rounding::ru> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL uint32_t   __fp64emu_to_uint_rd (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_uint<__fpemu_rounding::rd> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL int64_t    __fp64emu_to_ll_rn (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_ll<__fpemu_rounding::rn> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL int64_t    __fp64emu_to_ll_rz (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_ll<__fpemu_rounding::rz> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL int64_t    __fp64emu_to_ll_ru (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_ll<__fpemu_rounding::ru> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL int64_t    __fp64emu_to_ll_rd (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_ll<__fpemu_rounding::rd> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_to_ull_rn (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_ull<__fpemu_rounding::rn> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_to_ull_rz (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_ull<__fpemu_rounding::rz> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_to_ull_ru (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_ull<__fpemu_rounding::ru> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_to_ull_rd (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_ull<__fpemu_rounding::rd> (__internal_fp64emu_unpack (__x)); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_double (double __x) noexcept   { return __internal_fp64emu_pack<__fpemu_rounding::rn> (__internal_fp64emu_double_to_fpbits64_unpacked (__x)); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_float  (float __x) noexcept    { return __internal_fp64emu_pack<__fpemu_rounding::rn> (__internal_fp64emu_float_to_fpbits64_unpacked (__x)); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_int    (int32_t __x) noexcept  { return __internal_fp64emu_pack<__fpemu_rounding::rn> (__internal_fp64emu_int_to_fpbits64_unpacked (__x)); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_uint   (uint32_t __x) noexcept { return __internal_fp64emu_pack<__fpemu_rounding::rn> (__internal_fp64emu_uint_to_fpbits64_unpacked (__x)); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_ll     (int64_t __x) noexcept  { return __internal_fp64emu_pack<__fpemu_rounding::rn> (__internal_fp64emu_ll_to_fpbits64_unpacked (__x)); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_ull    (uint64_t __x) noexcept { return __internal_fp64emu_pack<__fpemu_rounding::rn> (__internal_fp64emu_ull_to_fpbits64_unpacked (__x)); }
#else
_CCCL_FPEMU_BUILTIN_DECL double     __fp64emu_to_double (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_double (__x); }
_CCCL_FPEMU_BUILTIN_DECL float      __fp64emu_to_float  (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_float (__x); }
_CCCL_FPEMU_BUILTIN_DECL int32_t    __fp64emu_to_int_rn (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_int<__fpemu_rounding::rn> (__x); }
_CCCL_FPEMU_BUILTIN_DECL int32_t    __fp64emu_to_int_rz (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_int<__fpemu_rounding::rz> (__x); }
_CCCL_FPEMU_BUILTIN_DECL int32_t    __fp64emu_to_int_ru (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_int<__fpemu_rounding::ru> (__x); }
_CCCL_FPEMU_BUILTIN_DECL int32_t    __fp64emu_to_int_rd (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_int<__fpemu_rounding::rd> (__x); }
_CCCL_FPEMU_BUILTIN_DECL uint32_t   __fp64emu_to_uint_rn (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_uint<__fpemu_rounding::rn> (__x); }
_CCCL_FPEMU_BUILTIN_DECL uint32_t   __fp64emu_to_uint_rz (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_uint<__fpemu_rounding::rz> (__x); }
_CCCL_FPEMU_BUILTIN_DECL uint32_t   __fp64emu_to_uint_ru (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_uint<__fpemu_rounding::ru> (__x); }
_CCCL_FPEMU_BUILTIN_DECL uint32_t   __fp64emu_to_uint_rd (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_uint<__fpemu_rounding::rd> (__x); }
_CCCL_FPEMU_BUILTIN_DECL int64_t    __fp64emu_to_ll_rn (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_ll<__fpemu_rounding::rn> (__x); }
_CCCL_FPEMU_BUILTIN_DECL int64_t    __fp64emu_to_ll_rz (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_ll<__fpemu_rounding::rz> (__x); }
_CCCL_FPEMU_BUILTIN_DECL int64_t    __fp64emu_to_ll_ru (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_ll<__fpemu_rounding::ru> (__x); }
_CCCL_FPEMU_BUILTIN_DECL int64_t    __fp64emu_to_ll_rd (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_ll<__fpemu_rounding::rd> (__x); }
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_to_ull_rn (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_ull<__fpemu_rounding::rn> (__x); }
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_to_ull_rz (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_ull<__fpemu_rounding::rz> (__x); }
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_to_ull_ru (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_ull<__fpemu_rounding::ru> (__x); }
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_to_ull_rd (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_to_ull<__fpemu_rounding::rd> (__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_double (double __x) noexcept   { return __internal_fp64emu_double_to_fpbits64 (__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_float  (float __x) noexcept    { return __internal_fp64emu_float_to_fpbits64 (__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_int    (int32_t __x) noexcept  { return __internal_fp64emu_int_to_fpbits64 (__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_uint   (uint32_t __x) noexcept { return __internal_fp64emu_uint_to_fpbits64 (__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_ll     (int64_t __x) noexcept  { return __internal_fp64emu_ll_to_fpbits64 (__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_ull    (uint64_t __x) noexcept { return __internal_fp64emu_ull_to_fpbits64 (__x); }
#endif // _CCCL_FPEMU_PACKED_VIA_UNPACKED
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_fpbits64_cast_ull  (__fpbits64 __x) noexcept { return __internal_fp64emu_fpbits64_cast_ull (__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_ull_cast_fpbits64  (uint64_t __x) noexcept   { return __internal_fp64emu_ull_cast_fpbits64 (__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpack  (__fpbits64 __a) noexcept          { return __internal_fp64emu_unpack(__a); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64          __fp64emu_pack_rn (__fpbits64_unpacked __a) noexcept { return __internal_fp64emu_pack<__fpemu_rounding::rn>(__a); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64          __fp64emu_pack_rz (__fpbits64_unpacked __a) noexcept { return __internal_fp64emu_pack<__fpemu_rounding::rz>(__a); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64          __fp64emu_pack_ru (__fpbits64_unpacked __a) noexcept { return __internal_fp64emu_pack<__fpemu_rounding::ru>(__a); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64          __fp64emu_pack_rd (__fpbits64_unpacked __a) noexcept { return __internal_fp64emu_pack<__fpemu_rounding::rd>(__a); }
_CCCL_FPEMU_BUILTIN_DECL int32_t  __fp64emu_unpacked_to_int            (__fpbits64_unpacked __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_int<__fpemu_rounding::rz>(__x); }
_CCCL_FPEMU_BUILTIN_DECL uint32_t __fp64emu_unpacked_to_uint           (__fpbits64_unpacked __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_uint<__fpemu_rounding::rz>(__x); }
_CCCL_FPEMU_BUILTIN_DECL int64_t  __fp64emu_unpacked_to_ll             (__fpbits64_unpacked __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_ll<__fpemu_rounding::rz>(__x); }
_CCCL_FPEMU_BUILTIN_DECL uint64_t __fp64emu_unpacked_to_ull            (__fpbits64_unpacked __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_ull<__fpemu_rounding::rz>(__x); }
_CCCL_FPEMU_BUILTIN_DECL float    __fp64emu_unpacked_to_float          (__fpbits64_unpacked __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_float(__x); }
_CCCL_FPEMU_BUILTIN_DECL double   __fp64emu_unpacked_to_double         (__fpbits64_unpacked __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_double(__x); }
_CCCL_FPEMU_BUILTIN_DECL double   __fp64emu_unpacked_high_to_double(__fpbits64_unpacked __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_double<__fpemu_rounding::rn, fpemu_accuracy::high>(__x); }
_CCCL_FPEMU_BUILTIN_DECL double   __fp64emu_unpacked_mid_to_double     (__fpbits64_unpacked __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_double<__fpemu_rounding::rn, fpemu_accuracy::mid>(__x); }
_CCCL_FPEMU_BUILTIN_DECL double   __fp64emu_unpacked_low_to_double    (__fpbits64_unpacked __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_to_double<__fpemu_rounding::rn, fpemu_accuracy::low>(__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_from_int             (int32_t __x) noexcept  { return __internal_fp64emu_int_to_fpbits64_unpacked(__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_from_uint            (uint32_t __x) noexcept { return __internal_fp64emu_uint_to_fpbits64_unpacked(__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_from_ll              (int64_t __x) noexcept  { return __internal_fp64emu_ll_to_fpbits64_unpacked(__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_from_ull             (uint64_t __x) noexcept { return __internal_fp64emu_ull_to_fpbits64_unpacked(__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_from_float           (float __x) noexcept    { return __internal_fp64emu_float_to_fpbits64_unpacked(__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_from_double          (double __x) noexcept   { return __internal_fp64emu_double_to_fpbits64_unpacked(__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_high_from_double (double __x) noexcept   { return __internal_fp64emu_double_to_fpbits64_unpacked<__fpemu_rounding::rn, fpemu_accuracy::high>(__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_mid_from_double      (double __x) noexcept   { return __internal_fp64emu_double_to_fpbits64_unpacked<__fpemu_rounding::rn, fpemu_accuracy::mid>(__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_low_from_double     (double __x) noexcept   { return __internal_fp64emu_double_to_fpbits64_unpacked<__fpemu_rounding::rn, fpemu_accuracy::low>(__x); }
_CCCL_FPEMU_BUILTIN_DECL uint64_t            __fp64emu_unpacked_fpbits64_cast_ull           (__fpbits64_unpacked __x) noexcept { return __internal_fp64emu_fpbits64_unpacked_cast_ull(__x); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_ull_cast_fpbits64           (uint64_t __x) noexcept { return __internal_fp64emu_ull_cast_fpbits64_unpacked(__x); }
#else
_CCCL_FPEMU_BUILTIN_DECL double     __fp64emu_to_double (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_double (double x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL float      __fp64emu_to_float  (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_float  (float x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_int    (int32_t x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_uint   (uint32_t x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_ll     (int64_t x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_from_ull    (uint64_t x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_fpbits64_cast_ull  (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_ull_cast_fpbits64  (uint64_t x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL int32_t    __fp64emu_to_int_rn (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL int32_t    __fp64emu_to_int_rz (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL int32_t    __fp64emu_to_int_ru (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL int32_t    __fp64emu_to_int_rd (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL uint32_t   __fp64emu_to_uint_rn (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL uint32_t   __fp64emu_to_uint_rz (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL uint32_t   __fp64emu_to_uint_ru (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL uint32_t   __fp64emu_to_uint_rd (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL int64_t    __fp64emu_to_ll_rn (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL int64_t    __fp64emu_to_ll_rz (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL int64_t    __fp64emu_to_ll_ru (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL int64_t    __fp64emu_to_ll_rd (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_to_ull_rn (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_to_ull_rz (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_to_ull_ru (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL uint64_t   __fp64emu_to_ull_rd (__fpbits64 x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpack  (__fpbits64 a) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64          __fp64emu_pack_rn (__fpbits64_unpacked a) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64          __fp64emu_pack_rz (__fpbits64_unpacked a) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64          __fp64emu_pack_ru (__fpbits64_unpacked a) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64          __fp64emu_pack_rd (__fpbits64_unpacked a) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL int32_t  __fp64emu_unpacked_to_int            (__fpbits64_unpacked x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL uint32_t __fp64emu_unpacked_to_uint           (__fpbits64_unpacked x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL int64_t  __fp64emu_unpacked_to_ll             (__fpbits64_unpacked x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL uint64_t __fp64emu_unpacked_to_ull            (__fpbits64_unpacked x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL float    __fp64emu_unpacked_to_float          (__fpbits64_unpacked x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL double   __fp64emu_unpacked_to_double         (__fpbits64_unpacked x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL double   __fp64emu_unpacked_high_to_double(__fpbits64_unpacked x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL double   __fp64emu_unpacked_mid_to_double     (__fpbits64_unpacked x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL double   __fp64emu_unpacked_low_to_double    (__fpbits64_unpacked x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_from_int             (int32_t x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_from_uint            (uint32_t x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_from_ll              (int64_t x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_from_ull             (uint64_t x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_from_float           (float x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_from_double          (double x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_high_from_double (double x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_mid_from_double      (double x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_low_from_double     (double x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL uint64_t            __fp64emu_unpacked_fpbits64_cast_ull           (__fpbits64_unpacked x) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_ull_cast_fpbits64           (uint64_t x) noexcept ;
#endif // _CCCL_FPEMU_INLINE

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_CVT_H

#if defined(_CCCL_FPEMU_API_CLASSES_DEFINED) && !defined(_CCCL_FPEMU_CVT_API_MERGED)
#define _CCCL_FPEMU_CVT_API_MERGED
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_cvt_api.hpp)
// ============================================================================

    // Type conversion to fpemu with other method
    template<typename _FpType, fpemu_accuracy _AccSrc> 
    template<fpemu_accuracy _AccDst> 
        _CCCL_API inline fpemu<_FpType, _AccSrc>::operator fpemu<double, _AccDst>() const noexcept 
        { 
            return fpemu<double, _AccDst>(__fpbits64_construct, bits); 
        }

    // Type conversion from fpemu to fpemu_unpacked
    template<typename _FpType, fpemu_accuracy _AccSrc> 
    template<fpemu_accuracy _AccDst> 
        _CCCL_API inline fpemu<_FpType, _AccSrc>::operator fpemu_unpacked<double, _AccDst>() const noexcept 
        { 
            __fpbits64_unpacked __bits_unpacked = __fp64emu_unpack(bits);
            return fpemu_unpacked<double, _AccDst>(__fpbits64_construct, __bits_unpacked); 
        }

    /*
    // Type conversions from other types to fpemu 
    */
    // from double
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline fpemu<_FpType, _Acc>::fpemu(double __d) noexcept { 
        bits = __fp64emu_from_double (__d); }
    // from float
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline fpemu<_FpType, _Acc>::fpemu(float __d) noexcept { 
        bits = __fp64emu_from_float (__d); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline fpemu<double, _Acc> __float2double  (float __x) noexcept { 
        return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_from_float (__x)); }
    // from int32_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline void fpemu<_FpType, _Acc>::__set_from_int (int32_t __d) noexcept { 
        bits = __fp64emu_from_int (__d); }
    template<fpemu_accuracy _Acc> _CCCL_API inline fpemu<double, _Acc> __int2double  (int32_t __x) noexcept { 
        return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_from_int (__x)); }
    // from uint32_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline void fpemu<_FpType, _Acc>::__set_from_uint(uint32_t __d) noexcept { 
        bits = __fp64emu_from_uint (__d); }
    template<fpemu_accuracy _Acc> _CCCL_API inline fpemu<double, _Acc> __uint2double (uint32_t __x) noexcept { 
        return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_from_uint (__x)); }
    // from int64_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline void fpemu<_FpType, _Acc>::__set_from_ll  (int64_t __d) noexcept { 
        bits = __fp64emu_from_ll (__d); }
    template<fpemu_accuracy _Acc> _CCCL_API inline fpemu<double, _Acc> __ll2double (int64_t __x) noexcept { 
        return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_from_ll (__x)); }
    // from uint64_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline void fpemu<_FpType, _Acc>::__set_from_ull (uint64_t __d) noexcept { 
        bits = __fp64emu_from_ull (__d); }
    template<fpemu_accuracy _Acc> _CCCL_API inline fpemu<double, _Acc> __ull2double(uint64_t __x) noexcept { 
        return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_from_ull (__x)); }

    /*
    // Type conversions from fpemu to other types
    */
    // to double
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline fpemu<_FpType, _Acc>::operator double() const noexcept { 
        return __fp64emu_to_double (bits); }
    // to float
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline fpemu<_FpType, _Acc>::operator float()  const noexcept { 
        return __fp64emu_to_float (bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline float  __double2float (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_float (__x.bits); }
    // to int32_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline int32_t fpemu<_FpType, _Acc>::__to_int ()  const noexcept { 
        return __fp64emu_to_int_rz (bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline int32_t __double2int_rn (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_int_rn (__x.bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline int32_t __double2int_rz (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_int_rz (__x.bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline int32_t __double2int_ru (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_int_ru (__x.bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline int32_t __double2int_rd (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_int_rd (__x.bits); }
    // to uint32_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline uint32_t fpemu<_FpType, _Acc>::__to_uint() const noexcept { 
        return __fp64emu_to_uint_rz (bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline uint32_t __double2uint_rn (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_uint_rn (__x.bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline uint32_t __double2uint_rz (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_uint_rz (__x.bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline uint32_t __double2uint_ru (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_uint_ru (__x.bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline uint32_t __double2uint_rd (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_uint_rd (__x.bits); }
    // to int64_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline int64_t fpemu<_FpType, _Acc>::__to_ll ()  const noexcept { 
        return __fp64emu_to_ll_rz (bits); }    
    template<fpemu_accuracy _Acc>  _CCCL_API inline int64_t __double2ll_rn (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_ll_rn (__x.bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline int64_t __double2ll_rz (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_ll_rz (__x.bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline int64_t __double2ll_ru (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_ll_ru (__x.bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline int64_t __double2ll_rd (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_ll_rd (__x.bits); }    
    // to uint64_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline uint64_t fpemu<_FpType, _Acc>::__to_ull () const noexcept { 
        return __fp64emu_to_ull_rz (bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline uint64_t __double2ull_rn (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_ull_rn (__x.bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline uint64_t __double2ull_rz (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_ull_rz (__x.bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline uint64_t __double2ull_ru (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_ull_ru (__x.bits); }
    template<fpemu_accuracy _Acc>  _CCCL_API inline uint64_t __double2ull_rd (fpemu<double, _Acc> __x) noexcept { 
        return __fp64emu_to_ull_rd (__x.bits); }


    // Type conversion from fpemu_unpacked with other method
    template<typename _FpType, fpemu_accuracy _AccSrc> 
    template<fpemu_accuracy _AccDst> 
        _CCCL_API inline fpemu_unpacked<_FpType, _AccSrc>::operator fpemu_unpacked<double, _AccDst>() const noexcept 
        { 
            return fpemu_unpacked<double, _AccDst>(__fpbits64_construct, bits); 
        }

    // Type conversion from fpemu_unpacked to fpemu
    template<typename _FpType, fpemu_accuracy _AccSrc> 
    template<fpemu_accuracy _AccDst> 
        _CCCL_API inline fpemu_unpacked<_FpType, _AccSrc>::operator fpemu<double, _AccDst>() const noexcept 
        { 
            __fpbits64 __bits_packed = __fp64emu_pack_rn(bits);
            return fpemu<double, _AccDst>(__fpbits64_construct, __bits_packed); 
        }

    /*
    // Type conversions from other types to fpemu_unpacked 
    */
    // from double 
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline fpemu_unpacked<_FpType, _Acc>::fpemu_unpacked(double __d) noexcept 
    { 
    if      constexpr (_Acc == fpemu_accuracy::high) { bits = __fp64emu_unpacked_high_from_double(__d); }
        else if constexpr (_Acc == fpemu_accuracy::mid)      { bits = __fp64emu_unpacked_mid_from_double(__d); }
        else                                      { bits = __fp64emu_unpacked_from_double(__d); }
    }
    // from float 
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline fpemu_unpacked<_FpType, _Acc>::fpemu_unpacked(float __d) noexcept { 
        bits = __fp64emu_unpacked_from_float (__d);  }
    template<fpemu_accuracy _Acc> _CCCL_API inline fpemu_unpacked<double, _Acc>  __float2double (float __x) noexcept { 
        return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_from_float (__x)); }
    // from int32_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline void fpemu_unpacked<_FpType, _Acc>::__set_from_int (int32_t __d) noexcept { 
        bits = __fp64emu_unpacked_from_int (__d); }
    template<fpemu_accuracy _Acc> _CCCL_API inline fpemu_unpacked<double, _Acc>  __int2double (int32_t __x) noexcept { 
        return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_from_int (__x)); }
    // from uint32_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline void fpemu_unpacked<_FpType, _Acc>::__set_from_uint(uint32_t __d) noexcept { 
        bits = __fp64emu_unpacked_from_uint (__d); }
    template<fpemu_accuracy _Acc> _CCCL_API inline fpemu_unpacked<double, _Acc>  __uint2double (uint32_t __x) noexcept { 
        return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_from_uint (__x)); }
    // from int64_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline void fpemu_unpacked<_FpType, _Acc>::__set_from_ll  (int64_t __d) noexcept { 
        bits = __fp64emu_unpacked_from_ll (__d); }
    template<fpemu_accuracy _Acc> _CCCL_API inline fpemu_unpacked<double, _Acc>  __ll2double (int64_t __x) noexcept  { 
        return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_from_ll (__x)); }
    // from uint64_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline void fpemu_unpacked<_FpType, _Acc>::__set_from_ull (uint64_t __d) noexcept { 
        bits = __fp64emu_unpacked_from_ull (__d); }
    template<fpemu_accuracy _Acc> _CCCL_API inline fpemu_unpacked<double, _Acc>  __ull2double (uint64_t __x) noexcept { 
        return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_from_ull (__x)); }

    /*
    // Conversion operators from fpemu_unpacked to other types
    */
    // to double
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline fpemu_unpacked<_FpType, _Acc>::operator double() const noexcept 
    { 
if      constexpr (_Acc == fpemu_accuracy::high) { return __fp64emu_unpacked_high_to_double(bits); }
       else if constexpr (_Acc == fpemu_accuracy::mid)      { return __fp64emu_unpacked_mid_to_double(bits); }
       else                                      { return __fp64emu_unpacked_to_double(bits); }
    }
    // to float
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline fpemu_unpacked<_FpType, _Acc>::operator float()  const noexcept { 
        return __fp64emu_unpacked_to_float (bits); }
    template<fpemu_accuracy _Acc> _CCCL_API inline float    __double2float   (fpemu_unpacked<double, _Acc> __x) noexcept { 
        return __fp64emu_unpacked_to_float (__x.bits); }
    // to int32_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline int32_t fpemu_unpacked<_FpType, _Acc>::__to_int ()  const noexcept { 
        return __fp64emu_unpacked_to_int (bits); }
    template<fpemu_accuracy _Acc> _CCCL_API inline int32_t  __double2int_rz  (fpemu_unpacked<double, _Acc> __x) noexcept { 
        return __fp64emu_unpacked_to_int (__x.bits); }
    // to uint32_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline uint32_t fpemu_unpacked<_FpType, _Acc>::__to_uint() const noexcept { 
        return __fp64emu_unpacked_to_uint (bits); }
    template<fpemu_accuracy _Acc> _CCCL_API inline uint32_t __double2uint_rz (fpemu_unpacked<double, _Acc> __x) noexcept { 
        return __fp64emu_unpacked_to_uint (__x.bits); }
    // to int64_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline int64_t fpemu_unpacked<_FpType, _Acc>::__to_ll ()  const noexcept { 
        return __fp64emu_unpacked_to_ll (bits); } 
    template<fpemu_accuracy _Acc> _CCCL_API inline int64_t  __double2ll_rz   (fpemu_unpacked<double, _Acc> __x) noexcept { 
        return __fp64emu_unpacked_to_ll (__x.bits); }
    // to uint64_t
    template <typename _FpType, fpemu_accuracy _Acc> _CCCL_API inline uint64_t fpemu_unpacked<_FpType, _Acc>::__to_ull () const noexcept { 
        return __fp64emu_unpacked_to_ull (bits); } 
    template<fpemu_accuracy _Acc> _CCCL_API inline uint64_t __double2ull_rz  (fpemu_unpacked<double, _Acc> __x) noexcept { 
        return __fp64emu_unpacked_to_ull (__x.bits); }
    template<typename _To, fpemu_accuracy _M2>
        _CCCL_API inline _To bit_cast(const fpemu_unpacked<double, _M2>& __from) noexcept
        {
            // Pack the unpacked value to get IEEE-754 representation
            __fpbits64 __packed = __fp64emu_pack_rn(__from.bits);
            return __fpemu_bit_cast<_To>(__packed);
        }

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CCCL_FPEMU_CVT_API_MERGED
