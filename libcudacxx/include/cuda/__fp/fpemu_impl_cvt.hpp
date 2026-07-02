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
 * @file fpemu_impl_cvt.hpp
 * @brief Implementation of type conversion operations for FPEMU floating point emulation library
 *
 * This header provides the implementation of type conversion operations for the FPEMU library.
 * It includes:
 *
 * - Conversion functions from fp64emu_t to other types
 * - Conversion operators to other types
 * - Conversion functions to other types
 *
 * The conversion functions are designed to work across both host and device code
 * through appropriate decorators and provide bit-exact results matching hardware
 * floating point units.    
 */

#include <cuda/__fp/fpemu_common.hpp>
#include <cuda/__fp/fpemu_impl_utils.hpp>
#include <cuda/__fp/fpemu_impl_unpack.hpp>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

namespace impl
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

    // Right shift keeping a sticky (jam) bit: __nv_internal_fp64emu_shr_jam64,
    // shared with the divide/sqrt cores (see fpemu_impl_utils.hpp).

    /// @brief Round a 64-bit fixed-point significand (12 fractional bits) to int32 (CUDA saturation).
    template<fpemu::rounding rm>
    __FPEMU_INTERNAL_DECL__ int32_t __nv_internal_fp64emu_round_to_i32 (bool sign, uint64_t sig)
    {
        const int32_t sat = sign ? (int32_t)0x80000000 : (int32_t)0x7FFFFFFF;
        uint32_t roundIncrement = 0x800;
        if constexpr (rm != fpemu::rounding::rn)
        {
            roundIncrement = 0;
            const bool toward_inf = sign ? (rm == fpemu::rounding::rd)
                                         : (rm == fpemu::rounding::ru);
            if (toward_inf) roundIncrement = 0xFFF;
        }
        uint32_t roundBits = (uint32_t)(sig & 0xFFF);
        sig += roundIncrement;
        if (sig & 0xFFFFF00000000000ULL) return sat;

        uint32_t sig32 = (uint32_t)(sig >> 12);
        if constexpr (rm == fpemu::rounding::rn)
        {
            if (roundBits == 0x800) sig32 &= ~(uint32_t)1;
        }
        uint32_t uz = sign ? (uint32_t)(0u - sig32) : sig32;
        int32_t z = (int32_t)uz;
        if (z && ((z < 0) != sign)) return sat;
        return z;
    }

    /// @brief Round a 64-bit fixed-point significand (12 fractional bits) to uint32 (CUDA saturation).
    template<fpemu::rounding rm>
    __FPEMU_INTERNAL_DECL__ uint32_t __nv_internal_fp64emu_round_to_ui32 (bool sign, uint64_t sig)
    {
        if (sign) return 0; // any negative real saturates to 0

        uint32_t roundIncrement = (rm == fpemu::rounding::rn) ? 0x800
                                : (rm == fpemu::rounding::ru) ? 0xFFF : 0;
        uint32_t roundBits = (uint32_t)(sig & 0xFFF);
        sig += roundIncrement;
        if (sig & 0xFFFFF00000000000ULL) return 0xFFFFFFFFu;

        uint32_t z = (uint32_t)(sig >> 12);
        if constexpr (rm == fpemu::rounding::rn)
        {
            if (roundBits == 0x800) z &= ~(uint32_t)1;
        }
        return z;
    }

    /// @brief Round (sig : sigExtra) to int64 (CUDA saturation).
    template<fpemu::rounding rm>
    __FPEMU_INTERNAL_DECL__ int64_t __nv_internal_fp64emu_round_to_i64 (bool sign, uint64_t sig, uint64_t sigExtra)
    {
        const int64_t sat = sign ? (int64_t)0x8000000000000000ULL
                                 : (int64_t)0x7FFFFFFFFFFFFFFFULL;
        bool increment;
        if constexpr (rm == fpemu::rounding::rn)
        {
            increment = (sigExtra >= 0x8000000000000000ULL);
        }
        else
        {
            const bool toward_inf = sign ? (rm == fpemu::rounding::rd)
                                         : (rm == fpemu::rounding::ru);
            increment = (sigExtra != 0) && toward_inf;
        }
        if (increment)
        {
            ++sig;
            if (!sig) return sat;
            if constexpr (rm == fpemu::rounding::rn)
            {
                if (sigExtra == 0x8000000000000000ULL) sig &= ~(uint64_t)1;
            }
        }
        uint64_t uz = sign ? (uint64_t)(0ULL - sig) : sig;
        int64_t z = (int64_t)uz;
        if (z && ((z < 0) != sign)) return sat;
        return z;
    }

    /// @brief Round (sig : sigExtra) to uint64 (CUDA saturation).
    template<fpemu::rounding rm>
    __FPEMU_INTERNAL_DECL__ uint64_t __nv_internal_fp64emu_round_to_ui64 (bool sign, uint64_t sig, uint64_t sigExtra)
    {
        if (sign) return 0; // any negative real saturates to 0

        bool increment;
        if constexpr (rm == fpemu::rounding::rn)
        {
            increment = (sigExtra >= 0x8000000000000000ULL);
        }
        else
        {
            increment = (rm == fpemu::rounding::ru) && (sigExtra != 0);
        }
        if (increment)
        {
            ++sig;
            if (!sig) return 0xFFFFFFFFFFFFFFFFULL;
            if constexpr (rm == fpemu::rounding::rn)
            {
                if (sigExtra == 0x8000000000000000ULL) sig &= ~(uint64_t)1;
            }
        }
        return sig;
    }

    template<fpemu::rounding rm  = fpemu::rounding::rz>
    __FPEMU_INTERNAL_DECL__  int32_t __nv_internal_fp64emu_fpbits64_to_int (fpbits64_t x)
    {
        const bool    sign = ((uint64_t)x >> 63) != 0;
        const int32_t exp  = (int32_t)(((uint64_t)x >> FP64_MANT_BITS) & 0x7FF);
        uint64_t      sig  = (uint64_t)x & fpemu::MANTISSA_MASK;

        if (exp == 0x7FF && sig) return (int32_t)0x80000000; // NaN -> integer indefinite

        if (exp) sig |= __FPEMU_HIDDEN_64__;
        int32_t shiftDist = 0x427 - exp;
        if (shiftDist > 0) sig = __nv_internal_fp64emu_shr_jam64(sig, (uint32_t)shiftDist);
        return __nv_internal_fp64emu_round_to_i32<rm>(sign, sig);
    } // __nv_internal_fp64emu_fpbits64_to_int

    template<fpemu::rounding rm  = fpemu::rounding::rz>
    __FPEMU_INTERNAL_DECL__  uint32_t __nv_internal_fp64emu_fpbits64_to_uint (fpbits64_t x)
    {
        const bool    sign = ((uint64_t)x >> 63) != 0;
        const int32_t exp  = (int32_t)(((uint64_t)x >> FP64_MANT_BITS) & 0x7FF);
        uint64_t      sig  = (uint64_t)x & fpemu::MANTISSA_MASK;

        if (exp == 0x7FF && sig) return 0x80000000u; // NaN -> integer indefinite

        if (exp) sig |= __FPEMU_HIDDEN_64__;
        int32_t shiftDist = 0x427 - exp;
        if (shiftDist > 0) sig = __nv_internal_fp64emu_shr_jam64(sig, (uint32_t)shiftDist);
        return __nv_internal_fp64emu_round_to_ui32<rm>(sign, sig);
    } // __nv_internal_fp64emu_fpbits64_to_uint

    template<fpemu::rounding rm  = fpemu::rounding::rz>
    __FPEMU_INTERNAL_DECL__  int64_t __nv_internal_fp64emu_fpbits64_to_ll (fpbits64_t x)
    {
        const bool    sign = ((uint64_t)x >> 63) != 0;
        const int32_t exp  = (int32_t)(((uint64_t)x >> FP64_MANT_BITS) & 0x7FF);
        uint64_t      sig  = (uint64_t)x & fpemu::MANTISSA_MASK;

        if (exp == 0x7FF && sig) return (int64_t)0x8000000000000000ULL; // NaN -> integer indefinite

        if (exp) sig |= __FPEMU_HIDDEN_64__;
        int32_t shiftDist = 0x433 - exp;
        uint64_t sig_int, sig_extra;
        if (shiftDist <= 0)
        {
            if (shiftDist < -11) return sign ? (int64_t)0x8000000000000000ULL
                                             : (int64_t)0x7FFFFFFFFFFFFFFFULL;
            sig_int   = sig << (-shiftDist);
            sig_extra = 0;
        }
        else if (shiftDist < 64)
        {
            sig_int   = sig >> shiftDist;
            sig_extra = sig << (-shiftDist & 63);
        }
        else
        {
            sig_int   = 0;
            sig_extra = (shiftDist == 64) ? sig : (uint64_t)(sig != 0);
        }
        return __nv_internal_fp64emu_round_to_i64<rm>(sign, sig_int, sig_extra);
    } // __nv_internal_fp64emu_fpbits64_to_ll

    template<fpemu::rounding rm  = fpemu::rounding::rz>
    __FPEMU_INTERNAL_DECL__  uint64_t __nv_internal_fp64emu_fpbits64_to_ull (fpbits64_t x)
    {
        const bool    sign = ((uint64_t)x >> 63) != 0;
        const int32_t exp  = (int32_t)(((uint64_t)x >> FP64_MANT_BITS) & 0x7FF);
        uint64_t      sig  = (uint64_t)x & fpemu::MANTISSA_MASK;

        if (exp == 0x7FF && sig) return 0x8000000000000000ULL; // NaN -> integer indefinite

        if (exp) sig |= __FPEMU_HIDDEN_64__;
        int32_t shiftDist = 0x433 - exp;
        uint64_t sig_int, sig_extra;
        if (shiftDist <= 0)
        {
            // Negative saturates to 0; positive out-of-range to UINT64_MAX.
            if (shiftDist < -11) return sign ? 0ULL : 0xFFFFFFFFFFFFFFFFULL;
            sig_int   = sig << (-shiftDist);
            sig_extra = 0;
        }
        else if (shiftDist < 64)
        {
            sig_int   = sig >> shiftDist;
            sig_extra = sig << (-shiftDist & 63);
        }
        else
        {
            sig_int   = 0;
            sig_extra = (shiftDist == 64) ? sig : (uint64_t)(sig != 0);
        }
        return __nv_internal_fp64emu_round_to_ui64<rm>(sign, sig_int, sig_extra);
    } // __nv_internal_fp64emu_fpbits64_to_ull

    __FPEMU_INTERNAL_DECL__  float __nv_internal_fp64emu_fpbits64_to_float (fpbits64_t x)
    {
        uint64_t bits = (uint64_t)x;
        uint32_t sign = (uint32_t)(bits >> 63) << 31;
        int32_t  exp  = (int32_t)((bits >> FP64_MANT_BITS) & 0x7FF);
        uint64_t frac = bits & fpemu::MANTISSA_MASK;

        if (exp == 0x7FF) 
        {
            if (frac == 0)
                return fpemu::bit_cast<float>(sign | 0x7F800000u);
            // NaN: preserve sign, set quiet bit, keep upper payload
            uint32_t frac32 = (uint32_t)(frac >> 29) | 0x00400000u;
            return fpemu::bit_cast<float>(sign | 0x7F800000u | frac32);
        }

        // Zero or subnormal double (below float range) → ±0
        if (exp == 0)
            return fpemu::bit_cast<float>(sign);

        int32_t  e_f = exp - (FP64_BIAS - FP32_BIAS);
        uint64_t sig = (1ULL << FP64_MANT_BITS) | frac;

        if (e_f >= 0xFF)
            return fpemu::bit_cast<float>(sign | 0x7F800000u);

        // Number of mantissa bits to discard: 52 - 23 = 29
        constexpr int32_t DROP = FP64_MANT_BITS - FP32_MANT_BITS;

        if (e_f > 0) 
        {
            uint64_t half  = 1ULL << (DROP - 1);
            uint64_t trail = sig & ((1ULL << DROP) - 1);
            uint32_t sig24 = (uint32_t)(sig >> DROP);

            if (trail > half || (trail == half && (sig24 & 1))) 
            {
                sig24++;
                if (sig24 >> (FP32_MANT_BITS + 1)) 
                {
                    sig24 >>= 1;
                    e_f++;
                    if (e_f >= 0xFF)
                        return fpemu::bit_cast<float>(sign | 0x7F800000u);
                }
            }

            uint32_t frac_f = sig24 & ((1u << FP32_MANT_BITS) - 1);
            return fpemu::bit_cast<float>(sign | ((uint32_t)e_f << FP32_MANT_BITS) | frac_f);
        } // if (e_f > 0)

        // Subnormal float output (e_f <= 0)
        int32_t total_shift = DROP + 1 - e_f;

        if (total_shift >= 54)
            return fpemu::bit_cast<float>(sign);

        uint64_t half  = 1ULL << (total_shift - 1);
        uint64_t trail = sig & ((1ULL << total_shift) - 1);
        uint32_t sig_sub = (uint32_t)(sig >> total_shift);

        if (trail > half || (trail == half && (sig_sub & 1)))
            sig_sub++;

        // Overflow to 2^23 naturally becomes exponent=1, frac=0 (min normal)
        return fpemu::bit_cast<float>(sign | sig_sub);
    } // __nv_internal_fp64emu_fpbits64_to_float

    __FPEMU_INTERNAL_DECL__ fpbits64_t __nv_internal_fp64emu_float_to_fpbits64  (float x)
    {
        uint32_t bits = fpemu::bit_cast<uint32_t>(x);
        uint64_t sign = (uint64_t)(bits >> 31) << 63;
        int32_t  exp  = (int32_t)((bits >> FP32_MANT_BITS) & 0xFF);
        uint32_t frac = bits & ((1u << FP32_MANT_BITS) - 1);

        if (exp == 0xFF) 
        {
            if (frac == 0)
                return (fpbits64_t)(sign | ((uint64_t)0x7FF << FP64_MANT_BITS));
            // NaN: preserve sign, set quiet bit, widen payload
            uint64_t d_frac = ((uint64_t)frac << 29) | ((uint64_t)1 << (FP64_MANT_BITS - 1));
            return (fpbits64_t)(sign | ((uint64_t)0x7FF << FP64_MANT_BITS) | d_frac);
        }

        if (exp == 0 && frac == 0)
            return (fpbits64_t)sign;

        // Subnormal float → normalize
        if (exp == 0) 
        {
            int32_t nz = fpemu::__internal_clz((int)frac) - (32 - FP32_MANT_BITS - 1);
            frac = (frac << nz) & ((1u << FP32_MANT_BITS) - 1);
            exp  = 1 - nz;
        }

        // Exact widening conversion
        uint64_t d_exp  = (uint64_t)(exp + (FP64_BIAS - FP32_BIAS));
        uint64_t d_frac = (uint64_t)frac << (FP64_MANT_BITS - FP32_MANT_BITS);
        return (fpbits64_t)(sign | (d_exp << FP64_MANT_BITS) | d_frac);
    } // __nv_internal_fp64emu_float_to_fpbits64

    __FPEMU_INTERNAL_DECL__ 
    fpbits64_t __nv_internal_fp64emu_int_to_fpbits64  (int32_t x)  
    { 
        if (x == 0) return (fpbits64_t)0;

        uint64_t sign  = (x < 0) ? (1ULL << 63) : 0ULL;
        uint32_t abs_x = (uint32_t)((x < 0) ? -(int64_t)x : (int64_t)x);

        int32_t nz        = fpemu::__internal_clz((int)abs_x);
        uint64_t exp      = (uint64_t)(FP64_BIAS + 31 - nz);
        uint64_t mantissa = ((uint64_t)abs_x << (21 + nz)) & fpemu::MANTISSA_MASK;

        return (fpbits64_t)(sign | (exp << FP64_MANT_BITS) | mantissa);
    } // __nv_internal_fp64emu_int_to_fpbits64

    __FPEMU_INTERNAL_DECL__ 
    fpbits64_t __nv_internal_fp64emu_uint_to_fpbits64 (uint32_t x) 
    { 
        if (x == 0) return (fpbits64_t)0;

        int32_t nz        = fpemu::__internal_clz((int)x);
        uint64_t exp      = (uint64_t)(FP64_BIAS + 31 - nz);
        uint64_t mantissa = ((uint64_t)x << (21 + nz)) & fpemu::MANTISSA_MASK;

        return (fpbits64_t)((exp << FP64_MANT_BITS) | mantissa);
    } // __nv_internal_fp64emu_uint_to_fpbits64

    __FPEMU_INTERNAL_DECL__ 
    fpbits64_t __nv_internal_fp64emu_ll_to_fpbits64   (int64_t x)  
    { 
        if (x == 0) return (fpbits64_t)0;

        uint64_t sign = (x < 0) ? (1ULL << 63) : 0ULL;
        uint64_t absA = (x < 0) ? -(uint64_t)x : (uint64_t)x;

        int32_t nz = fpemu::__internal_clzll((int64_t)absA);
        int32_t exp = FP64_BIAS + 63 - nz;

        if (nz >= 11) 
        {
            // <= 53 significant bits: exact
            uint64_t mantissa = (absA << (nz - 11)) & fpemu::MANTISSA_MASK;
            return (fpbits64_t)(sign | ((uint64_t)exp << FP64_MANT_BITS) | mantissa);
        }

        // > 53 significant bits: round to nearest even
        int32_t shift  = 11 - nz;
        uint64_t half  = 1ULL << (shift - 1);
        uint64_t trail = absA & ((1ULL << shift) - 1);
        uint64_t sig53 = absA >> shift;

        if (trail > half || (trail == half && (sig53 & 1))) 
        {
            sig53++;
            if (sig53 >> 53) { sig53 >>= 1; exp++; }
        }

        uint64_t mantissa = sig53 & fpemu::MANTISSA_MASK;
        return (fpbits64_t)(sign | ((uint64_t)exp << FP64_MANT_BITS) | mantissa);
    } // __nv_internal_fp64emu_ll_to_fpbits64

    __FPEMU_INTERNAL_DECL__ 
    fpbits64_t __nv_internal_fp64emu_ull_to_fpbits64  (uint64_t x) 
    {
        if (x == 0) return (fpbits64_t)0;

        int32_t nz = fpemu::__internal_clzll((int64_t)x);
        int32_t exp = FP64_BIAS + 63 - nz;

        if (nz >= 11) 
        {
            // <= 53 significant bits: exact
            uint64_t mantissa = (x << (nz - 11)) & fpemu::MANTISSA_MASK;
            return (fpbits64_t)(((uint64_t)exp << FP64_MANT_BITS) | mantissa);
        }

        // > 53 significant bits: round to nearest even
        int32_t shift  = 11 - nz;
        uint64_t half  = 1ULL << (shift - 1);
        uint64_t trail = x & ((1ULL << shift) - 1);
        uint64_t sig53 = x >> shift;

        if (trail > half || (trail == half && (sig53 & 1))) 
        {
            sig53++;
            if (sig53 >> 53) { sig53 >>= 1; exp++; }
        }

        uint64_t mantissa = sig53 & fpemu::MANTISSA_MASK;
        return (fpbits64_t)(((uint64_t)exp << FP64_MANT_BITS) | mantissa);
    } // __nv_internal_fp64emu_ull_to_fpbits64


    // fpbits64<->uint64 casts
    __FPEMU_INTERNAL_DECL__ uint64_t   __nv_internal_fp64emu_fpbits64_cast_ull (fpbits64_t x) { return x; }
    __FPEMU_INTERNAL_DECL__ fpbits64_t __nv_internal_fp64emu_ull_cast_fpbits64 (uint64_t x)   { return fpbits64_t{x}; }

    // double<->fpbits64 conversions
    __FPEMU_INTERNAL_DECL__ double     __nv_internal_fp64emu_fpbits64_to_double (fpbits64_t x) { return fpemu::bit_cast<double>(x); }
    __FPEMU_INTERNAL_DECL__ fpbits64_t __nv_internal_fp64emu_double_to_fpbits64 (double x)
    { 
        return fpemu::bit_cast<fpbits64_t>(x);
    }

#if __FPEMU_UNPACKED__ == 1

    __FPEMU_INTERNAL_DECL__ uint64_t   __nv_internal_fp64emu_fpbits64_unpacked_cast_ull (fpbits64_unpacked_t x) 
    { 
        fpbits64_t x_packed = __nv_internal_fp64emu_pack(x);
        return __nv_internal_fp64emu_fpbits64_cast_ull(x_packed);
    }

    __FPEMU_INTERNAL_DECL__ fpbits64_unpacked_t __nv_internal_fp64emu_ull_cast_fpbits64_unpacked (uint64_t x)
    { 
        fpbits64_t x_packed = __nv_internal_fp64emu_ull_cast_fpbits64(x);
        return __nv_internal_fp64emu_unpack(x_packed);
    }

    /**
     * @brief Convert a fpbits64_unpacked_t to a double
     * 
     * This function converts a fpbits64_unpacked_t to a double.
     * 
     * @param x The fpbits64_unpacked_t to convert
     * @return The converted double
     */
    template<fpemu::rounding rm   = fpemu::rounding::def,
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__ double __nv_internal_fp64emu_fpbits64_unpacked_to_double (fpbits64_unpacked_t x) 
    { 
        fpbits64_t x_packed = __nv_internal_fp64emu_pack<rm>(x);
        return __nv_internal_fp64emu_fpbits64_to_double(x_packed); 
    }

    /**
     * @brief Convert a double to a fpbits64_unpacked_t
     * 
     * This function converts a double to a fpbits64_unpacked_t.
     * 
     * @param x The double to convert
     * @return The converted fpbits64_unpacked_t
     */
    template<fpemu::rounding rm   = fpemu::rounding::def,
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__ fpbits64_unpacked_t __nv_internal_fp64emu_double_to_fpbits64_unpacked (double x)
    { 
        fpbits64_t x_packed = __nv_internal_fp64emu_double_to_fpbits64(x);
        return __nv_internal_fp64emu_unpack(x_packed);
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
    static constexpr int32_t __FP64EMU_CVT_NAN_EXP = 0x0007ff00;
    static constexpr int32_t __FP64EMU_CVT_INF_EXP = 0x00007ff0;

    template<fpemu::rounding rm   = fpemu::rounding::def,
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__  int32_t __nv_internal_fp64emu_fpbits64_unpacked_to_int (fpbits64_unpacked_t x)
    {
        const bool    sign = (x.sign != 0);
        const int32_t exp  = (int32_t)x.exponent;
        if (exp == __FP64EMU_CVT_NAN_EXP) return (int32_t)0x80000000;                       // NaN -> indefinite
        if (exp == __FP64EMU_CVT_INF_EXP) return sign ? (int32_t)0x80000000 : (int32_t)0x7FFFFFFF;
        uint64_t sig = x.mantissa >> EXTRA_BITS;   // 53-bit significand (implicit at 52), 0 for zero
        int32_t  shiftDist = 0x427 - exp;
        if (shiftDist > 0) sig = __nv_internal_fp64emu_shr_jam64(sig, (uint32_t)shiftDist);
        return __nv_internal_fp64emu_round_to_i32<rm>(sign, sig);
    }
    template<fpemu::rounding rm   = fpemu::rounding::def,
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__  uint32_t __nv_internal_fp64emu_fpbits64_unpacked_to_uint (fpbits64_unpacked_t x)
    {
        const bool    sign = (x.sign != 0);
        const int32_t exp  = (int32_t)x.exponent;
        if (exp == __FP64EMU_CVT_NAN_EXP) return 0x80000000u;                               // NaN -> indefinite
        if (exp == __FP64EMU_CVT_INF_EXP) return sign ? 0u : 0xFFFFFFFFu;
        uint64_t sig = x.mantissa >> EXTRA_BITS;
        int32_t  shiftDist = 0x427 - exp;
        if (shiftDist > 0) sig = __nv_internal_fp64emu_shr_jam64(sig, (uint32_t)shiftDist);
        return __nv_internal_fp64emu_round_to_ui32<rm>(sign, sig);
    }

    template<fpemu::rounding rm   = fpemu::rounding::def,
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__  int64_t __nv_internal_fp64emu_fpbits64_unpacked_to_ll (fpbits64_unpacked_t x)
    {
        const bool    sign = (x.sign != 0);
        const int32_t exp  = (int32_t)x.exponent;
        if (exp == __FP64EMU_CVT_NAN_EXP) return (int64_t)0x8000000000000000ULL;            // NaN -> indefinite
        if (exp == __FP64EMU_CVT_INF_EXP) return sign ? (int64_t)0x8000000000000000ULL
                                                      : (int64_t)0x7FFFFFFFFFFFFFFFULL;
        uint64_t sig = x.mantissa >> EXTRA_BITS;
        int32_t  shiftDist = 0x433 - exp;
        uint64_t sig_int, sig_extra;
        if (shiftDist <= 0)
        {
            if (shiftDist < -11) return sign ? (int64_t)0x8000000000000000ULL
                                             : (int64_t)0x7FFFFFFFFFFFFFFFULL;
            sig_int   = sig << (-shiftDist);
            sig_extra = 0;
        }
        else if (shiftDist < 64)
        {
            sig_int   = sig >> shiftDist;
            sig_extra = sig << (-shiftDist & 63);
        }
        else
        {
            sig_int   = 0;
            sig_extra = (shiftDist == 64) ? sig : (uint64_t)(sig != 0);
        }
        return __nv_internal_fp64emu_round_to_i64<rm>(sign, sig_int, sig_extra);
    }

    template<fpemu::rounding rm   = fpemu::rounding::def,
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__  uint64_t __nv_internal_fp64emu_fpbits64_unpacked_to_ull (fpbits64_unpacked_t x)
    {
        const bool    sign = (x.sign != 0);
        const int32_t exp  = (int32_t)x.exponent;
        if (exp == __FP64EMU_CVT_NAN_EXP) return 0x8000000000000000ULL;                     // NaN -> indefinite
        if (exp == __FP64EMU_CVT_INF_EXP) return sign ? 0ULL : 0xFFFFFFFFFFFFFFFFULL;
        uint64_t sig = x.mantissa >> EXTRA_BITS;
        int32_t  shiftDist = 0x433 - exp;
        uint64_t sig_int, sig_extra;
        if (shiftDist <= 0)
        {
            if (shiftDist < -11) return sign ? 0ULL : 0xFFFFFFFFFFFFFFFFULL;
            sig_int   = sig << (-shiftDist);
            sig_extra = 0;
        }
        else if (shiftDist < 64)
        {
            sig_int   = sig >> shiftDist;
            sig_extra = sig << (-shiftDist & 63);
        }
        else
        {
            sig_int   = 0;
            sig_extra = (shiftDist == 64) ? sig : (uint64_t)(sig != 0);
        }
        return __nv_internal_fp64emu_round_to_ui64<rm>(sign, sig_int, sig_extra);
    }

    template<fpemu::rounding rm   = fpemu::rounding::def,
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__  float __nv_internal_fp64emu_fpbits64_unpacked_to_float (fpbits64_unpacked_t x)
    {
        fpbits64_t x_packed = __nv_internal_fp64emu_pack<rm>(x);
        return __nv_internal_fp64emu_fpbits64_to_float(x_packed);
    }

    template<fpemu::rounding rm   = fpemu::rounding::def,
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__ fpbits64_unpacked_t __nv_internal_fp64emu_float_to_fpbits64_unpacked  (float x)     
    { 
        fpbits64_t x_packed = __nv_internal_fp64emu_float_to_fpbits64(x);
        return __nv_internal_fp64emu_unpack(x_packed);
    }

    template<fpemu::rounding rm   = fpemu::rounding::def,
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__ fpbits64_unpacked_t __nv_internal_fp64emu_int_to_fpbits64_unpacked  (int32_t x)     
    { 
        fpbits64_t x_packed = __nv_internal_fp64emu_int_to_fpbits64(x);
        return __nv_internal_fp64emu_unpack(x_packed);
    }

    template<fpemu::rounding rm   = fpemu::rounding::def,
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__ fpbits64_unpacked_t __nv_internal_fp64emu_uint_to_fpbits64_unpacked  (uint32_t x)     
    { 
        fpbits64_t x_packed = __nv_internal_fp64emu_uint_to_fpbits64(x);
        return __nv_internal_fp64emu_unpack(x_packed);
    }   

    template<fpemu::rounding rm   = fpemu::rounding::def,
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__ fpbits64_unpacked_t __nv_internal_fp64emu_ull_to_fpbits64_unpacked  (uint64_t x)     
    { 
        fpbits64_t x_packed = __nv_internal_fp64emu_ull_to_fpbits64(x);
        return __nv_internal_fp64emu_unpack(x_packed);
    }

    template<fpemu::rounding rm   = fpemu::rounding::def,
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__ fpbits64_unpacked_t __nv_internal_fp64emu_ll_to_fpbits64_unpacked  (int64_t x)     
    { 
        fpbits64_t x_packed = __nv_internal_fp64emu_ll_to_fpbits64(x);
        return __nv_internal_fp64emu_unpack(x_packed);
    }

#endif // __FPEMU_UNPACKED__ == 1

} // namespace impl

// ============================================================================
// Builtin declarations/implementations for conversion operations
// ============================================================================
#if defined(__FPEMU_INLINE__)
#if (__FPEMU_PACKED_VIA_UNPACKED__ == 1)
// Packed-via-unpacked (testing): route the packed conversion builtins through the
// unpacked cores. fp->int unpack(x) then the rounding-aware unpacked core; fp->fp
// goes through the universal unpack/pack; int/fp->fp builds the unpacked value and
// packs (rn -- the integer/widening conversions are exact or already rounded).
// The pure bit-reinterpret casts (fpbits64<->ull) are NOT rerouted: they must
// preserve the exact bit pattern and have no unpacked-core equivalent.
__FPEMU_BUILTIN_DECL__ double     __nv_fp64emu_to_double (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_double (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ float      __nv_fp64emu_to_float  (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_float (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rn (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_int<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rz (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_int<fpemu::rounding::rz> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_ru (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_int<fpemu::rounding::ru> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rd (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_int<fpemu::rounding::rd> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rn (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_uint<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rz (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_uint<fpemu::rounding::rz> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_ru (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_uint<fpemu::rounding::ru> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rd (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_uint<fpemu::rounding::rd> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rn (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ll<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rz (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ll<fpemu::rounding::rz> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_ru (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ll<fpemu::rounding::ru> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rd (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ll<fpemu::rounding::rd> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rn (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ull<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rz (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ull<fpemu::rounding::rz> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_ru (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ull<fpemu::rounding::ru> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rd (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ull<fpemu::rounding::rd> (impl::__nv_internal_fp64emu_unpack (x)); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_double (double x)   { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_double_to_fpbits64_unpacked (x)); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_float  (float x)    { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_float_to_fpbits64_unpacked (x)); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_int    (int32_t x)  { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_int_to_fpbits64_unpacked (x)); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_uint   (uint32_t x) { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_uint_to_fpbits64_unpacked (x)); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ll     (int64_t x)  { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_ll_to_fpbits64_unpacked (x)); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ull    (uint64_t x) { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_ull_to_fpbits64_unpacked (x)); }
#else
__FPEMU_BUILTIN_DECL__ double     __nv_fp64emu_to_double (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_double (x); }
__FPEMU_BUILTIN_DECL__ float      __nv_fp64emu_to_float  (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_float (x); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rn (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_int<fpemu::rounding::rn> (x); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rz (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_int<fpemu::rounding::rz> (x); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_ru (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_int<fpemu::rounding::ru> (x); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rd (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_int<fpemu::rounding::rd> (x); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rn (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_uint<fpemu::rounding::rn> (x); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rz (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_uint<fpemu::rounding::rz> (x); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_ru (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_uint<fpemu::rounding::ru> (x); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rd (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_uint<fpemu::rounding::rd> (x); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rn (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_ll<fpemu::rounding::rn> (x); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rz (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_ll<fpemu::rounding::rz> (x); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_ru (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_ll<fpemu::rounding::ru> (x); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rd (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_ll<fpemu::rounding::rd> (x); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rn (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_ull<fpemu::rounding::rn> (x); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rz (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_ull<fpemu::rounding::rz> (x); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_ru (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_ull<fpemu::rounding::ru> (x); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rd (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_to_ull<fpemu::rounding::rd> (x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_double (double x)   { return impl::__nv_internal_fp64emu_double_to_fpbits64 (x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_float  (float x)    { return impl::__nv_internal_fp64emu_float_to_fpbits64 (x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_int    (int32_t x)  { return impl::__nv_internal_fp64emu_int_to_fpbits64 (x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_uint   (uint32_t x) { return impl::__nv_internal_fp64emu_uint_to_fpbits64 (x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ll     (int64_t x)  { return impl::__nv_internal_fp64emu_ll_to_fpbits64 (x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ull    (uint64_t x) { return impl::__nv_internal_fp64emu_ull_to_fpbits64 (x); }
#endif // __FPEMU_PACKED_VIA_UNPACKED__
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_fpbits64_cast_ull  (fpbits64_t x) { return impl::__nv_internal_fp64emu_fpbits64_cast_ull (x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_ull_cast_fpbits64  (uint64_t x)   { return impl::__nv_internal_fp64emu_ull_cast_fpbits64 (x); }
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpack  (fpbits64_t a)          { return impl::__nv_internal_fp64emu_unpack(a); }
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_rn (fpbits64_unpacked_t a) { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn>(a); }
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_rz (fpbits64_unpacked_t a) { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rz>(a); }
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_ru (fpbits64_unpacked_t a) { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::ru>(a); }
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_rd (fpbits64_unpacked_t a) { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rd>(a); }
__FPEMU_BUILTIN_DECL__ int32_t  __nv_fp64emu_unpacked_to_int            (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_int<fpemu::rounding::rz>(x); }
__FPEMU_BUILTIN_DECL__ uint32_t __nv_fp64emu_unpacked_to_uint           (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_uint<fpemu::rounding::rz>(x); }
__FPEMU_BUILTIN_DECL__ int64_t  __nv_fp64emu_unpacked_to_ll             (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ll<fpemu::rounding::rz>(x); }
__FPEMU_BUILTIN_DECL__ uint64_t __nv_fp64emu_unpacked_to_ull            (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ull<fpemu::rounding::rz>(x); }
__FPEMU_BUILTIN_DECL__ float    __nv_fp64emu_unpacked_to_float          (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_float(x); }
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_to_double         (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_double(x); }
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_high_to_double(fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_double<fpemu::rounding::rn, fp64emu_accuracy::high>(x); }
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_mid_to_double     (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_double<fpemu::rounding::rn, fp64emu_accuracy::mid>(x); }
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_low_to_double    (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_double<fpemu::rounding::rn, fp64emu_accuracy::low>(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_int             (int32_t x)  { return impl::__nv_internal_fp64emu_int_to_fpbits64_unpacked(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_uint            (uint32_t x) { return impl::__nv_internal_fp64emu_uint_to_fpbits64_unpacked(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_ll              (int64_t x)  { return impl::__nv_internal_fp64emu_ll_to_fpbits64_unpacked(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_ull             (uint64_t x) { return impl::__nv_internal_fp64emu_ull_to_fpbits64_unpacked(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_float           (float x)    { return impl::__nv_internal_fp64emu_float_to_fpbits64_unpacked(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_double          (double x)   { return impl::__nv_internal_fp64emu_double_to_fpbits64_unpacked(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_from_double (double x)   { return impl::__nv_internal_fp64emu_double_to_fpbits64_unpacked<fpemu::rounding::rn, fp64emu_accuracy::high>(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_from_double      (double x)   { return impl::__nv_internal_fp64emu_double_to_fpbits64_unpacked<fpemu::rounding::rn, fp64emu_accuracy::mid>(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_from_double     (double x)   { return impl::__nv_internal_fp64emu_double_to_fpbits64_unpacked<fpemu::rounding::rn, fp64emu_accuracy::low>(x); }
__FPEMU_BUILTIN_DECL__ uint64_t            __nv_fp64emu_unpacked_fpbits64_cast_ull           (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_fpbits64_unpacked_cast_ull(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_ull_cast_fpbits64           (uint64_t x) { return impl::__nv_internal_fp64emu_ull_cast_fpbits64_unpacked(x); }
#endif
#else
__FPEMU_BUILTIN_DECL__ double     __nv_fp64emu_to_double (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_double (double x);
__FPEMU_BUILTIN_DECL__ float      __nv_fp64emu_to_float  (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_float  (float x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_int    (int32_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_uint   (uint32_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ll     (int64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ull    (uint64_t x);
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_fpbits64_cast_ull  (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_ull_cast_fpbits64  (uint64_t x);
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rn (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rz (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_ru (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rd (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rn (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rz (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_ru (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rd (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rn (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rz (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_ru (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rd (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rn (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rz (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_ru (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rd (fpbits64_t x);
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpack  (fpbits64_t a);
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_rn (fpbits64_unpacked_t a);
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_rz (fpbits64_unpacked_t a);
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_ru (fpbits64_unpacked_t a);
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_rd (fpbits64_unpacked_t a);
__FPEMU_BUILTIN_DECL__ int32_t  __nv_fp64emu_unpacked_to_int            (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ uint32_t __nv_fp64emu_unpacked_to_uint           (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ int64_t  __nv_fp64emu_unpacked_to_ll             (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ uint64_t __nv_fp64emu_unpacked_to_ull            (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ float    __nv_fp64emu_unpacked_to_float          (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_to_double         (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_high_to_double(fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_mid_to_double     (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_low_to_double    (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_int             (int32_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_uint            (uint32_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_ll              (int64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_ull             (uint64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_float           (float x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_double          (double x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_from_double (double x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_from_double      (double x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_from_double     (double x);
__FPEMU_BUILTIN_DECL__ uint64_t            __nv_fp64emu_unpacked_fpbits64_cast_ull           (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_ull_cast_fpbits64           (uint64_t x);
#endif
#endif // __FPEMU_INLINE__

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_CVT_HPP__

#if defined(__FPEMU_API_CLASSES_DEFINED__) && !defined(__FPEMU_CVT_API_MERGED__)
#define __FPEMU_CVT_API_MERGED__
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_cvt_api.hpp)
// ============================================================================

    // Type conversion to fp64emu_t with other method
    template<fp64emu_accuracy m_src> 
    template<fp64emu_accuracy m_dst> 
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m_src>::operator fp64emu_t<m_dst>() const 
        { 
            return fp64emu_t<m_dst>(fpbits64_construct, bits); 
        }

#if __FPEMU_UNPACKED__ == 1
    // Type conversion from fp64emu_t to fp64emu_unpacked_t
    template<fp64emu_accuracy m_src> 
    template<fp64emu_accuracy m_dst> 
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m_src>::operator fp64emu_unpacked_t<m_dst>() const 
        { 
            fpbits64_unpacked_t bits_unpacked = __nv_fp64emu_unpack(bits);
            return fp64emu_unpacked_t<m_dst>(fpbits64_construct, bits_unpacked); 
        }
#endif // __FPEMU_UNPACKED__ == 1

    /*
    // Type conversions from other types to fp64emu_t 
    */
    // from double
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m>::fp64emu_t(double d){ 
        bits = __nv_fp64emu_from_double (d); }
    // from float
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m>::fp64emu_t(float d) { 
        bits = __nv_fp64emu_from_float (d); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m> __float2double  (float x) { 
        return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_from_float (x)); }
    // from int32_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m>::fp64emu_t(int32_t d) { 
        bits = __nv_fp64emu_from_int (d); }
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m> __int2double  (int32_t x) { 
        return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_from_int (x)); }
    // from uint32_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m>::fp64emu_t(uint32_t d) { 
        bits = __nv_fp64emu_from_uint (d); }
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m> __uint2double (uint32_t x) { 
        return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_from_uint (x)); }
    // from int64_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m>::fp64emu_t(int64_t d) { 
        bits = __nv_fp64emu_from_ll (d); }
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m> __ll2double (int64_t x) { 
        return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_from_ll (x)); }
    // from uint64_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m>::fp64emu_t(uint64_t d) { 
        bits = __nv_fp64emu_from_ull (d); }
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m> __ull2double(uint64_t x) { 
        return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_from_ull (x)); }

    /*
    // Type conversions from fp64emu_t to other types
    */
    // to double
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m>::operator double() const { 
        return __nv_fp64emu_to_double (bits); }
    // to float
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m>::operator float()  const { 
        return __nv_fp64emu_to_float (bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline float  __double2float (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_float (x.bits); }
    // to int32_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m>::operator int32_t()  const { 
        return __nv_fp64emu_to_int_rz (bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline int32_t __double2int_rn (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_int_rn (x.bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline int32_t __double2int_rz (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_int_rz (x.bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline int32_t __double2int_ru (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_int_ru (x.bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline int32_t __double2int_rd (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_int_rd (x.bits); }
    // to uint32_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m>::operator uint32_t() const { 
        return __nv_fp64emu_to_uint_rz (bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline uint32_t __double2uint_rn (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_uint_rn (x.bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline uint32_t __double2uint_rz (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_uint_rz (x.bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline uint32_t __double2uint_ru (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_uint_ru (x.bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline uint32_t __double2uint_rd (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_uint_rd (x.bits); }
    // to int64_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m>::operator int64_t()  const { 
        return __nv_fp64emu_to_ll_rz (bits); }    
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline int64_t __double2ll_rn (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_ll_rn (x.bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline int64_t __double2ll_rz (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_ll_rz (x.bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline int64_t __double2ll_ru (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_ll_ru (x.bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline int64_t __double2ll_rd (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_ll_rd (x.bits); }    
    // to uint64_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t<m>::operator uint64_t() const { 
        return __nv_fp64emu_to_ull_rz (bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline uint64_t __double2ull_rn (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_ull_rn (x.bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline uint64_t __double2ull_rz (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_ull_rz (x.bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline uint64_t __double2ull_ru (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_ull_ru (x.bits); }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ inline uint64_t __double2ull_rd (fp64emu_t<m> x) { 
        return __nv_fp64emu_to_ull_rd (x.bits); }

#if __FPEMU_UNPACKED__ == 1

    // Type conversion from fp64emu_unpacked_t with other method
    template<fp64emu_accuracy m_src> 
    template<fp64emu_accuracy m_dst> 
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m_src>::operator fp64emu_unpacked_t<m_dst>() const 
        { 
            return fp64emu_unpacked_t<m_dst>(fpbits64_construct, bits); 
        }

    // Type conversion from fp64emu_unpacked_t to fp64emu_t
    template<fp64emu_accuracy m_src> 
    template<fp64emu_accuracy m_dst> 
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m_src>::operator fp64emu_t<m_dst>() const 
        { 
            fpbits64_t bits_packed = __nv_fp64emu_pack_rn(bits);
            return fp64emu_t<m_dst>(fpbits64_construct, bits_packed); 
        }

    /*
    // Type conversions from other types to fp64emu_unpacked_t 
    */
    // from double 
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>::fp64emu_unpacked_t(double d) 
    { 
    if      constexpr (m == fp64emu_accuracy::high) { bits = __nv_fp64emu_unpacked_high_from_double(d); }
        else if constexpr (m == fp64emu_accuracy::mid)      { bits = __nv_fp64emu_unpacked_mid_from_double(d); }
        else                                      { bits = __nv_fp64emu_unpacked_from_double(d); }
    }
    // from float 
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>::fp64emu_unpacked_t(float d) { 
        bits = __nv_fp64emu_unpacked_from_float (d);  }
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>  __float2double (float x) { 
        return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_from_float (x)); }
    // from int32_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>::fp64emu_unpacked_t(int32_t d) { 
        bits = __nv_fp64emu_unpacked_from_int (d); }
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>  __int2double (int32_t x) { 
        return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_from_int (x)); }
    // from uint32_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>::fp64emu_unpacked_t(uint32_t d) { 
        bits = __nv_fp64emu_unpacked_from_uint (d); }
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>  __uint2double (uint32_t x) { 
        return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_from_uint (x)); }
    // from int64_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>::fp64emu_unpacked_t(int64_t d) { 
        bits = __nv_fp64emu_unpacked_from_ll (d); }
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>  __ll2double (int64_t x)  { 
        return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_from_ll (x)); }
    // from uint64_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>::fp64emu_unpacked_t(uint64_t d) { 
        bits = __nv_fp64emu_unpacked_from_ull (d); }
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>  __ull2double (uint64_t x) { 
        return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_from_ull (x)); }

    /*
    // Conversion operators from fp64emu_unpacked_t to other types
    */
    // to double
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>::operator double() const 
    { 
if      constexpr (m == fp64emu_accuracy::high) { return __nv_fp64emu_unpacked_high_to_double(bits); }
       else if constexpr (m == fp64emu_accuracy::mid)      { return __nv_fp64emu_unpacked_mid_to_double(bits); }
       else                                      { return __nv_fp64emu_unpacked_to_double(bits); }
    }
    // to float
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>::operator float()  const { 
        return __nv_fp64emu_unpacked_to_float (bits); }
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline float    __double2float   (fp64emu_unpacked_t<m> x) { 
        return __nv_fp64emu_unpacked_to_float (x.bits); }
    // to int32_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>::operator int32_t()  const { 
        return __nv_fp64emu_unpacked_to_int (bits); }
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline int32_t  __double2int_rz  (fp64emu_unpacked_t<m> x) { 
        return __nv_fp64emu_unpacked_to_int (x.bits); }
    // to uint32_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>::operator uint32_t() const { 
        return __nv_fp64emu_unpacked_to_uint (bits); }
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline uint32_t __double2uint_rz (fp64emu_unpacked_t<m> x) { 
        return __nv_fp64emu_unpacked_to_uint (x.bits); }
    // to int64_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>::operator int64_t()  const { 
        return __nv_fp64emu_unpacked_to_ll (bits); } 
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline int64_t  __double2ll_rz   (fp64emu_unpacked_t<m> x) { 
        return __nv_fp64emu_unpacked_to_ll (x.bits); }
    // to uint64_t
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t<m>::operator uint64_t() const { 
        return __nv_fp64emu_unpacked_to_ull (bits); } 
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ inline uint64_t __double2ull_rz  (fp64emu_unpacked_t<m> x) { 
        return __nv_fp64emu_unpacked_to_ull (x.bits); }
    template<typename To, fp64emu_accuracy m2>
        __FPEMU_HOST_DEVICE_DECL__ inline To bit_cast(const fp64emu_unpacked_t<m2>& from)
        {
            // Pack the unpacked value to get IEEE-754 representation
            fpbits64_t packed = __nv_fp64emu_pack_rn(from.bits);
            return fpemu::bit_cast<To>(packed);
        }
#endif // __FPEMU_UNPACKED__ == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_CVT_API_MERGED__
