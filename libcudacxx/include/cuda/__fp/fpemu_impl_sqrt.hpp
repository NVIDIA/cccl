//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_SQRT_H
#define _CUDA___FP_FPEMU_IMPL_SQRT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file fpemu_dsqrt_impl.hpp
 * @brief Implementation of double-precision square root operations for FPEMU floating point emulation library
 *
 * This header provides the implementation of double-precision square root operations for the FPEMU library.
 * It includes:
 *
 * - Square root functions for fp64emu_t
 * - Square root operators for fp64emu_t
 * - Square root functions to other types
 *
 * The square root functions are designed to work across both host and device code
 * through appropriate decorators and provide bit-exact results matching hardware
 * floating point units.
 */

#include <cuda/__fp/fpemu_common.hpp>
#include <cuda/__fp/fpemu_impl_utils.hpp>
#include <cuda/__fp/fpemu_impl_unpack.hpp>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

#if !(defined (__CUDA_ARCH__))
extern "C" double sqrt(double x);
extern "C" float  sqrtf(float x);   // host seed for the reciprocal-sqrt builtin
#endif

namespace impl
{
    // ========================================================================
    // Native fp64 square root.
    //
    // Split sign/exp/mantissa, normalize subnormals, form a 32-bit reciprocal
    // square root of the significand, then refine with fixed-point integer
    // remainder arithmetic and round/pack. The reciprocal-sqrt seed comes from
    // the fp32 rsqrt builtin (rsqrt.approx on device, 1/sqrtf on host) and is
    // refined by a single Newton step plus a trim that guarantees a strict
    // underestimate, which the remainder refinement needs for correctly-rounded
    // results.
    // ========================================================================

    /// @brief Approximation of 2^47 / sqrt(a / 2^odd_exp), a in [2^31, 2^32),
    ///        in [2^31, 2^32). Seeded by the fp32 rsqrt builtin, refined by one
    ///        Newton step, then trimmed to a strict underestimate so that the
    ///        derived root stays a lower bound (keeps the remainder unsigned).
    __FPEMU_INTERNAL_DECL__ uint32_t __nv_internal_fp64emu_sqrt_recip_sqrt32 (uint32_t odd_exp, 
                                                                              uint32_t a)
    {
        // Seed: r ~ 2^47 * rsqrt(a / 2^odd_exp). Halving the radicand for the
        // odd-exponent case folds the sqrt(2) factor into the same 2^47 scale.
        float    af = odd_exp ? (float)a * 0.5f : (float)a;
    #if defined(__CUDA_ARCH__)
        float    rf;
        asm ("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rf) : "f"(af));
    #else
        float    rf = 1.0f / sqrtf(af);                      // host fallback
    #endif
        uint64_t r  = (uint64_t)(rf * 140737488355328.0f);   // * 2^47

        if (r < 0x80000000ULL) r = 0x80000000ULL;
        if (r > 0xFFFFFFFFULL) r = 0xFFFFFFFFULL;

        // One Newton step (reciprocal-sqrt): r <- r*(3K - a*r^2)/(2K), K = 2^(94+odd).
        uint64_t r2      = r * r;                              // < 2^64
        uint64_t a_r2_hi = __nv_internal_fp64emu_mulhi64((uint64_t)a, r2);   // a*r^2 >> 64
        uint64_t a_r2_lo = (uint64_t)a * r2;                 // low 64 bits (wraps)
        uint64_t k3_hi   = 3ULL << (30 + odd_exp);            // (3 * 2^(94+odd)) >> 64
        uint64_t t_lo    = 0ULL - a_r2_lo;
        uint64_t borrow  = (a_r2_lo != 0) ? 1u : 0u;
        uint64_t t_hi    = k3_hi - a_r2_hi - borrow;          // t = 3K - a*r^2 (128-bit)
        // r' = (r * t) >> (95 + odd); pre-shift t by 33 so the product fits 128 bits.
        uint64_t t_s    = (t_hi << 31) | (t_lo >> 33);
        uint64_t pr_hi  = __nv_internal_fp64emu_mulhi64(r, t_s);
        uint64_t pr_lo  = r * t_s;
        r = (pr_hi << (2 - odd_exp)) | (pr_lo >> (62 + odd_exp));
        if (r < 0x80000000ULL) r = 0x80000000ULL;
        if (r > 0xFFFFFFFFULL) r = 0xFFFFFFFFULL;

        // Trim to a strict underestimate of 2^47*2^(odd/2)/sqrt(a): a*r^2 <= 2^(94+odd).
        while (r > 0x80000000ULL)
        {
            uint64_t rr2 = r * r;
            uint64_t hi  = __nv_internal_fp64emu_mulhi64((uint64_t)a, rr2);
            uint64_t lo  = (uint64_t)a * rr2;
            uint64_t thr = 1ULL << (30 + odd_exp);
            if ((hi > thr) || (hi == thr && lo != 0)) --r; else break;
        }

        return (uint32_t)r;
    } // __nv_internal_fp64emu_sqrt_recip_sqrt32

    // Forward declaration: the unpacked sqrt core is defined below, but the packed
    // wrapper references it for the packed-via-unpacked (testing) path.
    template<fp64emu_accuracy meth>
    __FPEMU_INTERNAL_DECL__
    fpbits64_unpacked_t __nv_internal_fp64emu_dsqrt_unpacked(fpbits64_unpacked_t x);

    template<fpemu::rounding rm   = fpemu::rounding::def, 
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__ fpbits64_t __nv_internal_fp64emu_dsqrt(fpbits64_t x)
    {
    #if (__FPEMU_PACKED_VIA_UNPACKED__ == 1)
        // Packed-via-unpacked (testing): pack(dsqrt_unpacked(unpack(x))). The
        // dsqrt_unpacked core handles special operands and method selection; the
        // universal unpack/pack are the shared prologue/epilogue. Rounding is
        // applied only at pack, preserving the packed builtins' per-mode behavior.
        {
            fpbits64_unpacked_t a = __nv_internal_fp64emu_unpack(x);
            fpbits64_unpacked_t r = __nv_internal_fp64emu_dsqrt_unpacked<meth>(a);
            return __nv_internal_fp64emu_pack<rm>(r);
        }
    #else

        const uint64_t ui64_x = (uint64_t)x;

        bool           sign_x = (ui64_x >> 63) != 0;
        int32_t        exp_x  = (int32_t)((ui64_x >> 52) & 0x7FF);
        uint64_t       mant_x = ui64_x & __FPEMU_MANT_64__;

        // -------- special operands (NaN / Inf / negative / zero) --------
        if (exp_x == 0x7FF)
        {
            if (mant_x)  return (fpbits64_t)(ui64_x | __FPEMU_QNAN_BIT_64__);   // NaN -> quiet NaN
            if (!sign_x) return (fpbits64_t)ui64_x;                            // +inf -> +inf
            return (fpbits64_t)__FPEMU_DEFNAN_64__;                          // sqrt(-inf) -> NaN
        }
        if (sign_x)
        {
            if (!(exp_x | (int32_t)(mant_x != 0))) return (fpbits64_t)ui64_x;   // -0 -> -0
            return (fpbits64_t)__FPEMU_DEFNAN_64__;                          // sqrt(negative) -> NaN
        }
        if (!exp_x)
        {
            if (!mant_x) return (fpbits64_t)ui64_x;                             // +0 -> +0

            int mant_shft = fpemu::__internal_clzll((int64_t)mant_x) - 11;           // normalize subnormal

            exp_x   = 1 - mant_shft;
            mant_x  = mant_x << mant_shft;
        }

        // -------- fixed-point reciprocal-sqrt root --------
        int32_t exp_z = ((exp_x - 0x3FF) >> 1) + 0x3FE;
        exp_x        &= 1;
        mant_x       |= __FPEMU_HIDDEN_64__;

        uint32_t mant32_x  = (uint32_t)(mant_x >> 21);
        uint32_t rcp32     = __nv_internal_fp64emu_sqrt_recip_sqrt32((uint32_t)exp_x, mant32_x);
        uint32_t mant32_z  = (uint32_t)(((uint64_t)mant32_x * rcp32) >> 32);

        if (exp_x) { mant_x <<= 8; mant32_z >>= 1; } else { mant_x <<= 9; }

        uint64_t rem64    = mant_x - (uint64_t)mant32_z * mant32_z;
        uint32_t q32      = (uint32_t)(((uint32_t)(rem64 >> 2) * (uint64_t)rcp32) >> 32);
        // form mantissa: (1 << 52) + (mant32_z << 21) + (q32 << 3)
        uint64_t mant64_z = ((uint64_t)mant32_z << 32 | (1u << 5)) + ((uint64_t)q32 << 3);

        // Refine if the root is close to a rounding boundary (exact remainder).
        if ((mant64_z & 0x1FF) < 0x22)
        {
            mant64_z               &= ~(uint64_t)0x3F;
            uint64_t mant64_z_shftd = mant64_z >> 6;
            rem64                   = (mant_x << 52) - mant64_z_shftd * mant64_z_shftd;

            if (rem64 & __FPEMU_SIGN_64__) { --mant64_z; }
            else { if (rem64) mant64_z |= 1; }
        }

        return __nv_internal_fp64emu_round_pack<rm>(false, exp_z, mant64_z);
    #endif // __FPEMU_PACKED_VIA_UNPACKED__
    } // __nv_internal_fp64emu_dsqrt

    // Unpacked operation
    template<fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_unpacked_t __nv_internal_fp64emu_dsqrt_unpacked(fpbits64_unpacked_t x)
    {
        // ---- True unpacked square root --------------------------------------
        // Operates directly on the fully-accurate unpacked operand (no operand
        // pack, no legacy packed kernel). The full unpack already normalized
        // denormals and encoded inf/nan, so the significand is mantissa>>EXTRA_BITS
        // (implicit bit at 52) and the exponent is the IEEE-biased value (same as
        // the packed core's post-normalization exponent, so the parity bit and the
        // halved result exponent are identical). The proven fixed-point
        // reciprocal-sqrt root is computed exactly as the packed core, then
        // expressed on the universal unpacked scale (implicit bit at 61, sticky LSB)
        // so the full pack performs the single correctly-rounded finalization.
        // sqrt is correctly rounded for every accuracy level (no mid/low variant).
        constexpr int32_t NAN_EXP = 0x0007ff00;
        constexpr int32_t INF_EXP = 0x00007ff0;

        const int32_t exp_x  = (int32_t)x.exponent;
        const bool    sign_x = (x.sign != 0);
        const bool    zero_x = (x.mantissa == 0);

        // Special operands (canonical packed result, then unpack -- rare path).
        if (exp_x == NAN_EXP)
            return __nv_internal_fp64emu_unpack((fpbits64_t)__FPEMU_DEFNAN_64__);
        if (exp_x == INF_EXP)
        {
            if (!sign_x) return __nv_internal_fp64emu_unpack((fpbits64_t)__FPEMU_INF_64__); // +inf -> +inf
            return __nv_internal_fp64emu_unpack((fpbits64_t)__FPEMU_DEFNAN_64__);           // sqrt(-inf) -> NaN
        }
        if (sign_x)
        {
            if (zero_x) return __nv_internal_fp64emu_unpack((fpbits64_t)__FPEMU_SIGN_64__);  // -0 -> -0
            return __nv_internal_fp64emu_unpack((fpbits64_t)__FPEMU_DEFNAN_64__);            // sqrt(negative) -> NaN
        }
        if (zero_x)
            return __nv_internal_fp64emu_unpack((fpbits64_t)0);                              // +0 -> +0

        // ---- finite positive : fixed-point reciprocal-sqrt root -------------
        int32_t  exp_z  = ((exp_x - 0x3FF) >> 1) + 0x3FE;
        int32_t  odd    = exp_x & 1;                    // exponent parity
        uint64_t mant_x = x.mantissa >> EXTRA_BITS;     // 53-bit significand, implicit bit at 52

        uint32_t mant32_x = (uint32_t)(mant_x >> 21);
        uint32_t rcp32    = __nv_internal_fp64emu_sqrt_recip_sqrt32((uint32_t)odd, mant32_x);
        uint32_t mant32_z = (uint32_t)(((uint64_t)mant32_x * rcp32) >> 32);

        if (odd) { mant_x <<= 8; mant32_z >>= 1; } else { mant_x <<= 9; }

        uint64_t rem64    = mant_x - (uint64_t)mant32_z * mant32_z;
        uint32_t q32      = (uint32_t)(((uint32_t)(rem64 >> 2) * (uint64_t)rcp32) >> 32);
        uint64_t mant64_z = ((uint64_t)mant32_z << 32 | (1u << 5)) + ((uint64_t)q32 << 3);

        // Refine if the root is close to a rounding boundary (exact remainder).
        if ((mant64_z & 0x1FF) < 0x22)
        {
            mant64_z               &= ~(uint64_t)0x3F;
            uint64_t mant64_z_shftd = mant64_z >> 6;
            rem64                   = (mant_x << 52) - mant64_z_shftd * mant64_z_shftd;
            if (rem64 & __FPEMU_SIGN_64__) { --mant64_z; }
            else { if (rem64) mant64_z |= 1; }
        }

        // Same conversion as the unpacked divide: leading bit 62 -> 61 (sticky
        // preserved), exponent biased-1 -> IEEE-biased; full pack rounds.
        fpbits64_unpacked_t r;
        r.sign     = 0u;                                // sqrt result is non-negative
        r.exponent = (uint32_t)(exp_z + 1);
        r.mantissa = (mant64_z >> 1) | (mant64_z & 1);
        return r;
    } // __nv_internal_fp64emu_dsqrt_unpacked

} // namespace impl

// ============================================================================
// Builtin declarations/implementations for sqrt operations
// ============================================================================
#if defined(__FPEMU_INLINE__)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_rn (fpbits64_t x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::rn, fp64emu_accuracy::high>(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_rz (fpbits64_t x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::rz, fp64emu_accuracy::high>(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_ru (fpbits64_t x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::ru, fp64emu_accuracy::high>(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_rd (fpbits64_t x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::rd, fp64emu_accuracy::high>(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_dsqrt_rn (fpbits64_t x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::rn, fp64emu_accuracy::high>(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dsqrt_rn      (fpbits64_t x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::rn, fp64emu_accuracy::mid>(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dsqrt_rn     (fpbits64_t x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::rn, fp64emu_accuracy::low>(x); }
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dsqrt          (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_dsqrt_unpacked<fp64emu_accuracy::high>(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dsqrt (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_dsqrt_unpacked<fp64emu_accuracy::high>(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dsqrt      (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_dsqrt_unpacked<fp64emu_accuracy::mid>(x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dsqrt     (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_dsqrt_unpacked<fp64emu_accuracy::low>(x); }
#endif
#else
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_rn (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_rz (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_ru (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_rd (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_dsqrt_rn (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dsqrt_rn      (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dsqrt_rn     (fpbits64_t x);
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dsqrt          (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dsqrt (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dsqrt      (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dsqrt     (fpbits64_unpacked_t x);
#endif
#endif // __FPEMU_INLINE__

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_SQRT_HPP__ (builtins)

#if defined(__FPEMU_API_CLASSES_DEFINED__) && !defined(__FPEMU_DSQRT_API_MERGED__)
#define __FPEMU_DSQRT_API_MERGED__
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_dsqrt_api.hpp)
// ============================================================================


    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> sqrt (const fp64emu_t<m>& x) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_high_dsqrt_rn(x.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dsqrt_rn(x.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dsqrt_rn(x.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dsqrt_rn (const fp64emu_t<m>& x) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_high_dsqrt_rn(x.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dsqrt_rn(x.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dsqrt_rn(x.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dsqrt_rz (const fp64emu_t<m>& x) { 
        return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dsqrt_rz(x.bits)); }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dsqrt_ru (const fp64emu_t<m>& x) { 
        return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dsqrt_ru(x.bits)); }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dsqrt_rd (const fp64emu_t<m>& x) { 
        return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dsqrt_rd(x.bits)); } 

#if __FPEMU_UNPACKED__ == 1


    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<m> sqrt (const fp64emu_unpacked_t<m>& x) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_high_dsqrt(x.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_low_dsqrt(x.bits)); }
        else                                             { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_mid_dsqrt(x.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<m> __dsqrt_rn (const fp64emu_unpacked_t<m>& x) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_high_dsqrt(x.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_low_dsqrt(x.bits)); }
        else                                             { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_mid_dsqrt(x.bits)); }
    }


#endif // __FPEMU_UNPACKED__ == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_SQRT_HPP__
