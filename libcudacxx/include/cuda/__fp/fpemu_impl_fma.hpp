//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_FMA_H
#define _CUDA___FP_FPEMU_IMPL_FMA_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file fpemu_impl_fma.hpp
 * @brief Implementation of fused multiply-add operations (FMA & MAD) for FPEMU floating point emulation library
 *
 * This header provides the implementation of fused multiply-add operations for the FPEMU library.
 * It includes:
 *   - Fused multiply-add functions for different accuracy and range configurations 
 *   - Special case handling for NaN, inf, zero, etc
 *
 * The implementation is designed to work across both host and device code
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
extern "C" double fma(double x, double y, double z);
#endif

namespace impl
{
    /**
     * @brief Pure FMA core operating on the unpacked representation.
     *
     * Consumes/produces fpbits64_unpacked_t exactly as produced by the universal
     * impl::__nv_internal_fp64emu_unpack and consumed by impl::__nv_internal_fp64emu_pack.
     * Inputs carry a normalized mantissa (implicit bit set, denormals normalized)
     * with inf/nan encoded in the exponent band (around the 0x00007ff0 / 0x0007ff00
     * magics), so the floating-point class is read from the exponent rather than a
     * separate field. The returned value is the pre-rounding intermediate: a 64-bit
     * mantissa with a sticky LSB and an exponent that may be <= 0 (subnormal) or in
     * the inf/nan band. Final rounding, subnormal shifting and inf/saturate are the
     * job of pack. Templated on the rounding mode (sign-of-zero, rd handling) and
     * the method (product accuracy via __mul_128 and range-based special handling).
     */
    template<fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_unpacked_t __nv_internal_fp64emu_fma_unpacked(fpbits64_unpacked_t a,
                                                           fpbits64_unpacked_t b,
                                                           fpbits64_unpacked_t c)
    {
        constexpr fp64emu_accuracy   meth_forced = fp64emu_accuracy::__FPEMU_FMA_METHOD__;
        constexpr fp64emu_accuracy   meth_used   = (meth_forced != fp64emu_accuracy::unset) ? meth_forced : meth;
        // Unpacked cores always run on the fully-accurate full-range unpack/pack
        // boundary (method-independent): the inf/nan folds stay live and underflow
        // flows to the full pack (no FTZ, correct subnormal + min_normal round-up).

        // Inf/Nan exponent magics produced by the universal unpack.
        constexpr uint32_t INF_EXP = 0x00007ff0u;
        constexpr int32_t  NAN_EXP = 0x0007ff00;

        fpbits64_unpacked_t r;
        __uint128_t mantissa_ab;
        fpemu::uint64x2_t ab_res;

        // MUL START:
        // Compute mantissa_ab - the product of a and b in 128-bit
        fpemu::uint32x2_t a_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(a.mantissa);
        fpemu::uint32x2_t b_32x2 = fpemu::bit_cast<fpemu::uint32x2_t>(b.mantissa);
        ab_res = fpemu::bit_cast<fpemu::uint64x2_t>(fpemu::__mul_128<meth_used>(a_32x2, b_32x2));

        mantissa_ab = fpemu::bit_cast<__uint128_t>(ab_res);
        fpemu::uint32x4_t mantissa_ab32 = fpemu::bit_cast<fpemu::uint32x4_t>( mantissa_ab );

        // Exponents/signs (read with explicit signedness: the public field is
        // uint32 but the core needs signed arithmetic for subnormal exponents).
        int32_t exponent_ab = (int32_t)a.exponent + (int32_t)b.exponent - (int32_t)fpemu::BIAS;
        int32_t exponent_c  = (int32_t)c.exponent;
        int32_t sign_ab     = (int32_t)(a.sign ^ b.sign);
        int32_t sign_c      = (int32_t)c.sign;

        // Compute exponent_ab_new - the exponent of the product of a and b
        int mul_nzeros = mantissa_ab32.hi.x[1] < 0x08000000;
        int32_t exponent_ab_new = exponent_ab - mul_nzeros + 1;
        // Shift mantissa_ab
        mantissa_ab = mantissa_ab << (11 - EXTRA_BITS + mul_nzeros);
        // Compute mantissa_c - the mantissa of c
        __uint128_t mantissa_c = c.mantissa;
        // Compute mantissa_r - the result of the product of a and b and c
        __uint128_t mantissa_r;

        {
            // Check if a or b is inf and c is inf and sign_ab != sign_c then return NaN
            if ((a.exponent == INF_EXP || b.exponent == INF_EXP) && c.exponent == INF_EXP && sign_ab != sign_c)
            {
                exponent_ab_new = NAN_EXP;
            }
        }

        // Check if exponent_ab_new is INF_ZERO then return NaN
        if (exponent_ab_new == (int32_t)fpemu::INF_ZERO) 
        { 
            exponent_ab_new = NAN_EXP;
        }

        //ADD START:
        // Compute exponent_r - the larger of exponent_ab_new and exponent_c
        int32_t exponent_r = __max_fp64emu( exponent_ab_new, exponent_c );

        // Compute delta_a and delta_b for mantissas shift
        int32_t delta_a = exponent_r - exponent_ab_new;
        int32_t delta_b = exponent_r - exponent_c;

        #ifndef __CUDA_ARCH__
        delta_a = (delta_a>127) ? 127 : delta_a;
        delta_b = (delta_b>127) ? 127 : delta_b;
        #endif
        // Shift mantissas with jam only (SoftFloat shiftRightJam*); round at pack
        mantissa_ab = fpemu::__shr_128_jam(mantissa_ab, delta_a);
        mantissa_c = fpemu::__shr_128_jam(mantissa_c << 64, delta_b);

        // Add or subtract mantissas
        uint32_t sign_r = sign_ab;
        if (sign_ab == sign_c) 
        {
            mantissa_r = mantissa_ab + mantissa_c;
        }
        else if (mantissa_ab == mantissa_c)
        {
            mantissa_r = 0;
        }
        else if ((mantissa_ab > mantissa_c))
        {
            mantissa_r = mantissa_ab - mantissa_c;
        } 
        else // mantissa_ab < mantissa_c
        {
            sign_r = sign_c;
            mantissa_r = mantissa_c - mantissa_ab;
        }

        if (mantissa_r == 0)
        {
            // Exact cancellation -> zero. IEEE-754 6.3 would make this -0 under
            // round-toward-negative (rd); that rounding-dependent zero sign is
            // intentionally NOT honored here (the core is rounding-independent), so
            // the zero is -0 only when both effective signs are negative.
            sign_r = sign_ab & sign_c;
        }

        // Normalize mantissa_r
        // use reinterpret_cast to avoid slowdown from bit_cast
        uint64_t *m = reinterpret_cast<uint64_t*>( &mantissa_r );
        int nzeros = (m[1] == 0)? (fpemu::__internal_clzll(m[0] + 64)):
                                (fpemu::__internal_clzll(m[1] << 1));

        // Shift mantissa_r
        mantissa_r = (nzeros == 0)? (mantissa_r >> 1): 
                                    (mantissa_r << (nzeros - 1)); 

        fpemu::uint64x2_t mantissa_r64 = fpemu::bit_cast<fpemu::uint64x2_t>(mantissa_r);

        // The result class (inf vs finite-overflow) is recoverable from the
        // exponent band by pack: genuine infinities inherit the huge exponent of
        // their inf operand, finite results never reach it. So no class field is
        // written here; the inf-inf -> NaN and inf*0 -> NaN cases were already
        // folded into exponent_ab_new (NAN_EXP) above.
        r.sign     = sign_r;
        // +1 matches the unified pack's "mask the implicit bit" convention
        // (pack reconstitutes via exp-1); the +1/-1 cancel so packed FMA is
        // bit-exact with the legacy add-convention packer.
        r.exponent = static_cast<uint32_t>(exponent_r - nzeros + 1);
        r.mantissa = mantissa_r64.x[1] | (mantissa_r64.x[0] != 0);

        // The unpacked core runs on the full-range boundary: underflow (and the rare
        // top-subnormal -> min_normal round-up) flows to the full pack, which has the
        // complete mantissa and resolves the correct subnormal / min_normal for every
        // rounding mode. No FTZ here, so no rounding-dependent fix-up is needed.

        return r;
    } // __nv_internal_fp64emu_fma_unpacked

    template<fpemu::rounding rm   = fpemu::rounding::def, 
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_fma (fpbits64_t x, 
                                        fpbits64_t y, 
                                        fpbits64_t z)
    {
        // Forced parameters for the fused multiply-add operation
        constexpr fp64emu_accuracy   meth_forced = fp64emu_accuracy::__FPEMU_FMA_METHOD__;
        constexpr fp64emu_accuracy   meth_used   = (meth_forced != fp64emu_accuracy::unset) ? meth_forced : meth;

        {
                {
                    // FMA = pack(fma_unpacked(unpack(x), unpack(y), unpack(z))). The fma_unpacked
                    // core selects accurate/def/fast internally; the universal full-range
                    // unpack/pack are the shared prologue/epilogue (def/fast are full-range here).
                    fpbits64_unpacked_t a = __nv_internal_fp64emu_unpack(x);
                    fpbits64_unpacked_t b = __nv_internal_fp64emu_unpack(y);
                    fpbits64_unpacked_t c = __nv_internal_fp64emu_unpack(z);
                    fpbits64_unpacked_t r = __nv_internal_fp64emu_fma_unpacked<meth_used>(a, b, c);
                    fpbits64_t result = __nv_internal_fp64emu_pack<rm>(r);

                    if constexpr (rm == fpemu::rounding::rd)
                    {
                        // Exact cancellation (a*b + c == 0 with opposite effective signs)
                        // must yield -0 under round-toward-negative. The rounding-independent
                        // core packs an exact zero to +0 (r.mantissa == 0); a misaligned
                        // remainder can also surface as a tiny artifact. Both map to -0 here.
                        // A genuine underflow keeps a nonzero core mantissa and stays +0.
                        const bool opposite_signs = ((a.sign ^ b.sign) != c.sign);
                        const bool exact_zero      = (r.mantissa == 0) && ((result << 1) == 0);
                        const bool tiny_artifact   = (result == UINT64_C(0x0000000100000000));
                        if (opposite_signs && (exact_zero || tiny_artifact))
                        {
                            fpbits64_unpacked_t zneg;
                            zneg.sign     = 1U << 31;
                            zneg.exponent = 0;
                            zneg.mantissa = 0;
                            result = __nv_internal_fp64emu_pack<rm>(zneg);
                        }
                    }
                    return result;
                }
        }
    } // __nv_internal_fp64emu_fma

} // namespace impl

// ============================================================================
// Builtin declarations/implementations for FMA operations
// ============================================================================
#if defined(__FPEMU_INLINE__)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_rn          (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rn, fp64emu_accuracy::high>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_rz          (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rz, fp64emu_accuracy::high>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_ru          (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::ru, fp64emu_accuracy::high>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_rd          (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rd, fp64emu_accuracy::high>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_fma_rn (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rn, fp64emu_accuracy::high>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_rn      (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rn, fp64emu_accuracy::mid>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_rz      (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rz, fp64emu_accuracy::mid>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_ru      (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::ru, fp64emu_accuracy::mid>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_rd      (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rd, fp64emu_accuracy::mid>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_rn     (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rn, fp64emu_accuracy::low>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_rz     (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rz, fp64emu_accuracy::low>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_ru     (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::ru, fp64emu_accuracy::low>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_rd     (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rd, fp64emu_accuracy::low>(x, y, z); }
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_fma          (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z) { return impl::__nv_internal_fp64emu_fma_unpacked<fp64emu_accuracy::high>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_fma (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z) { return impl::__nv_internal_fp64emu_fma_unpacked<fp64emu_accuracy::high>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_fma      (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z) { return impl::__nv_internal_fp64emu_fma_unpacked<fp64emu_accuracy::mid>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_fma     (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z) { return impl::__nv_internal_fp64emu_fma_unpacked<fp64emu_accuracy::low>(x, y, z); }
#endif
#else
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_rn          (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_rz          (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_ru          (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_rd          (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_fma_rn (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_rn      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_rz      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_ru      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_rd      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_rn     (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_rz     (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_ru     (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_rd     (fpbits64_t x, fpbits64_t y, fpbits64_t z);
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_fma          (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_fma (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_fma      (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_fma     (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
#endif
#endif // __FPEMU_INLINE__

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_FMA_HPP__

#if defined(__FPEMU_API_CLASSES_DEFINED__) && !defined(__FPEMU_FMA_API_MERGED__)
#define __FPEMU_FMA_API_MERGED__
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_fma_api.hpp)
// ============================================================================


    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> fma (const fp64emu_t<m>& x, const fp64emu_t<m>& y, const fp64emu_t<m>& z) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_high_fma_rn(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_fma_rn(x.bits, y.bits, z.bits)); }
        else                                      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_fma_rn(x.bits, y.bits, z.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __fma_rn (const fp64emu_t<m>& x, const fp64emu_t<m>& y, const fp64emu_t<m>& z) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_high_fma_rn(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_fma_rn(x.bits, y.bits, z.bits)); }
        else                                      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_fma_rn(x.bits, y.bits, z.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __fma_rz (const fp64emu_t<m>& x, const fp64emu_t<m>& y, const fp64emu_t<m>& z) {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_fma_rz(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_fma_rz(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_fma_rz(x.bits, y.bits, z.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_fma_rz(x.bits, y.bits, z.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __fma_ru (const fp64emu_t<m>& x, const fp64emu_t<m>& y, const fp64emu_t<m>& z) {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_fma_ru(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_fma_ru(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_fma_ru(x.bits, y.bits, z.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_fma_ru(x.bits, y.bits, z.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __fma_rd (const fp64emu_t<m>& x, const fp64emu_t<m>& y, const fp64emu_t<m>& z) {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_fma_rd(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_fma_rd(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_fma_rd(x.bits, y.bits, z.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_fma_rd(x.bits, y.bits, z.bits)); }
    }

#if __FPEMU_UNPACKED__ == 1


    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<m> fma (const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y, const fp64emu_unpacked_t<m>& z) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_high_fma(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_low_fma(x.bits, y.bits, z.bits)); }
        else                                      { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_mid_fma(x.bits, y.bits, z.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<m> __fma_rn (const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y, const fp64emu_unpacked_t<m>& z) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_high_fma(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_low_fma(x.bits, y.bits, z.bits)); }
        else                                      { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_mid_fma(x.bits, y.bits, z.bits)); }
    }


#endif // __FPEMU_UNPACKED__ == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_FMA_API_MERGED__
