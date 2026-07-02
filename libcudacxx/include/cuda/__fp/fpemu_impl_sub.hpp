//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_SUB_H
#define _CUDA___FP_FPEMU_IMPL_SUB_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/** 
 * @file fpemu_dsub_impl.hpp
 * @brief Implementation of double-precision subtraction operations for FPEMU floating point emulation library
 *
 * This header provides the implementation of double-precision subtraction operations for the FPEMU library.
 * It includes:
 *
 * - Subtraction functions for fp64emu_t
 * - Subtraction operators for fp64emu_t
 * - Subtraction functions to other types
 *
 * The subtraction functions are designed to work across both host and device code
 * through appropriate decorators and provide bit-exact results matching hardware
 * floating point units.
 */

#include <cuda/__fp/fpemu_common.hpp>
#include <cuda/__fp/fpemu_impl_utils.hpp>
#include <cuda/__fp/fpemu_impl_unpack.hpp>
#include <cuda/__fp/fpemu_impl_add.hpp>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

namespace impl
{
    /**
     * @brief Subtract two fpbits64_unpacked_t
     * 
     * This function subtracts two fpbits64_unpacked_t.
     * 
     * @param a The first fpbits64_unpacked_t
     * @param b The second fpbits64_unpacked_t
     * @return The result of the subtraction
     */
    template<fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_unpacked_t __nv_internal_fp64emu_dsub_unpacked(fpbits64_unpacked_t a, 
                                                          fpbits64_unpacked_t b)
    {
        return __nv_internal_fp64emu_dadd_unpacked<meth, true>(a, b);
    }

    /**
     * @brief Subtract two fpbits64_t
     * 
     * This function subtracts two fpbits64_t.
     * 
     * @param x The first fpbits64_t
     * @param y The second fpbits64_t
     * @return The result of the subtraction
     */
    template<fpemu::rounding rm   = fpemu::rounding::def, 
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_dsub(fpbits64_t x, 
                                        fpbits64_t y)
    {
        // Forced parameters for the subtraction operation
        constexpr fp64emu_accuracy   meth_forced = fp64emu_accuracy::__FPEMU_ADD_METHOD__;
        constexpr fp64emu_accuracy   meth_used   = (meth_forced != fp64emu_accuracy::unset) ? meth_forced : meth;

        {
            // Pass true to the dadd function to indicate that we are subtracting
            return __nv_internal_fp64emu_dadd<rm, meth_used, true>(x, y);
        }
    } // __nv_internal_fp64emu_dsub
} // namespace impl

// ============================================================================
// Builtin declarations/implementations for subtraction operations
// ============================================================================
#if defined(__FPEMU_INLINE__)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsub_rn (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dsub<fpemu::rounding::rn, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsub_rz (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dsub<fpemu::rounding::rz, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsub_ru (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dsub<fpemu::rounding::ru, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsub_rd (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dsub<fpemu::rounding::rd, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_dsub_rn (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dsub<fpemu::rounding::rn, fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dsub_rn      (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dsub<fpemu::rounding::rn, fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dsub_rz      (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dsub<fpemu::rounding::rz, fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dsub_ru      (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dsub<fpemu::rounding::ru, fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dsub_rd      (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dsub<fpemu::rounding::rd, fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dsub_rn     (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dsub<fpemu::rounding::rn, fp64emu_accuracy::low>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dsub_rz     (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dsub<fpemu::rounding::rz, fp64emu_accuracy::low>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dsub_ru     (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dsub<fpemu::rounding::ru, fp64emu_accuracy::low>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dsub_rd     (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_dsub<fpemu::rounding::rd, fp64emu_accuracy::low>(x, y); }
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dsub          (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_dsub_unpacked<fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dsub (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_dsub_unpacked<fp64emu_accuracy::high>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dsub      (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_dsub_unpacked<fp64emu_accuracy::mid>(x, y); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dsub     (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_dsub_unpacked<fp64emu_accuracy::low>(x, y); }
#endif
#else
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsub_rn (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsub_rz (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsub_ru (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsub_rd (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_dsub_rn (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dsub_rn      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dsub_rz      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dsub_ru      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dsub_rd      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dsub_rn     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dsub_rz     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dsub_ru     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dsub_rd     (fpbits64_t x, fpbits64_t y);
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dsub          (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dsub (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dsub      (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dsub     (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
#endif
#endif // __FPEMU_INLINE__

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_SUB_HPP__ (builtins)

#if defined(__FPEMU_API_CLASSES_DEFINED__) && !defined(__FPEMU_DSUB_API_MERGED__)
#define __FPEMU_DSUB_API_MERGED__
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_dsub_api.hpp)
// ============================================================================

    // Default API implementation - binary subtraction operator
    template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ static fp64emu_t<m> operator- (const fp64emu_t<m>& x, 
                                                                        const fp64emu_t<m>& y)
    {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_high_dsub_rn(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dsub_rn(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dsub_rn(x.bits, y.bits)); }
        else                                      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dsub_rn(x.bits, y.bits)); }
    } // operator-


    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dsub_rn (const fp64emu_t<m>& x, const fp64emu_t<m>& y) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_high_dsub_rn(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dsub_rn(x.bits, y.bits)); }
        else                                      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dsub_rn(x.bits, y.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dsub_rz (const fp64emu_t<m>& x, const fp64emu_t<m>& y) {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dsub_rz(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dsub_rz(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dsub_rz(x.bits, y.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dsub_rz(x.bits, y.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dsub_ru (const fp64emu_t<m>& x, const fp64emu_t<m>& y) {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dsub_ru(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dsub_ru(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dsub_ru(x.bits, y.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dsub_ru(x.bits, y.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __dsub_rd (const fp64emu_t<m>& x, const fp64emu_t<m>& y) {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dsub_rd(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dsub_rd(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dsub_rd(x.bits, y.bits)); }
        else                                             { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_dsub_rd(x.bits, y.bits)); }
    }

#if __FPEMU_UNPACKED__ == 1

    // Operator- for unpacked subtraction
    template<fp64emu_accuracy m>
    __FPEMU_DEVICE_DECL__ static fp64emu_unpacked_t<m> operator- (const fp64emu_unpacked_t<m>& x, 
                                                                            const fp64emu_unpacked_t<m>& y)
    {
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_high_dsub(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::mid)      { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_mid_dsub(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_low_dsub(x.bits, y.bits)); }
        else                                      { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_dsub(x.bits, y.bits)); }
    } // operator-


    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<m> __dsub_rn (const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_high_dsub(x.bits, y.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_low_dsub(x.bits, y.bits)); }
        else                                      { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_mid_dsub(x.bits, y.bits)); }
    }


#endif // __FPEMU_UNPACKED__ == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_SUB_HPP__
