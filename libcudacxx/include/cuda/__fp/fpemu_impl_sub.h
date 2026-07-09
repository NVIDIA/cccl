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
 * - Subtraction functions for fpemu
 * - Subtraction operators for fpemu
 * - Subtraction functions to other types
 *
 * The subtraction functions are designed to work across both host and device code
 * through appropriate decorators and provide bit-exact results matching hardware
 * floating point units.
 */

#include <cuda/__fp/fpemu_impl.h>
#include <cuda/__fp/fpemu_impl_unpack.h>
#include <cuda/__fp/fpemu_impl_add.h>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


    /**
     * @brief Subtract two __fpbits64_unpacked
     * 
     * This function subtracts two __fpbits64_unpacked.
     * 
     * @param a The first __fpbits64_unpacked
     * @param b The second __fpbits64_unpacked
     * @return The result of the subtraction
     */
    template<fpemu_accuracy   _Acc = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API
    __fpbits64_unpacked __internal_fp64emu_dsub_unpacked(__fpbits64_unpacked __a, 
                                                         __fpbits64_unpacked __b) noexcept
    {
        return __internal_fp64emu_dadd_unpacked<_Acc, true>(__a, __b);
    }

    /**
     * @brief Subtract two __fpbits64
     * 
     * This function subtracts two __fpbits64.
     * 
     * @param x The first __fpbits64
     * @param y The second __fpbits64
     * @return The result of the subtraction
     */
    template<__fpemu_rounding    _Rm  = __fpemu_rounding::def, 
             fpemu_accuracy      _Acc = fpemu_accuracy::def>
    _CCCL_TRIVIAL_API
    __fpbits64 __internal_fp64emu_dsub(__fpbits64 __x, 
                                       __fpbits64 __y) noexcept
    {
        // Forced parameters for the subtraction operation
        constexpr fpemu_accuracy   __acc_forced = fpemu_accuracy::_CCCL_FPEMU_ADD_METHOD;
        constexpr fpemu_accuracy   __acc_used   = (__acc_forced != fpemu_accuracy::unset) ? __acc_forced : _Acc;

        {
            // Pass true to the dadd function to indicate that we are subtracting
            return __internal_fp64emu_dadd<_Rm, __acc_used, true>(__x, __y);
        }
    } // __internal_fp64emu_dsub


// ============================================================================
// Builtin declarations/implementations for subtraction operations
// ============================================================================
#if defined(_CCCL_FPEMU_INLINE)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_rn (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rn, fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_rz (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rz, fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_ru (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::ru, fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_rd (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rd, fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_high_dsub_rn (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rn, fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_rn  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rn, fpemu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_rz  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rz, fpemu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_ru  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::ru, fpemu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_rd  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rd, fpemu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_rn  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rn, fpemu_accuracy::low>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_rz  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rz, fpemu_accuracy::low>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_ru  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::ru, fpemu_accuracy::low>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_rd  (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rd, fpemu_accuracy::low>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_dsub          (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_dsub_unpacked<fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_high_dsub (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_dsub_unpacked<fpemu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_mid_dsub      (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_dsub_unpacked<fpemu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_low_dsub     (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_dsub_unpacked<fpemu_accuracy::low>(__x, __y); }
#else
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_rn (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_rz (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_ru (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_rd (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_high_dsub_rn (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_rn  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_rz  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_ru  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_rd  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_rn  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_rz  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_ru  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_rd  (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_dsub      (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_high_dsub (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_mid_dsub  (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_low_dsub  (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
#endif // _CCCL_FPEMU_INLINE

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_SUB_H (builtins)

#if defined(_CCCL_FPEMU_API_CLASSES_DEFINED) && !defined(_CCCL_FPEMU_DSUB_API_MERGED)
#define _CCCL_FPEMU_DSUB_API_MERGED
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_dsub_api.hpp)
// ============================================================================

    // Default API implementation - binary subtraction operator
    template<fpemu_accuracy _Acc> _CCCL_API fpemu<double, _Acc> operator- (const fpemu<double, _Acc>& __x, 
                                                                                  const fpemu<double, _Acc>& __y) noexcept
    {
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_high_dsub_rn(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::mid)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_mid_dsub_rn(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_low_dsub_rn(__x.bits, __y.bits)); }
        else                                             { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dsub_rn(__x.bits, __y.bits)); }
    } // operator-


    template<fpemu_accuracy _Acc>
    _CCCL_API fpemu<double, _Acc> __dsub_rn (const fpemu<double, _Acc>& __x, 
                                             const fpemu<double, _Acc>& __y) noexcept { 
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_high_dsub_rn(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_low_dsub_rn(__x.bits, __y.bits)); }
        else                                             { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_mid_dsub_rn(__x.bits, __y.bits)); }
    }
    template<fpemu_accuracy _Acc>
    _CCCL_API fpemu<double, _Acc> __dsub_rz (const fpemu<double, _Acc>& __x, 
                                             const fpemu<double, _Acc>& __y) noexcept {
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dsub_rz(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::mid)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_mid_dsub_rz(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_low_dsub_rz(__x.bits, __y.bits)); }
        else                                             { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dsub_rz(__x.bits, __y.bits)); }
    }
    template<fpemu_accuracy _Acc>
    _CCCL_API fpemu<double, _Acc> __dsub_ru (const fpemu<double, _Acc>& __x, 
                                             const fpemu<double, _Acc>& __y) noexcept {
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dsub_ru(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::mid)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_mid_dsub_ru(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_low_dsub_ru(__x.bits, __y.bits)); }
        else                                             { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dsub_ru(__x.bits, __y.bits)); }
    }
    template<fpemu_accuracy _Acc>
    _CCCL_API fpemu<double, _Acc> __dsub_rd (const fpemu<double, _Acc>& __x, 
                                             const fpemu<double, _Acc>& __y) noexcept {
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dsub_rd(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::mid)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_mid_dsub_rd(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_low_dsub_rd(__x.bits, __y.bits)); }
        else                                             { return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_dsub_rd(__x.bits, __y.bits)); }
    }


    // Operator- for unpacked subtraction
    template<fpemu_accuracy _Acc>
    _CCCL_DEVICE_API fpemu_unpacked<double, _Acc> operator- (const fpemu_unpacked<double, _Acc>& __x, 
                                                                    const fpemu_unpacked<double, _Acc>& __y) noexcept
    {
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_high_dsub(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::mid)  { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_mid_dsub(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_low_dsub(__x.bits, __y.bits)); }
        else                                             { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_dsub(__x.bits, __y.bits)); }
    } // operator-


    template<fpemu_accuracy _Acc>
    _CCCL_API fpemu_unpacked<double, _Acc> __dsub_rn (const fpemu_unpacked<double, _Acc>& __x, 
                                                      const fpemu_unpacked<double, _Acc>& __y) noexcept { 
        if      constexpr (_Acc == fpemu_accuracy::high) { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_high_dsub(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fpemu_accuracy::low)  { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_low_dsub(__x.bits, __y.bits)); }
        else                                             { return fpemu_unpacked<double, _Acc>(__fpbits64_construct, __fp64emu_unpacked_mid_dsub(__x.bits, __y.bits)); }
    }



} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_SUB_H
