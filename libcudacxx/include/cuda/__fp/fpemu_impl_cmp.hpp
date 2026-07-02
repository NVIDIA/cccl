//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_CMP_H
#define _CUDA___FP_FPEMU_IMPL_CMP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file fpemu_impl_cmp.hpp
 * @brief Implementation of comparison operations for FPEMU floating point emulation library
 *
 * This header provides the implementation of comparison operations for the FPEMU library.  
 * It includes:
 *
 * - Comparison functions for equality, less than, greater than, etc
 * - Special case handling for NaN, inf, zero, etc
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

namespace impl
{
    // ------------------------------------------------------------------------
    // Bit-level helpers (IEEE-754 binary64 layout). No SoftFloat dependency;
    // the comparison logic mirrors SoftFloat's f64_eq / f64_lt / f64_le.
    // ------------------------------------------------------------------------

    // Magnitude mask: all bits except the sign bit.
    static constexpr fpbits64_t __FP64EMU_CMP_ABS_MASK = __FPEMU_ABS_64__;

    /// @brief True if the bit pattern encodes a NaN (max exponent, nonzero mantissa).
    __FPEMU_INTERNAL_DECL__
    bool __nv_internal_fp64emu_is_nan_bits (fpbits64_t ui)
    {
        return ((~ui & __FPEMU_EXP_64__) == 0)
            && ((ui & __FPEMU_MANT_64__) != 0);
    } // __nv_internal_fp64emu_is_nan_bits

    /// @brief IEEE-754 equality. Unordered (NaN) compares false; +0 equals -0.
    __FPEMU_INTERNAL_DECL__ 
    bool __nv_internal_fp64emu_cmp_eq (fpbits64_t x, fpbits64_t y)
    {
        if (__nv_internal_fp64emu_is_nan_bits(x) || __nv_internal_fp64emu_is_nan_bits(y))
        {
            return false;
        }
        // Equal bit patterns, or both are zero (+0 / -0 ignore the sign bit).
        return (x == y) || (((x | y) & __FP64EMU_CMP_ABS_MASK) == 0);
    } // __nv_internal_fp64emu_cmp_eq

    /// @brief IEEE-754 less-than. Unordered (NaN) compares false.
    __FPEMU_INTERNAL_DECL__ 
    bool __nv_internal_fp64emu_cmp_lt (fpbits64_t x, fpbits64_t y)
    {
        if (__nv_internal_fp64emu_is_nan_bits(x) || __nv_internal_fp64emu_is_nan_bits(y))
        {
            return false;
        }
        const bool sign_x = (x >> 63) != 0;
        const bool sign_y = (y >> 63) != 0;
        // Different signs: x < y only if x is negative and not both zero.
        // Same sign: ordering of magnitudes, inverted when both are negative.
        return (sign_x != sign_y)
            ? (sign_x && (((x | y) & __FP64EMU_CMP_ABS_MASK) != 0))
            : ((x != y) && (sign_x ^ (x < y)));
    } // __nv_internal_fp64emu_cmp_lt

    /// @brief IEEE-754 less-or-equal. Unordered (NaN) compares false.
    __FPEMU_INTERNAL_DECL__ 
    bool __nv_internal_fp64emu_cmp_le (fpbits64_t x, fpbits64_t y)
    {
        if (__nv_internal_fp64emu_is_nan_bits(x) || __nv_internal_fp64emu_is_nan_bits(y))
        {
            return false;
        }
        const bool sign_x = (x >> 63) != 0;
        const bool sign_y = (y >> 63) != 0;
        return (sign_x != sign_y)
            ? (sign_x || (((x | y) & __FP64EMU_CMP_ABS_MASK) == 0))
            : ((x == y) || (sign_x ^ (x < y)));
    } // __nv_internal_fp64emu_cmp_le

    // ne is the logical negation of eq, so unordered (NaN) compares true.
    __FPEMU_INTERNAL_DECL__  bool     __nv_internal_fp64emu_cmp_ne (fpbits64_t x, fpbits64_t y) { return !impl::__nv_internal_fp64emu_cmp_eq (x, y); }
    // gt / ge are lt / le with swapped operands, preserving IEEE unordered=false.
    __FPEMU_INTERNAL_DECL__  bool     __nv_internal_fp64emu_cmp_gt (fpbits64_t x, fpbits64_t y) { return  impl::__nv_internal_fp64emu_cmp_lt (y, x); }
    __FPEMU_INTERNAL_DECL__  bool     __nv_internal_fp64emu_cmp_ge (fpbits64_t x, fpbits64_t y) { return  impl::__nv_internal_fp64emu_cmp_le (y, x); }

    #if __FPEMU_UNPACKED__ == 1

        // ---- True unpacked comparisons -------------------------------------
        // Operate directly on the fully-accurate unpacked fields (no pack round
        // trip). The full unpack yields an order-preserving form:
        //   * NaN  -> exponent == 0x0007ff00 (magic, distinct from any finite/inf);
        //   * Inf  -> exponent == 0x00007ff0;
        //   * the SIGNED exponent is monotonic with magnitude (zero/denormal <= 0,
        //     normals 1..2046, inf, nan), so (exponent, mantissa) is a magnitude key;
        //   * a zero value has mantissa == 0 (its exponent is unconstrained, e.g. an
        //     additive cancellation), so zero is detected by the mantissa, and
        //     +0 / -0 differ only in the sign field.
        // eq and lt are the primitives; le/ne/gt/ge derive from them and inherit the
        // IEEE unordered (NaN) semantics for free.
        static constexpr int32_t __FP64EMU_UNP_NAN_EXP = 0x0007ff00;

        __FPEMU_INTERNAL_DECL__
        bool __nv_internal_fp64emu_unp_is_nan (fpbits64_unpacked_t u)
        {
            return static_cast<int32_t>(u.exponent) == __FP64EMU_UNP_NAN_EXP;
        }
        __FPEMU_INTERNAL_DECL__
        bool __nv_internal_fp64emu_unp_is_zero (fpbits64_unpacked_t u)
        {
            return u.mantissa == 0;
        }

        __FPEMU_INTERNAL_DECL__ 
        bool __nv_internal_fp64emu_cmp_eq_unpacked (fpbits64_unpacked_t x, fpbits64_unpacked_t y)
        {
            if (__nv_internal_fp64emu_unp_is_nan(x) || __nv_internal_fp64emu_unp_is_nan(y))
                return false;
            const bool zx = __nv_internal_fp64emu_unp_is_zero(x);
            const bool zy = __nv_internal_fp64emu_unp_is_zero(y);
            if (zx || zy) return zx && zy;          // +0 == -0; zero != nonzero
            // Non-zero, non-NaN: equal iff same sign and identical magnitude key.
            return (x.sign == y.sign)
                && (x.exponent == y.exponent)
                && (x.mantissa == y.mantissa);
        }

        __FPEMU_INTERNAL_DECL__ 
        bool __nv_internal_fp64emu_cmp_lt_unpacked (fpbits64_unpacked_t x, fpbits64_unpacked_t y)
        {
            if (__nv_internal_fp64emu_unp_is_nan(x) || __nv_internal_fp64emu_unp_is_nan(y))
                return false;
            const bool zx = __nv_internal_fp64emu_unp_is_zero(x);
            const bool zy = __nv_internal_fp64emu_unp_is_zero(y);
            if (zx && zy) return false;             // +/-0 are equal
            const bool sx = (x.sign != 0);
            const bool sy = (y.sign != 0);
            // Different signs (and not both zero): x < y exactly when x is negative.
            // (A signed zero against an opposite-sign nonzero also resolves here.)
            if (sx != sy) return sx;
            // Same sign. Magnitude order with zero as the smallest magnitude.
            bool mag_x_lt_y;
            if (zx)      mag_x_lt_y = true;         // 0 < |y|  (y nonzero)
            else if (zy) mag_x_lt_y = false;        // |x| > 0
            else         mag_x_lt_y =
                (static_cast<int32_t>(x.exponent) < static_cast<int32_t>(y.exponent)) ||
                ((x.exponent == y.exponent) && (x.mantissa < y.mantissa));
            // Both negative reverses the magnitude order; both positive keeps it.
            return sx ? (!mag_x_lt_y &&
                         !__nv_internal_fp64emu_cmp_eq_unpacked(x, y))
                      : mag_x_lt_y;
        }

        __FPEMU_INTERNAL_DECL__ 
        bool __nv_internal_fp64emu_cmp_le_unpacked (fpbits64_unpacked_t x, fpbits64_unpacked_t y)
        {
            return __nv_internal_fp64emu_cmp_lt_unpacked(x, y)
                || __nv_internal_fp64emu_cmp_eq_unpacked(x, y);
        }

        __FPEMU_INTERNAL_DECL__ 
        bool __nv_internal_fp64emu_cmp_gt_unpacked (fpbits64_unpacked_t x, fpbits64_unpacked_t y)
        {
            return __nv_internal_fp64emu_cmp_lt_unpacked(y, x);
        }

        __FPEMU_INTERNAL_DECL__ 
        bool __nv_internal_fp64emu_cmp_ge_unpacked (fpbits64_unpacked_t x, fpbits64_unpacked_t y)
        {
            return __nv_internal_fp64emu_cmp_le_unpacked(y, x);
        }

        __FPEMU_INTERNAL_DECL__ 
        bool __nv_internal_fp64emu_cmp_ne_unpacked (fpbits64_unpacked_t x, fpbits64_unpacked_t y)
        {
            return !__nv_internal_fp64emu_cmp_eq_unpacked(x, y);
        }
 
    #endif
} // namespace impl

// ============================================================================
// Builtin declarations/implementations for comparison operations
// ============================================================================
#if defined(__FPEMU_INLINE__)
#if (__FPEMU_PACKED_VIA_UNPACKED__ == 1)
// Packed-via-unpacked (testing): route the packed comparison builtins through the
// unpacked cores. unpack(x) yields the fully-accurate, order-preserving form the
// unpacked comparators expect; comparison is rounding-independent, so no pack step.
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_eq (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_cmp_eq_unpacked (impl::__nv_internal_fp64emu_unpack (x), impl::__nv_internal_fp64emu_unpack (y)); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_ne (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_cmp_ne_unpacked (impl::__nv_internal_fp64emu_unpack (x), impl::__nv_internal_fp64emu_unpack (y)); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_le (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_cmp_le_unpacked (impl::__nv_internal_fp64emu_unpack (x), impl::__nv_internal_fp64emu_unpack (y)); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_lt (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_cmp_lt_unpacked (impl::__nv_internal_fp64emu_unpack (x), impl::__nv_internal_fp64emu_unpack (y)); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_ge (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_cmp_ge_unpacked (impl::__nv_internal_fp64emu_unpack (x), impl::__nv_internal_fp64emu_unpack (y)); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_gt (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_cmp_gt_unpacked (impl::__nv_internal_fp64emu_unpack (x), impl::__nv_internal_fp64emu_unpack (y)); }
#else
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_eq (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_cmp_eq (x, y); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_ne (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_cmp_ne (x, y); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_le (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_cmp_le (x, y); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_lt (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_cmp_lt (x, y); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_ge (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_cmp_ge (x, y); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_gt (fpbits64_t x, fpbits64_t y) { return impl::__nv_internal_fp64emu_cmp_gt (x, y); }
#endif // __FPEMU_PACKED_VIA_UNPACKED__
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_unpacked_cmp_eq (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_cmp_eq_unpacked (x, y); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_unpacked_cmp_ne (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_cmp_ne_unpacked (x, y); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_unpacked_cmp_le (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_cmp_le_unpacked (x, y); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_unpacked_cmp_lt (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_cmp_lt_unpacked (x, y); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_unpacked_cmp_ge (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_cmp_ge_unpacked (x, y); }
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_unpacked_cmp_gt (fpbits64_unpacked_t x, fpbits64_unpacked_t y) { return impl::__nv_internal_fp64emu_cmp_gt_unpacked (x, y); }
#endif
#else
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_eq (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_ne (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_le (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_lt (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_ge (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_cmp_gt (fpbits64_t x, fpbits64_t y);
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_unpacked_cmp_eq (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_unpacked_cmp_ne (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_unpacked_cmp_le (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_unpacked_cmp_lt (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_unpacked_cmp_ge (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__  bool __nv_fp64emu_unpacked_cmp_gt (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
#endif
#endif // __FPEMU_INLINE__


} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPEMU_IMPL_CMP_H

#if defined(__FPEMU_API_CLASSES_DEFINED__) && !defined(__FPEMU_CMP_API_MERGED__)
#define __FPEMU_CMP_API_MERGED__

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{




// ============================================================================
// API (merged from fp64emu_cmp_api.hpp)
// ============================================================================

    // Comparison operators
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ static inline bool operator==(const fp64emu_t<m>& x, const fp64emu_t<m>& y){
        return __nv_fp64emu_cmp_eq(x.bits, y.bits);
    }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ static inline bool operator!=(const fp64emu_t<m>& x, const fp64emu_t<m>& y){
        return __nv_fp64emu_cmp_ne(x.bits, y.bits);
    }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ static inline bool operator<(const fp64emu_t<m>& x, const fp64emu_t<m>& y){
        return __nv_fp64emu_cmp_lt(x.bits, y.bits);
    }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ static inline bool operator>(const fp64emu_t<m>& x, const fp64emu_t<m>& y){
        return __nv_fp64emu_cmp_gt(x.bits, y.bits);
    }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ static inline bool operator<=(const fp64emu_t<m>& x, const fp64emu_t<m>& y){
        return __nv_fp64emu_cmp_le(x.bits, y.bits);
    }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ static inline bool operator>=(const fp64emu_t<m>& x, const fp64emu_t<m>& y){
        return __nv_fp64emu_cmp_ge(x.bits, y.bits);
    }

#if __FPEMU_UNPACKED__ == 1

    // Unpacked comparison operators
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ static inline bool operator==(const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y){
        return __nv_fp64emu_unpacked_cmp_eq(x.bits, y.bits);
    }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ static inline bool operator!=(const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y){
        return __nv_fp64emu_unpacked_cmp_ne(x.bits, y.bits);
    }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ static inline bool operator<(const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y){
        return __nv_fp64emu_unpacked_cmp_lt(x.bits, y.bits);
    }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ static inline bool operator>(const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y){
        return __nv_fp64emu_unpacked_cmp_gt(x.bits, y.bits);
    }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ static inline bool operator<=(const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y){
        return __nv_fp64emu_unpacked_cmp_le(x.bits, y.bits);
    }
    template<fp64emu_accuracy m>  __FPEMU_HOST_DEVICE_DECL__ static inline bool operator>=(const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y){
        return __nv_fp64emu_unpacked_cmp_ge(x.bits, y.bits);
    }

#endif // __FPEMU_UNPACKED__ == 1


} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __FPEMU_CMP_API_MERGED__
