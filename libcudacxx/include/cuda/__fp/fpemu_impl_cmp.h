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
 * @file fpemu_impl_cmp.h
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

#include <cuda/__fp/fpemu_impl.h>
#include <cuda/__fp/fpemu_impl_unpack.h>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


    // ------------------------------------------------------------------------
    // Bit-level helpers (IEEE-754 binary64 layout). No SoftFloat dependency;
    // the comparison logic mirrors SoftFloat's f64_eq / f64_lt / f64_le.
    // ------------------------------------------------------------------------

    // Magnitude mask: all bits except the sign bit.
    static constexpr __fpbits64 __FP64EMU_CMP_ABS_MASK = _CCCL_FPEMU_ABS_64;

    /// @brief True if the bit pattern encodes a NaN (max exponent, nonzero mantissa).
    _CCCL_TRIVIAL_API
    bool __internal_fp64emu_is_nan_bits (__fpbits64 __ui) noexcept
    {
        return ((~__ui & _CCCL_FPEMU_EXP_64) == 0)
            && ((__ui & _CCCL_FPEMU_MANT_64) != 0);
    } // __internal_fp64emu_is_nan_bits

    /// @brief IEEE-754 equality. Unordered (NaN) compares false; +0 equals -0.
    _CCCL_TRIVIAL_API 
    bool __internal_fp64emu_cmp_eq (__fpbits64 __x, __fpbits64 __y) noexcept
    {
        if (__internal_fp64emu_is_nan_bits(__x) || __internal_fp64emu_is_nan_bits(__y))
        {
            return false;
        }
        // Equal bit patterns, or both are zero (+0 / -0 ignore the sign bit).
        return (__x == __y) || (((__x | __y) & __FP64EMU_CMP_ABS_MASK) == 0);
    } // __internal_fp64emu_cmp_eq

    /// @brief IEEE-754 less-than. Unordered (NaN) compares false.
    _CCCL_TRIVIAL_API 
    bool __internal_fp64emu_cmp_lt (__fpbits64 __x, __fpbits64 __y) noexcept
    {
        if (__internal_fp64emu_is_nan_bits(__x) || __internal_fp64emu_is_nan_bits(__y))
        {
            return false;
        }
        const bool __sign_x = (__x >> 63) != 0;
        const bool __sign_y = (__y >> 63) != 0;
        // Different signs: x < y only if x is negative and not both zero.
        // Same sign: ordering of magnitudes, inverted when both are negative.
        return (__sign_x != __sign_y)
            ? (__sign_x && (((__x | __y) & __FP64EMU_CMP_ABS_MASK) != 0))
            : ((__x != __y) && (__sign_x ^ (__x < __y)));
    } // __internal_fp64emu_cmp_lt

    /// @brief IEEE-754 less-or-equal. Unordered (NaN) compares false.
    _CCCL_TRIVIAL_API 
    bool __internal_fp64emu_cmp_le (__fpbits64 __x, __fpbits64 __y) noexcept
    {
        if (__internal_fp64emu_is_nan_bits(__x) || __internal_fp64emu_is_nan_bits(__y))
        {
            return false;
        }
        const bool __sign_x = (__x >> 63) != 0;
        const bool __sign_y = (__y >> 63) != 0;
        return (__sign_x != __sign_y)
            ? (__sign_x || (((__x | __y) & __FP64EMU_CMP_ABS_MASK) == 0))
            : ((__x == __y) || (__sign_x ^ (__x < __y)));
    } // __internal_fp64emu_cmp_le

    // ne is the logical negation of eq, so unordered (NaN) compares true.
    _CCCL_TRIVIAL_API  bool     __internal_fp64emu_cmp_ne (__fpbits64 __x, __fpbits64 __y) noexcept { return !__internal_fp64emu_cmp_eq (__x, __y); }
    // gt / ge are lt / le with swapped operands, preserving IEEE unordered=false.
    _CCCL_TRIVIAL_API  bool     __internal_fp64emu_cmp_gt (__fpbits64 __x, __fpbits64 __y) noexcept { return  __internal_fp64emu_cmp_lt (__y, __x); }
    _CCCL_TRIVIAL_API  bool     __internal_fp64emu_cmp_ge (__fpbits64 __x, __fpbits64 __y) noexcept { return  __internal_fp64emu_cmp_le (__y, __x); }


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

        _CCCL_TRIVIAL_API
        bool __internal_fp64emu_unp_is_nan (__fpbits64_unpacked __u) noexcept
        {
            return static_cast<int32_t>(__u.exponent) == __FP64EMU_UNP_NAN_EXP;
        }
        _CCCL_TRIVIAL_API
        bool __internal_fp64emu_unp_is_zero (__fpbits64_unpacked __u) noexcept
        {
            return __u.mantissa == 0;
        }

        _CCCL_TRIVIAL_API 
        bool __internal_fp64emu_cmp_eq_unpacked (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
        {
            if (__internal_fp64emu_unp_is_nan(__x) || __internal_fp64emu_unp_is_nan(__y))
                return false;
            const bool __zx = __internal_fp64emu_unp_is_zero(__x);
            const bool __zy = __internal_fp64emu_unp_is_zero(__y);
            if (__zx || __zy) return __zx && __zy;          // +0 == -0; zero != nonzero
            // Non-zero, non-NaN: equal iff same sign and identical magnitude key.
            return (__x.sign == __y.sign)
                && (__x.exponent == __y.exponent)
                && (__x.mantissa == __y.mantissa);
        }

        _CCCL_TRIVIAL_API 
        bool __internal_fp64emu_cmp_lt_unpacked (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
        {
            if (__internal_fp64emu_unp_is_nan(__x) || __internal_fp64emu_unp_is_nan(__y))
                return false;
            const bool __zx = __internal_fp64emu_unp_is_zero(__x);
            const bool __zy = __internal_fp64emu_unp_is_zero(__y);
            if (__zx && __zy) return false;             // +/-0 are equal
            const bool __sx = (__x.sign != 0);
            const bool __sy = (__y.sign != 0);
            // Different signs (and not both zero): x < y exactly when x is negative.
            // (A signed zero against an opposite-sign nonzero also resolves here.)
            if (__sx != __sy) return __sx;
            // Same sign. Magnitude order with zero as the smallest magnitude.
            bool __mag_x_lt_y;
            if (__zx)      __mag_x_lt_y = true;         // 0 < |y|  (y nonzero)
            else if (__zy) __mag_x_lt_y = false;        // |x| > 0
            else         __mag_x_lt_y =
                (static_cast<int32_t>(__x.exponent) < static_cast<int32_t>(__y.exponent)) ||
                ((__x.exponent == __y.exponent) && (__x.mantissa < __y.mantissa));
            // Both negative reverses the magnitude order; both positive keeps it.
            return __sx ? (!__mag_x_lt_y &&
                         !__internal_fp64emu_cmp_eq_unpacked(__x, __y))
                      : __mag_x_lt_y;
        }

        _CCCL_TRIVIAL_API 
        bool __internal_fp64emu_cmp_le_unpacked (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
        {
            return __internal_fp64emu_cmp_lt_unpacked(__x, __y)
                || __internal_fp64emu_cmp_eq_unpacked(__x, __y);
        }

        _CCCL_TRIVIAL_API 
        bool __internal_fp64emu_cmp_gt_unpacked (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
        {
            return __internal_fp64emu_cmp_lt_unpacked(__y, __x);
        }

        _CCCL_TRIVIAL_API 
        bool __internal_fp64emu_cmp_ge_unpacked (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
        {
            return __internal_fp64emu_cmp_le_unpacked(__y, __x);
        }

        _CCCL_TRIVIAL_API 
        bool __internal_fp64emu_cmp_ne_unpacked (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
        {
            return !__internal_fp64emu_cmp_eq_unpacked(__x, __y);
        }
 


// ============================================================================
// Builtin declarations/implementations for comparison operations
// ============================================================================
#if defined(_CCCL_FPEMU_INLINE)
#if (_CCCL_FPEMU_PACKED_VIA_UNPACKED == 1)
// Packed-via-unpacked (testing): route the packed comparison builtins through the
// unpacked cores. unpack(x) yields the fully-accurate, order-preserving form the
// unpacked comparators expect; comparison is rounding-independent, so no pack step.
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_eq (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_cmp_eq_unpacked (__internal_fp64emu_unpack (__x), __internal_fp64emu_unpack (__y)); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_ne (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_cmp_ne_unpacked (__internal_fp64emu_unpack (__x), __internal_fp64emu_unpack (__y)); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_le (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_cmp_le_unpacked (__internal_fp64emu_unpack (__x), __internal_fp64emu_unpack (__y)); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_lt (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_cmp_lt_unpacked (__internal_fp64emu_unpack (__x), __internal_fp64emu_unpack (__y)); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_ge (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_cmp_ge_unpacked (__internal_fp64emu_unpack (__x), __internal_fp64emu_unpack (__y)); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_gt (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_cmp_gt_unpacked (__internal_fp64emu_unpack (__x), __internal_fp64emu_unpack (__y)); }
#else
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_eq (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_cmp_eq (__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_ne (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_cmp_ne (__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_le (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_cmp_le (__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_lt (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_cmp_lt (__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_ge (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_cmp_ge (__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_gt (__fpbits64 __x, __fpbits64 __y) noexcept { return __internal_fp64emu_cmp_gt (__x, __y); }
#endif // _CCCL_FPEMU_PACKED_VIA_UNPACKED
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_unpacked_cmp_eq (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_cmp_eq_unpacked (__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_unpacked_cmp_ne (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_cmp_ne_unpacked (__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_unpacked_cmp_le (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_cmp_le_unpacked (__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_unpacked_cmp_lt (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_cmp_lt_unpacked (__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_unpacked_cmp_ge (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_cmp_ge_unpacked (__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_unpacked_cmp_gt (__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept { return __internal_fp64emu_cmp_gt_unpacked (__x, __y); }
#else
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_eq (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_ne (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_le (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_lt (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_ge (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_cmp_gt (__fpbits64 x, __fpbits64 y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_unpacked_cmp_eq (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_unpacked_cmp_ne (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_unpacked_cmp_le (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_unpacked_cmp_lt (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_unpacked_cmp_ge (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL  bool __fp64emu_unpacked_cmp_gt (__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept ;
#endif // _CCCL_FPEMU_INLINE


} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPEMU_IMPL_CMP_H

#if defined(_CCCL_FPEMU_API_CLASSES_DEFINED) && !defined(_CCCL_FPEMU_CMP_API_MERGED)
#define _CCCL_FPEMU_CMP_API_MERGED

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{




// ============================================================================
// API (merged from fp64emu_cmp_api.hpp)
// ============================================================================

    // Comparison operators
    template<fpemu_accuracy _Acc>  _CCCL_API inline bool operator==(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept {
        return __fp64emu_cmp_eq(__x.bits, __y.bits);
    }
    template<fpemu_accuracy _Acc>  _CCCL_API inline bool operator!=(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept {
        return __fp64emu_cmp_ne(__x.bits, __y.bits);
    }
    template<fpemu_accuracy _Acc>  _CCCL_API inline bool operator<(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept {
        return __fp64emu_cmp_lt(__x.bits, __y.bits);
    }
    template<fpemu_accuracy _Acc>  _CCCL_API inline bool operator>(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept {
        return __fp64emu_cmp_gt(__x.bits, __y.bits);
    }
    template<fpemu_accuracy _Acc>  _CCCL_API inline bool operator<=(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept {
        return __fp64emu_cmp_le(__x.bits, __y.bits);
    }
    template<fpemu_accuracy _Acc>  _CCCL_API inline bool operator>=(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept {
        return __fp64emu_cmp_ge(__x.bits, __y.bits);
    }


    // Unpacked comparison operators
    template<fpemu_accuracy _Acc>  _CCCL_API inline bool operator==(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept {
        return __fp64emu_unpacked_cmp_eq(__x.bits, __y.bits);
    }
    template<fpemu_accuracy _Acc>  _CCCL_API inline bool operator!=(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept {
        return __fp64emu_unpacked_cmp_ne(__x.bits, __y.bits);
    }
    template<fpemu_accuracy _Acc>  _CCCL_API inline bool operator<(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept {
        return __fp64emu_unpacked_cmp_lt(__x.bits, __y.bits);
    }
    template<fpemu_accuracy _Acc>  _CCCL_API inline bool operator>(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept {
        return __fp64emu_unpacked_cmp_gt(__x.bits, __y.bits);
    }
    template<fpemu_accuracy _Acc>  _CCCL_API inline bool operator<=(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept {
        return __fp64emu_unpacked_cmp_le(__x.bits, __y.bits);
    }
    template<fpemu_accuracy _Acc>  _CCCL_API inline bool operator>=(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept {
        return __fp64emu_unpacked_cmp_ge(__x.bits, __y.bits);
    }



} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_FPEMU_CMP_API_MERGED
