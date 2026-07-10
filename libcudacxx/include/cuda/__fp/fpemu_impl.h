//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_H
#define _CUDA___FP_FPEMU_IMPL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/**
 * @file fpemu_impl.h
 * @brief Internal implementation vocabulary and utilities for the FPEMU library
 *
 * This is the internal base header for the FPEMU emulation cores. It gathers all
 * library-internal (non-public) machinery in one place so that <cuda/__fp/fpemu.h>
 * and the per-operation cores (fpemu_impl_<op>.h) depend on a single impl header,
 * while <cuda/__fp/fpemu_common.h> is reserved for the public API surface
 * (the fpemu_accuracy selector and the CCCL_FPEMU_LIB / CCCL_FPEMU_INLINE knobs).
 *
 * It defines:
 * - Internal compilation-mode / decorator macros (inline vs library, host/device,
 *   the extern-"C" ABI decorator, and the builtin declaration macro)
 * - Internal vocabulary types (__fpbits64, __fpbits64_unpacked) and the internal
 *   __fpemu_rounding enum
 * - Compile-time assertion macros, bit-casting utilities, and platform-independent
 *   helper functions used by the emulation cores
 *
 * The utilities are designed to work across both host and device code through
 * appropriate decorators and provide consistent behavior across different
 * platforms and compilers.
 */
#if !defined(__CUDA_LIBDEVICE__)
    #include <cuda/std/cstdint>
    #include <cuda/std/cstring>
    #include <cuda/std/cstdlib>
#endif

#include <cuda/std/__bit/bit_cast.h>

#include <cuda/__fp/fpemu_common.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

/*********************************************************************
 * Compilation mode macros (internal)
 *********************************************************************/

// The public compile-mode knobs CCCL_FPEMU_LIB / CCCL_FPEMU_INLINE are defined
// (and mapped to the internal _CCCL_FPEMU_USE_LIB switch) in the public header
// <cuda/__fp/fpemu_common.h>. Here we only apply the internal fallback when
// nothing has been selected (e.g. the standalone libcufp build TU, which sets
// _CCCL_FPEMU_BUILD_LIB).

// If neither inline nor lib is defined internally, default to inline
#if !defined _CCCL_FPEMU_INLINE && !defined _CCCL_FPEMU_BUILD_LIB && !defined _CCCL_FPEMU_USE_LIB
    #define _CCCL_FPEMU_INLINE
#endif

// If neither host nor device is defined, default to device
#if defined __CUDACC__
    #undef  _CCCL_FPEMU_HOST
    #undef  _CCCL_FPEMU_DEVICE
    #define _CCCL_FPEMU_DEVICE
#else
    #undef  _CCCL_FPEMU_DEVICE
    #undef  _CCCL_FPEMU_HOST
    #define _CCCL_FPEMU_HOST
#endif

/*
// Custom ABI for builtins in static library
*/
#if ((defined __CUDA_LIBDEVICE__) || (defined _CCCL_FPEMU_BUILD_LIB) || (defined _CCCL_FPEMU_USE_LIB)) && \
     (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13))
  #ifndef _CCCL_FPEMU_ABI_PRESERVE_N_DATA
    #define _CCCL_FPEMU_ABI_PRESERVE_N_DATA    -1
  #endif
  #ifndef _CCCL_FPEMU_ABI_PRESERVE_N_CONTROL
    #define _CCCL_FPEMU_ABI_PRESERVE_N_CONTROL -1
  #endif
  #if (_CCCL_FPEMU_ABI_PRESERVE_N_DATA != -1) && (_CCCL_FPEMU_ABI_PRESERVE_N_CONTROL != -1)
    #define _CCCL_FPEMU_ABI_STR1(x) #x
    #define _CCCL_FPEMU_ABI_STR(x) _CCCL_FPEMU_ABI_STR1(x)
    #define _CCCL_FPEMU_ABI_PRAGMA_TEXT nv_abi preserve_n_data(_CCCL_FPEMU_ABI_PRESERVE_N_DATA) preserve_n_control(_CCCL_FPEMU_ABI_PRESERVE_N_CONTROL)
    #define _CCCL_FPEMU_ABI _Pragma(_CCCL_FPEMU_ABI_STR(_CCCL_FPEMU_ABI_PRAGMA_TEXT))
  #else
    #define _CCCL_FPEMU_ABI
  #endif
#else
  #define _CCCL_FPEMU_ABI
#endif

/*********************************************************************
 * Declaration macros (internal)
 *********************************************************************/

// Header (inline) builds decorate functions at the call site with CCCL
// visibility macros directly:
//   _CCCL_API         — public entry points (host/device, hidden from ABI)
//   _CCCL_TRIVIAL_API — force-inlined internal/impl helpers (hot paths)
//   _CCCL_DEVICE_API  — device-only overloads
// The only decorator that still needs a dedicated macro is the extern-"C"
// ABI symbol used when building or linking the standalone libcufp library.
#if defined __CUDA_LIBDEVICE__
    #define _CCCL_FPEMU_BUILTIN_DECL   _CCCL_FPEMU_ABI extern "C" _CCCL_HOST_DEVICE
#elif defined _CCCL_FPEMU_INLINE
    #define _CCCL_FPEMU_BUILTIN_DECL   _CCCL_TRIVIAL_API
#else // _CCCL_FPEMU_BUILD_LIB or _CCCL_FPEMU_USE_LIB
    #define _CCCL_FPEMU_BUILTIN_DECL   _CCCL_FPEMU_ABI extern "C" _CCCL_HOST_DEVICE
#endif

/*********************************************************************
 * Default configuration values (internal)
 *********************************************************************/

// Define the default values for the enums
#ifndef _CCCL_FPEMU_DEFAULT_ROUNDING
    #define _CCCL_FPEMU_DEFAULT_ROUNDING rn
#endif

// Route the PACKED API through the unpack -> *_unpacked core -> pack pipeline
// instead of the legacy fused kernels. Set by the Makefile's PACKED_VIA_UNPACKED=y.
// This is a TESTING knob (it lets the packed test harness exercise the unpacked
// cores); default OFF keeps the packed API on its legacy implementations.
#ifndef _CCCL_FPEMU_PACKED_VIA_UNPACKED
    #define _CCCL_FPEMU_PACKED_VIA_UNPACKED 0
#endif

// Verify that either _CCCL_FPEMU_INLINE or _CCCL_FPEMU_BUILD_LIB/_CCCL_FPEMU_USE_LIB is defined
#if !defined(_CCCL_FPEMU_INLINE) && !defined(_CCCL_FPEMU_BUILD_LIB) && !defined(_CCCL_FPEMU_USE_LIB)
    #error "ERROR: either _CCCL_FPEMU_INLINE or _CCCL_FPEMU_BUILD_LIB/_CCCL_FPEMU_USE_LIB must be defined"
#endif

/*********************************************************************
 * Internal vocabulary types and enums
 *********************************************************************/

/**
* @brief Bit representation of a double-precision floating point number
*
* @internal Library-internal type. It is the raw-bits vocabulary of the emulation
* builtins (and the extern-"C" library ABI); it is NOT part of the public C++ API.
* C++ users operate on the fpemu class (double in, overloaded ops, double out) and
* never handle __fpbits64 directly. If the builtins are ever exposed to compilers,
* a public alias can be introduced then.
*
* Holds the IEEE-754 binary encoding of a double (sign, exponent and mantissa bits).
*/
typedef uint64_t __fpbits64;

/**
* @brief Unpacked representation of a double-precision floating point number
*
* @internal Library-internal type (see __fpbits64). Represents a double in unpacked
* form (separate sign, exponent, mantissa) for the *_unpacked builtins; not public.
*/
typedef struct 
{
    uint32_t sign;
    uint32_t exponent;
    uint64_t mantissa;
} __fpbits64_unpacked;

/**
* @brief Rounding modes for floating point operations (internal)
*
* Enumeration of supported rounding modes:
* - rn: Round to nearest (ties to even) - default IEEE-754 rounding
* - rz: Round toward zero (truncation)
* - ru: Round toward positive infinity 
* - rd: Round toward negative infinity
*/
enum struct __fpemu_rounding
{
    unset = -1,
    rn    =  0,
    rz    =  1,
    ru    =  2,
    rd    =  3,
    def   = _CCCL_FPEMU_DEFAULT_ROUNDING
};

/*
 * Packed-via-unpacked TEST mode (_CCCL_FPEMU_PACKED_VIA_UNPACKED) is configured in
 * fpemu_common.h and set by the Makefile's PACKED_VIA_UNPACKED=y. When ON, the
 * packed (__fpbits64) builtins are routed through the combined unpack ->
 * *_unpacked core -> pack pipeline so the packed test harness exercises the
 * unpacked cores. When OFF (default), the legacy fused packed kernels are used
 * unchanged (byte-for-byte). The unpacked __fpbits64_unpacked ABI builtins
 * always co-exist with the packed API regardless of this flag.
 */

#if defined(__CUDA_LIBDEVICE__)
    // Map CUDA built-ins to NVVM analogs
    #define __clz(x)                __builtin_clz(x)
    #define __clzll(x)              __builtin_clzll(x)
    #define __umul64hi(x, y)        __nvvm_mulhi_ull(x, y)
    #define __ddiv_rn(x, y)         __nvvm_div_rn_d((x), (y))
    #define __dadd_rn(x, y)         __nvvm_add_rn_d((x), (y))
    #define __dsub_rn(x, y)         __nvvm_add_rn_d((x), -(y))//__nvvm_sub_rn_d((x), (y))
    #define __dmul_rn(x, y)         __nvvm_mul_rn_d((x), (y))

    #define __ddiv_rz(x, y)         __nvvm_div_rz_d((x), (y))
    #define __dadd_rz(x, y)         __nvvm_add_rz_d((x), (y))
    #define __dsub_rz(x, y)         __nvvm_add_rz_d((x), -(y))//__nvvm_sub_rz_d((x), (y))
    #define __dmul_rz(x, y)         __nvvm_mul_rz_d((x), (y))

    #define __ddiv_rd(x, y)         __nvvm_div_rm_d((x), (y))
    #define __dadd_rd(x, y)         __nvvm_add_rm_d((x), (y))
    #define __dsub_rd(x, y)         __nvvm_add_rm_d((x), -(y))//__nvvm_sub_rm_d((x), (y))
    #define __dmul_rd(x, y)         __nvvm_mul_rm_d((x), (y))

    #define __ddiv_ru(x, y)         __nvvm_div_rp_d((x), (y))
    #define __dadd_ru(x, y)         __nvvm_add_rp_d((x), (y))
    #define __dsub_ru(x, y)         __nvvm_add_rp_d((x), -(y))//__nvvm_sub_rp_d((x), (y))
    #define __dmul_ru(x, y)         __nvvm_mul_rp_d((x), (y))

    #define __dsqrt_rn(x)           __nvvm_sqrt_rn_d((x))
    #define __dsqrt_rz(x)           __nvvm_sqrt_rz_d((x))
    #define __dsqrt_rd(x)           __nvvm_sqrt_rm_d((x))
    #define __dsqrt_ru(x)           __nvvm_sqrt_rp_d((x))

    #define __fma_rn(x, y, z)       __nvvm_fma_rn_d((x), (y), (z))
    #define __fma_rz(x, y, z)       __nvvm_fma_rz_d((x), (y), (z))
    #define __fma_rd(x, y, z)       __nvvm_fma_rm_d((x), (y), (z))
    #define __fma_ru(x, y, z)       __nvvm_fma_rp_d((x), (y), (z))
#endif

/*********************************************************************
 * Forced method overrides
 *********************************************************************/

// Forced method overrides for arithmetic operations
#ifndef _CCCL_FPEMU_ADD_METHOD
    #define _CCCL_FPEMU_ADD_METHOD   unset
#endif
#ifndef _CCCL_FPEMU_MUL_METHOD
    #define _CCCL_FPEMU_MUL_METHOD   unset
#endif
#ifndef _CCCL_FPEMU_FMA_METHOD
    #define _CCCL_FPEMU_FMA_METHOD   unset
#endif
#ifndef _CCCL_FPEMU_DIV_METHOD
    #define _CCCL_FPEMU_DIV_METHOD   unset
#endif
#ifndef _CCCL_FPEMU_SQRT_METHOD
    #define _CCCL_FPEMU_SQRT_METHOD  unset
#endif


    #define _CCCL_FP32_TOTAL_BITS  32
    #define _CCCL_FP32_BIAS        127
    #define _CCCL_FP32_MANT_BITS   23
    #define _CCCL_FP32_EXP_BITS    8
    #define _CCCL_FP32_SIGN_BITS   1
    #define _CCCL_FP32_MANT_MASK   0x007fffff
    #define _CCCL_FP32_LO_EXP_MASK 0x000000FF

    #define _CCCL_FP64_TOTAL_BITS   64
    #define _CCCL_FP64_BIAS         1023
    #define _CCCL_FP64_MANT_BITS    52
    #define _CCCL_FP64_EXP_BITS     11
    #define _CCCL_FP64_SIGN_BITS    1
    #define _CCCL_FP64_HI_MANT_MASK 0x000FFFFF    
    #define _CCCL_FP64_HI_SIGN_MASK 0x80000000 
    #define _CCCL_FP64_LO_EXP_MASK  0x000007FF

    // ---- Full-width IEEE-754 binary64 bit-field masks and canonical values ----
    // Plain ULL literals (64-bit on every supported platform) avoid relying on
    // the UINT64_C() macro, which is not always available in device builds.
    #define _CCCL_FPEMU_SIGN_64          0x8000000000000000ULL   // sign bit
    #define _CCCL_FPEMU_EXP_64           0x7FF0000000000000ULL   // exponent field (also +infinity)
    #define _CCCL_FPEMU_MANT_64          0x000FFFFFFFFFFFFFULL   // 52-bit trailing significand
    #define _CCCL_FPEMU_ABS_64           0x7FFFFFFFFFFFFFFFULL   // magnitude (everything but sign)
    #define _CCCL_FPEMU_HIDDEN_64        0x0010000000000000ULL   // implicit leading 1 (2^52)
    #define _CCCL_FPEMU_QNAN_BIT_64      0x0008000000000000ULL   // quiet bit (significand MSB)
    #define _CCCL_FPEMU_SNAN_PAYLOAD_64  0x0007FFFFFFFFFFFFULL   // signaling-NaN payload (low 51 bits)
    #define _CCCL_FPEMU_INF_64           0x7FF0000000000000ULL   // +infinity
    #define _CCCL_FPEMU_QNAN_64          0x7FF8000000000000ULL   // exponent field + quiet bit (+qNaN)
    #define _CCCL_FPEMU_DEFNAN_64        0xFFF8000000000000ULL   // canonical default NaN

    // 1.0f in fp32
    #define _CCCL_FP32_ONE          0x3f800000
    // 2.0f in fp32
    #define _CCCL_FP32_TWO          0x40000000

    // Number of extra bits for precise multiplication of the mantissas
    #define _CCCL_FP64_EXTRA_BITS 9
    
    // The value to shift high part of fp64 mantissa to get the exponent
    #define _CCCL_FP64_HI_MANT_SHIFT         (_CCCL_FP32_TOTAL_BITS - _CCCL_FP64_EXP_BITS - _CCCL_FP64_SIGN_BITS)          // 20
    // The value to shift mantissas product to get the mantissa of the result (to the begin of the mantissa)
    // *2 because the mantissa is 52 bits and the result is 104 bits
    #define _CCCL_FP64_MANT_MUL_SHIFT        (_CCCL_FP64_MANT_BITS - ((_CCCL_FP64_MANT_BITS*2) - _CCCL_FP64_TOTAL_BITS))   // 12
    // The value to shift the mantissa of the result to the high part of the fp32 mantissa
    // to use fp32 computations for the fp64 mantissa
    #define _CCCL_FP64_MANT_TO_FP32_HI_SHIFT (_CCCL_FP64_EXP_BITS - _CCCL_FP32_EXP_BITS)                             // 3
    // The value to shift the mantissa of the result to the low part of the fp32 mantissa
    // to use fp32 computations for the fp64 mantissa
    #define _CCCL_FP64_MANT_TO_FP32_LO_SHIFT (_CCCL_FP32_TOTAL_BITS - _CCCL_FP64_MANT_TO_FP32_HI_SHIFT)              // 29
    // Position of the carry bit in the mantissa of the result
    // to be added to the exponent of the result
    #define _CCCL_FP64_MANT_MUL_CARRY_BIT    (((_CCCL_FP64_MANT_BITS*2) - _CCCL_FP64_TOTAL_BITS) - _CCCL_FP32_TOTAL_BITS)  // 8

    /* Total lengh of the internal representation of the mantissa  */
    //constexpr uint64_t bitwidth = (MANTISSA_WIDTH + EXTRA_BITS);
    /* Exponent bias is 2^(11-1) -1 */    
    /* fp64 mantissa is 52 */
    constexpr uint64_t __fpemu_mantissa_width = 52;
    constexpr uint64_t __fpemu_exponent_mask  = _CCCL_FPEMU_EXP_64;
    constexpr uint64_t __fpemu_mantissa_mask  = _CCCL_FPEMU_MANT_64;
    constexpr uint32_t __fpemu_extra_bits     = 9;
    constexpr uint32_t __fpemu_bias           = 1023;
    constexpr uint32_t __fpemu_inf_zero       = 0x00007ff0 - __fpemu_bias - 2048 - 1 + 0xC;; // - 128

    /*
    // by default route fpemu's internal bit-casts through
    // cuda::std::bit_cast. _CCCL_FPEMU_BIT_CAST is the single switch point --
    // define it before including the fpemu headers for a fast re-map back to the
    // in-house polyfill, e.g.:
    //   #define _CCCL_FPEMU_BIT_CAST(To, v) \
    //       ::cuda::experimental::__fpemu_builtin_bit_cast<To>(v)
    */
    #ifndef _CCCL_FPEMU_BIT_CAST
    #  define _CCCL_FPEMU_BIT_CAST(To, v) ::cuda::std::bit_cast<To>(v)
    #endif

    // In-house bit cast polyfill, kept available as the _CCCL_FPEMU_BIT_CAST
    // fallback target. Similar to C++20 std::bit_cast.
    template<typename _Tp, typename _Rp>
    _CCCL_API _Tp __fpemu_builtin_bit_cast(const _Rp __value) noexcept 
    {
        _Tp __dst;
    #if defined __DO_NOT_USE_MEMCPY__
        for (unsigned __i = 0U; __i < sizeof(_Tp); __i++)
        {
            unsigned char * __ptr_src = __i + (unsigned char *)(&__value);
            unsigned char * __ptr_dst = __i + (unsigned char *)(&__dst);
            *__ptr_dst = *__ptr_src;
        }
    #else
        #if !defined(__CUDA_LIBDEVICE__)
            std::memcpy(static_cast<void*>(&__dst), static_cast<const void*>(&__value),
                        sizeof(_Tp));
        #else
            memcpy(static_cast<void*>(&__dst), static_cast<const void*>(&__value),
                        sizeof(_Tp));
        #endif
    #endif
            return __dst;
    }

    // Internal bit cast utility used throughout the library. Delegates to the
    // _CCCL_FPEMU_BIT_CAST switch macro (cuda::std::bit_cast by default).
    template<typename _Tp, typename _Rp>
    _CCCL_API _Tp __fpemu_bit_cast(const _Rp __value) noexcept
    {
        return _CCCL_FPEMU_BIT_CAST(_Tp, __value);
    }
    


    /**
     * @brief Enumeration for classifying floating point numbers
     * 
     * This enum class defines the different categories that a floating point number
     * can belong to according to IEEE-754:
     *
     * - normal: A normalized floating point number with an implicit leading 1
     * - zero: Positive or negative zero (+0.0 or -0.0)  
     * - inf: Positive or negative infinity
     * - nan: Not a Number (quiet or signaling)
     * - denormal: Denormalized number with leading zeros
     */
    enum struct __fpclass
    {
        normal     = 0,
        zero       = 1,
        inf        = 2,
        nan        = 3,
        denormal   = 4
    };    
    
    /**
     * @brief Structure representing an unpacked double precision floating point number
     * 
     * This structure stores the components of a double precision floating point number
     * in an unpacked format for easier manipulation:
     *
     * @var mantissa The significand/mantissa bits including the implicit leading 1 for normalized numbers
     * @var exponent The unbiased exponent value 
     * @var sign The sign bit (0 for positive, 1<<31 for negative)
     * @var fpclass The floating point class (normal, zero, inf, nan, denormal)
     *
     * @note `fpclass` carries a default member initializer (`__fpclass::normal`)
     *       so that any code path producing an `__fp64emu_unpacked` that does not
     *       explicitly classify the value (e.g. intermediate result structs
     *       in fma/add/mul) leaves the field with
     *       a defined value. Reading an indeterminate `fpclass` is undefined
     *       behavior; UB-aware optimizers (NVVM 22.0.0 on sm_120/121) will
     *       happily DCE entire downstream computations on the assumption it
     *       cannot happen — observed previously as `fma_device_impl` emitting
     *       just `ret;` and callers reading zeros. The other fields stay
     *       uninitialized by design (they are written unconditionally by every
     *       producer and we don't want to pessimize the optimizer's SROA).
     */
    struct __fp64emu_unpacked 
    {
        uint64_t mantissa;
        int32_t exponent;
        uint32_t sign;
        __fpclass fpclass = __fpclass::normal;
    };

    /**
     * @brief Structure representing a 64-bit integer split into two 32-bit parts
     * 
     * This structure is used to store a 64-bit integer as two 32-bit parts,
     * which is useful for certain operations that require 64-bit arithmetic
     * but can be performed by 32-bit operations.    
     */
    struct __uint32x2 
    {
        uint32_t x[2];
    };

    /**
     * @brief Structure representing a 64-bit integer split into two 32-bit parts
     * 
     * This structure is used to store a 64-bit integer as two 32-bit parts,
     * which is useful for certain operations that require 64-bit arithmetic
     * but can be performed by 32-bit operations.    
     */
    struct __uint64x2 
    {
        uint64_t x[2];
    };

    /**
     * @brief Structure representing a 128-bit integer split into two 64-bit parts
     * 
     * This structure is used to store a 128-bit integer as two 64-bit parts,
     * which is useful for certain operations that require 128-bit arithmetic
     * but can be performed by 64-bit operations.    
     */
    struct __uint32x4
    {
        __uint32x2 lo;
        __uint32x2 hi;
    };

    #ifndef __CUDA_ARCH__
        /**
         * @brief Count leading zeros in a 32-bit integer
         * 
         * This function counts the number of leading zeros in a 32-bit integer.
         * It works by shifting the integer right until the most significant bit is found.
         *   
         * @param x The 32-bit integer to count leading zeros in
         * @return The number of leading zeros in the integer
         */
        _CCCL_TRIVIAL_API int __internal_clz(int __x) noexcept 
        {
            if (__x == 0) return 32;

            int __n = 0;
            unsigned int __u = (unsigned int)__x;

            if ((__u >> 16) == 0) { __n += 16; __u <<= 16; }
            if ((__u >> 24) == 0) { __n += 8;  __u <<= 8;  }
            if ((__u >> 28) == 0) { __n += 4;  __u <<= 4;  }
            if ((__u >> 30) == 0) { __n += 2;  __u <<= 2;  }
            if ((__u >> 31) == 0) { __n += 1; }

            return __n;
        }

        /**
         * @brief Count leading zeros in a 64-bit integer
         * 
         * This function counts the number of leading zeros in a 64-bit integer.
         * It works by shifting the integer right until the most significant bit is found.
         * 
         * @param x The 64-bit integer to count leading zeros in
         * @return The number of leading zeros in the integer
         */
        _CCCL_TRIVIAL_API int __internal_clzll(int64_t __x) noexcept 
        {
            uint64_t __ux = (uint64_t)__x;
            if (__ux == 0)
                return 64;

            int __count = 0;
            for (int __i = 63; __i >= 0; --__i) {
                if (__ux & (1ULL << __i))
                    break;
                __count++;
            }
            return __count;
        }
    #else
        _CCCL_TRIVIAL_API int __internal_clz(int __x) noexcept       { return __clz(__x); }
        _CCCL_TRIVIAL_API int __internal_clzll(int64_t __x) noexcept { return __clzll(__x); }
    #endif //__CUDA_ARCH__

        #undef  _CCCL_FPEMU_MAX
    #if defined(__CUDA_ARCH__) && !defined(__CUDA_LIBDEVICE__)
        // Global-scope qualifier: inside namespace cuda::experimental an
        // unqualified `max` now resolves to the fpmp2 max() template, which
        // shadows the CUDA device `::max(int, int)` builtin we want here.
        #define _CCCL_FPEMU_MAX      ::max
    #else
        #define _CCCL_FPEMU_MAX(a, b) ((a) > (b) ? (a) : (b))
    #endif

    #ifndef _CCCL_FP64EMU_PTX_XOR
      #define _CCCL_FP64EMU_PTX_XOR 0
    #endif
    /**
     * @brief Invert the most significant bit of a 32-bit integer
     * 
     * This function inverts the MSB (most significant bit) of a 32-bit integer.
     * Uses PTX inline assembly for CUDA to avoid unwanted compiler optimizations.
     * 
     * @param sign The 32-bit integer to invert the sign of
     * @return The inverted sign
     */
    _CCCL_TRIVIAL_API uint32_t __invert_msb(uint32_t __sign) noexcept
    {
    #if _CCCL_FP64EMU_PTX_XOR == 1
        uint32_t result;
        asm ("{\n\t"
            ".reg .u32 r0;\n\t"
            "xor.b32 r0, %1, 0x80000000;\n\t"
            "mov.b32 %0, r0;\n\t"
            "}"
            : "=r"(result)
            : "r"(sign));
        return result;
    #else
        return __sign ^ 0x80000000u;
    #endif
    }


    /**
     * @brief Multiply two 64-bit integers and return the high 32 bits of the result
     * 
     * This function performs multiplication of two 64-bit integers represented as pairs
     * of 32-bit integers (__uint32x2). It returns only the high 32 bits of the 128-bit
     * multiplication result.
     * 
     * The implementation is optimized differently for CUDA and CPU:
     * - On CUDA: Uses PTX assembly instructions (mad.hi.u32) for efficient high-bit multiplication
     * - On CPU: Uses standard C++ arithmetic with careful handling of carries and partial products
     * 
     * The CPU implementation:
     * 1. Computes all partial products (lo*lo, lo*hi, hi*lo, hi*hi)
     * 2. Handles carries between the partial products
     * 3. Extracts and combines the high bits to form the final 32-bit result
     * 
     * @tparam _Acc The accuracy level (fpemu_accuracy)
     * @param a_ First 64-bit multiplicand as two 32-bit integers
     * @param b_ Second 64-bit multiplicand as two 32-bit integers
     * @return The high 32 bits of the multiplication result
     */
    template<fpemu_accuracy _Acc = fpemu_accuracy::high>
    _CCCL_TRIVIAL_API uint32_t __mul_32(__uint32x2 __a, __uint32x2 __b) noexcept
    {
        uint32_t __res;
#if defined  __CUDA_ARCH__
        __uint32x2 __a64 = __fpemu_bit_cast<__uint32x2>(__a);
        __uint32x2 __b64 = __fpemu_bit_cast<__uint32x2>(__b);
        uint32_t __res32;
        asm ("{\n\t"
            ".reg .u32 r0, ahi, bhi;\n\t"
            "mov.b32         ahi, %1;   \n\t"
            "mov.b32         bhi, %2;   \n\t"
            "mad.hi.u32      r0, ahi, bhi,  0;\n\t"
            "mov.b32         %0, r0;     \n\t"
            "}"
            : "=r"(__res32)
            : "r"(__a64.x[1]), "r"(__b64.x[1]));
        __res = __res32;
#else
        // Split inputs into 32-bit parts
        uint32_t __ahi = __a.x[1];
        uint32_t __bhi = __b.x[1];

        // Compute high 32 bits directly
        uint64_t __hi_hi = uint64_t(__ahi) * __bhi;  // (a.hi * b.hi)
        __res = uint32_t(__hi_hi >> 32);
#endif
        return __res;
    } //__mul_32


    /**
     * @brief Multiply two 64-bit integers and return the high 64 bits of the result
     * 
     * This function performs multiplication of two 64-bit integers represented as pairs
     * of 32-bit integers (__uint32x2). It returns the high 64 bits of the 128-bit result.
     * 
     * The implementation is optimized differently for CUDA and CPU:
     * - On CUDA: Uses PTX assembly instructions for efficient 64-bit multiplication
     * - On CPU: Uses standard C++ arithmetic with careful handling of carries
     * 
     * @tparam _Acc The accuracy level (fpemu_accuracy)
     * @param a_ First 64-bit multiplicand as two 32-bit integers
     * @param b_ Second 64-bit multiplicand as two 32-bit integers
     * @return The high 64 bits of the multiplication result as two 32-bit integers
     */
    template<fpemu_accuracy _Acc = fpemu_accuracy::high>
    _CCCL_TRIVIAL_API __uint32x2 __mul_64(__uint32x2 __a, __uint32x2 __b) noexcept
    {
        __uint32x2 __res;
#if defined  __CUDA_ARCH__
        uint64_t __a64 = __fpemu_bit_cast<uint64_t>(__a);
        uint64_t __b64 = __fpemu_bit_cast<uint64_t>(__b);
        uint64_t __res64;
        asm ("{\n\t"
            ".reg .u32 r0, r1, r2, r3, alo, ahi, blo, bhi;\n\t"
            "mov.b64         {alo,ahi}, %1;   \n\t"
            "mov.b64         {blo,bhi}, %2;   \n\t"
            "mad.lo.cc.u32   r1, alo, bhi, 0;\n\t"
            "madc.hi.u32     r2, alo, bhi,  0;\n\t"
            "mad.lo.cc.u32   r1, ahi, blo, r1;\n\t"
            "madc.hi.cc.u32  r2, ahi, blo, r2;\n\t"
            "madc.hi.u32     r3, ahi, bhi,  0;\n\t"
            "mad.lo.cc.u32   r2, ahi, bhi, r2;\n\t"
            "addc.u32        r3, r3,  0;      \n\t"
            "mov.b64         %0, {r2,r3};     \n\t"
            "}"
            : "=l"(__res64)
            : "l"(__a64), "l"(__b64));
        __res = __fpemu_bit_cast<__uint32x2>(__res64);
#else
        // Split inputs into 32-bit parts
        uint32_t __alo = __a.x[0], __ahi = __a.x[1];
        uint32_t __blo = __b.x[0], __bhi = __b.x[1];

        // Compute partial products that contribute to high 64 bits
        uint64_t __lo_hi = uint64_t(__alo) * __bhi;  // (a.lo * b.hi)
        uint64_t __hi_lo = uint64_t(__ahi) * __blo;  // (a.hi * b.lo)
        uint64_t __hi_hi = uint64_t(__ahi) * __bhi;  // (a.hi * b.hi)

        // Extract high parts and combine
        uint32_t __lo_hi_hi = uint32_t(__lo_hi >> 32);
        uint32_t __hi_lo_hi = uint32_t(__hi_lo >> 32);
        uint32_t __hi_hi_lo = uint32_t(__hi_hi);
        uint32_t __hi_hi_hi = uint32_t(__hi_hi >> 32);

        // Combine high parts with carries
        uint64_t __hi = uint64_t(__lo_hi_hi) + __hi_lo_hi + __hi_hi_lo;
        
        // Store high result
        __res.x[0] = uint32_t(__hi);
        __res.x[1] = uint32_t(__hi >> 32) + __hi_hi_hi;
#endif
        return __res;
    } //__mul_64


    /**
     * @brief Multiply two 64-bit integers and return the full 128-bit result
     * 
     * This function performs multiplication of two 64-bit integers represented as pairs
     * of 32-bit integers (__uint32x2). It returns the full 128-bit result stored in
     * a __uint32x4 structure containing both high and low 64 bits.
     * 
     * The implementation handles the multiplication by:
     * 1. Computing partial products (lo*lo, lo*hi, hi*lo, hi*hi)
     * 2. Properly handling carries between the partial products
     * 3. Combining the results into the final 128-bit value
     * 
     * The implementation is optimized differently for CUDA and CPU:
     * - On CUDA: Uses built-in __umul64hi for high bits
     * - On CPU: Uses standard C++ arithmetic with careful handling of carries
     * 
     * @tparam _Acc The accuracy level (fpemu_accuracy)
     * @param a_ First 64-bit multiplicand as two 32-bit integers
     * @param b_ Second 64-bit multiplicand as two 32-bit integers
     * @return The full 128-bit multiplication result as four 32-bit integers
     */
    template<fpemu_accuracy _Acc = fpemu_accuracy::high>
    _CCCL_TRIVIAL_API __uint32x4 __mul_128(__uint32x2 __a, __uint32x2 __b) noexcept
    {
        __uint32x4 __res;

        // Split inputs into 32-bit parts
        uint32_t __alo = __a.x[0], __ahi = __a.x[1];
        uint32_t __blo = __b.x[0], __bhi = __b.x[1];

        // Compute partial products
        uint64_t __lo_lo = uint64_t(__alo) * __blo;  // (a.lo * b.lo)
        uint64_t __lo_hi = uint64_t(__alo) * __bhi;  // (a.lo * b.hi) 
        uint64_t __hi_lo = uint64_t(__ahi) * __blo;  // (a.hi * b.lo)

        // Extract 32-bit parts of intermediate results
        uint32_t __lo_lo_lo = uint32_t(__lo_lo);
        uint32_t __lo_lo_hi = uint32_t(__lo_lo >> 32);
        uint32_t __lo_hi_lo = uint32_t(__lo_hi);
        uint32_t __hi_lo_lo = uint32_t(__hi_lo);

        // Compute final low result with carries
        uint64_t __mid = uint64_t(__lo_lo_hi) + __lo_hi_lo + __hi_lo_lo;
        uint32_t __mid_lo = uint32_t(__mid);
        
        // Store low result
        __res.lo.x[0] = __lo_lo_lo;
        __res.lo.x[1] = __mid_lo;

#if defined  __CUDA_ARCH__
        uint64_t __a64 = __fpemu_bit_cast<uint64_t>(__a);
        uint64_t __b64 = __fpemu_bit_cast<uint64_t>(__b);
        uint64_t __res64 = __umul64hi(__a64, __b64);
        __res.hi = __fpemu_bit_cast<__uint32x2>(__res64);
#else
        uint64_t __hi_hi = uint64_t(__ahi) * __bhi;  // (a.hi * b.hi)
        uint32_t __lo_hi_hi = uint32_t(__lo_hi >> 32);
        uint32_t __hi_lo_hi = uint32_t(__hi_lo >> 32);
        uint32_t __hi_hi_lo = uint32_t(__hi_hi);
        uint32_t __mid_hi = uint32_t(__mid >> 32);
        uint64_t __hi = uint64_t(__mid_hi) + __lo_hi_hi + __hi_lo_hi + __hi_hi_lo;
        uint32_t __hi_hi_hi = uint32_t(__hi_hi >> 32);

        // Store high result
        __res.hi.x[0] = uint32_t(__hi);
        __res.hi.x[1] = uint32_t(__hi >> 32) + __hi_hi_hi;
#endif
        return __res;
    } //__mul_128

    /**
     * @brief Shift a 64-bit value left by a specified amount
     * 
     * This function performs a logical left shift on a 64-bit value represented
     * as two 32-bit integers. The shift amount can be positive or negative.
     * 
     * @param man The 64-bit value to shift as two 32-bit integers
     * @param shift The number of bits to shift (positive for left shift)
     * @return The shifted value as two 32-bit integers
     */
     _CCCL_TRIVIAL_API __uint32x2 __shl_64 (__uint32x2 __man, int __shift) noexcept
     {
         uint64_t __man64 = __fpemu_bit_cast<uint64_t>(__man);
         __man64 <<= __shift;
         return __fpemu_bit_cast<__uint32x2>(__man64);
     } //__shl_64
 
     /**
      * @brief Shift a 64-bit value right by a specified amount
      * 
      * This function performs a logical right shift on a 64-bit value represented
      * as two 32-bit integers. The shift amount can be positive or negative.
      * 
      * @param man The 64-bit value to shift as two 32-bit integers
      * @param shift The number of bits to shift (positive for right shift)
      * @return The shifted value as two 32-bit integers
      */
     _CCCL_TRIVIAL_API __uint32x2 __shr_64 (__uint32x2 __man, int __shift) noexcept
     {
         uint64_t __man64 = __fpemu_bit_cast<uint64_t>(__man);
 #ifndef __CUDA_ARCH__
         __shift = (__shift > 0) ? (__shift>64) ? 64 : __shift : 0;
 #endif
         __man64 = __man64 >> __shift;
         return __fpemu_bit_cast<__uint32x2>(__man64);
     } //__shr_64

     /**
      * @brief Logical right shift with directed rounding (HA/def paths)
      *
      * Truncates toward zero for rz and rn. For ru/rd, increments the result
      * when any discarded bit is set.
      *
      * @tparam rm  Rounding mode (default: nearest-even, treated as truncation)
      * @param man  The 64-bit value to shift as two 32-bit integers
      * @param shift The number of bits to shift (positive for right shift)
      * @param sign Result sign (used for directed rounding modes ru/rd)
      * @return The shifted value as two 32-bit integers
      */
    /**
     * @brief Single-precision multiply with directed rounding (fast mul paths)
     *
     * Uses CUDA intrinsics on device; host falls back to native multiply (RN).
     */
    template<__fpemu_rounding _Rm = __fpemu_rounding::rn>
    _CCCL_TRIVIAL_API float __fmul_dir (float __x, float __y) noexcept
    {
#if defined(__CUDA_ARCH__)
        if constexpr (_Rm == __fpemu_rounding::rn) return __fmul_rn(__x, __y);
        else if constexpr (_Rm == __fpemu_rounding::rz) return __fmul_rz(__x, __y);
        else if constexpr (_Rm == __fpemu_rounding::ru) return __fmul_ru(__x, __y);
        else return __fmul_rd(__x, __y);
#else
        (void)_Rm;
        return __x * __y;
#endif
    } //__fmul_dir

    /**
     * @brief Single-precision add with directed rounding (fast add paths)
     *
     * Uses CUDA intrinsics on device; host falls back to native add (RN).
     */
    template<__fpemu_rounding _Rm = __fpemu_rounding::rn>
    _CCCL_TRIVIAL_API float __fadd_dir (float __x, float __y) noexcept
    {
#if defined(__CUDA_ARCH__)
        if constexpr (_Rm == __fpemu_rounding::rn) return __fadd_rn(__x, __y);
        else if constexpr (_Rm == __fpemu_rounding::rz) return __fadd_rz(__x, __y);
        else if constexpr (_Rm == __fpemu_rounding::ru) return __fadd_ru(__x, __y);
        else return __fadd_rd(__x, __y);
#else
        (void)_Rm;
        return __x + __y;
#endif
    } //__fadd_dir

     template<__fpemu_rounding _Rm = __fpemu_rounding::rn>
     _CCCL_TRIVIAL_API __uint32x2 __shr_64_rnd (__uint32x2 __man, int __shift, bool __sign = false) noexcept
     {
         uint64_t __man64 = __fpemu_bit_cast<uint64_t>(__man);
 #ifndef __CUDA_ARCH__
         __shift = (__shift > 0) ? (__shift>64) ? 64 : __shift : 0;
 #endif
         if (__shift <= 0) return __man;

         const uint64_t __discard_mask = (__shift >= 64) ? ~0ULL : ((1ULL << __shift) - 1);
         const bool __inexact = (__man64 & __discard_mask) != 0;
         __man64 >>= __shift;

         if constexpr (_Rm == __fpemu_rounding::ru)
         {
             if (!__sign && __inexact) __man64++;
         }
         else if constexpr (_Rm == __fpemu_rounding::rd)
         {
             if (__sign && __inexact) __man64++;
         }
         return __fpemu_bit_cast<__uint32x2>(__man64);
     } //__shr_64_rnd

    /**
     * @brief Logical right shift of 128-bit mantissa with directed rounding
     */
    template<__fpemu_rounding _Rm = __fpemu_rounding::rn>
    _CCCL_TRIVIAL_API __uint128_t __shr_128_rnd (__uint128_t __man, int __shift, bool __sign = false) noexcept
    {
#ifndef __CUDA_ARCH__
        __shift = (__shift > 0) ? ((__shift > 127) ? 127 : __shift) : 0;
#endif
        if (__shift <= 0) return __man;

        const __uint128_t __discard_mask =
            (__shift >= 128) ? ~(__uint128_t)0 : ((((__uint128_t)1) << __shift) - 1);
        const bool __inexact = (__man & __discard_mask) != 0;
        __man >>= __shift;

        if constexpr (_Rm == __fpemu_rounding::rn || _Rm == __fpemu_rounding::rz)
        {
            if (__inexact) __man |= 1;
        }
        else if constexpr (_Rm == __fpemu_rounding::ru)
        {
            if (!__sign && __inexact) __man |= 1;
        }
        else if constexpr (_Rm == __fpemu_rounding::rd)
        {
            if (__sign && __inexact) __man |= 1;
        }
        return __man;
    } //__shr_128_rnd

    /**
     * @brief Logical right shift of 128-bit mantissa with jam (sticky) only.
     * Used during FMA alignment; directed rounding is deferred to the pack epilogue.
     */
    _CCCL_TRIVIAL_API __uint128_t __shr_128_jam (__uint128_t __man, int __shift) noexcept
    {
        return __shr_128_rnd<__fpemu_rounding::rn>(__man, __shift);
    } //__shr_128_jam
 
          /**
      * @brief Arithmetic Shift a 64-bit value right with rounding
      * 
      * This function performs a arithmetic right shift on a 64-bit value represented
      * as two 32-bit integers, with rounding to the nearest value. The shift amount
      * can be positive or negative.
      * 
      * @param man The 64-bit value to shift as two 32-bit integers
      * @param shift The number of bits to shift (positive for right shift)
      * @return The shifted and rounded value as two 32-bit integers
      */
      template<fpemu_accuracy _Acc = fpemu_accuracy::high>
     _CCCL_TRIVIAL_API __uint32x2 __sar_64 (__uint32x2 __man, int __shift) noexcept
     {
 #ifndef __CUDA_ARCH__
         __shift = (__shift > 0) ? (__shift>63) ? 63 : __shift : 0;
 #endif
         int64_t __man64 = __fpemu_bit_cast<int64_t>(__man);
        __man64 = __man64 >> __shift;
         __uint32x2 __res = __fpemu_bit_cast<__uint32x2>(__man64);
         return __res;
     } //__sar_64_rnd



     /**
      * @brief Arithmetic Shift a 64-bit value right with rounding
      * 
      * This function performs a arithmetic right shift on a 64-bit value represented
      * as two 32-bit integers, with rounding according to the specified mode.
      * The shift amount can be positive or negative.
      * 
      * @tparam _Acc Accuracy level (sticky-bit preservation applies only for high)
      * @tparam rm  Rounding mode (default: nearest-even)
      * @param man  The 64-bit value to shift as two 32-bit integers
      * @param shift The number of bits to shift (positive for right shift)
      * @param sign Result sign (used for directed rounding modes ru/rd)
      * @return The shifted and rounded value as two 32-bit integers
      */
      template<fpemu_accuracy   _Acc = fpemu_accuracy::high,
               __fpemu_rounding _Rm  = __fpemu_rounding::rn>
     _CCCL_TRIVIAL_API __uint32x2 __sar_64_rnd (__uint32x2 __man, int __shift, bool __sign = false) noexcept
     {
 #ifndef __CUDA_ARCH__
         __shift = (__shift > 0) ? (__shift>63) ? 63 : __shift : 0;
 #endif
         int64_t __man64 = __fpemu_bit_cast<int64_t>(__man);
         int64_t __man64_res = __man64 >> __shift;
         __uint32x2 __res = __fpemu_bit_cast<__uint32x2>(__man64_res);
         if constexpr (_Acc == fpemu_accuracy::high)
         {
            uint64_t __mask = (1LLU << __shift) - 1;
            bool __sticky = (__man64 & __mask) != 0;
            if constexpr (_Rm == __fpemu_rounding::rn)
            {
                __res.x[0] |= __sticky;
            }
            else if constexpr (_Rm == __fpemu_rounding::rz)
            {
                // truncate toward zero
            }
            else if constexpr (_Rm == __fpemu_rounding::ru)
            {
                if (!__sign && __sticky) __res.x[0] |= 1;
            }
            else if constexpr (_Rm == __fpemu_rounding::rd)
            {
                if (__sign && __sticky) __res.x[0] |= 1;
            }
         }
         return __res;
     } //__sar_64_rnd
 

    /**
     * @brief Unpack the exponent from a 64-bit floating point number
     * 
     * This function extracts the exponent from a 64-bit floating point number
     * represented as two 32-bit integers. It handles the exponent extraction
     * according to IEEE-754 double precision format.
     * 
     * @tparam _Acc Accuracy level (full-range special handling only for high)
     * @param input The input number as two 32-bit integers
     * @return The extracted exponent value
     */
     template<fpemu_accuracy _Acc = fpemu_accuracy::high>
    _CCCL_TRIVIAL_API int32_t __unpack_exp(__uint32x2 __input) noexcept
    {
        int32_t __exp;

        //remove sign bit
        __input.x[1] &= 0x7fffffff;

        //Shift exponent bits to the beggining
        __exp = __input.x[1] >> 20;

        //Extract mantissa bits
        __input.x[1] = __input.x[1] & 0x000fffff;

        //Check for NAN and INF
        if constexpr (_Acc == fpemu_accuracy::high) 
        {
            if (__exp == 0x7ff) 
            {
                if (__input.x[0] == 0 && __input.x[1] == 0) 
                {
                    __exp = 0x000017ff; //Magic value for INF
                } 
                else 
                {
                    __exp = 0x000047ff; //Magic value for NAN
                }
            }
        }

        return __exp;
    } //__unpack_exp

    /**
     * @brief Unpack the mantissa from a 64-bit floating point number
     * 
     * This function extracts the mantissa from a 64-bit floating point number
     * represented as two 32-bit integers. It handles both normal and denormal
     * numbers according to IEEE-754 double precision format.
     * 
     * @tparam twos_comp_flag Whether to use two's complement representation
     * @tparam _Acc Accuracy level (full-range special handling only for high)
     * @param sign Pointer to store the sign bit
     * @param input The input number as two 32-bit integers
     * @param is_zero_exp Whether the exponent is zero (denormal case)
     * @return The extracted mantissa as two 32-bit integers
     */
    template<fpemu_accuracy _Acc = fpemu_accuracy::high>
    _CCCL_TRIVIAL_API __uint32x2 __unpack_mant(bool *__sign, __uint32x2 __input, bool __is_zero_exp) noexcept
    {
        __uint32x2 __man32x2;

        //Extract sign
        *__sign = (__input.x[1] & 0x80000000) != 0;

        //Extract mantissa bits
        __man32x2.x[0] = __input.x[0];
        __man32x2.x[1] = __input.x[1] & 0x000fffff;

        //Set implicit-bit for normal numbers
        if (!__is_zero_exp) __man32x2.x[1] |= ( 1 << 20 );

        __man32x2 = __shl_64(__man32x2, __fpemu_extra_bits);
        return __man32x2;
    } //__unpack_mant

    /**
     * @brief Saturate unbiased exponent and mantissa on fp64 overflow
     *
     * Sets exp (unbiased exponent field, 0..2047) and man (without exponent
     * in man.x[1]) according to rounding mode. Caller folds exp into man via
     * man.x[1] |= (uint32_t)exp << _CCCL_FP64_HI_MANT_SHIFT when needed.
     *
     * @tparam rm   Rounding mode
     * @param sign  Result sign (true for negative)
     * @param exp   Output unbiased exponent field
     * @param man   Output mantissa as two 32-bit integers
     */
    template<__fpemu_rounding _Rm = __fpemu_rounding::rn>
    _CCCL_TRIVIAL_API
    void __fp64_ovfl_sat (bool __sign, int32_t& __exp, __uint32x2& __man) noexcept
    {
        if constexpr (_Rm == __fpemu_rounding::rz)
        {
            __exp = _CCCL_FP64_BIAS * 2;
            __man = {0xffffffff, _CCCL_FP64_HI_MANT_MASK};
        }
        else if constexpr (_Rm == __fpemu_rounding::rn)
        {
            __exp = _CCCL_FP64_BIAS * 2 + 1;
            __man = {0, 0};
        }
        else if constexpr (_Rm == __fpemu_rounding::ru)
        {
            if (__sign)
            {
                __exp = _CCCL_FP64_BIAS * 2;
                __man = {0xffffffff, _CCCL_FP64_HI_MANT_MASK};
            }
            else
            {
                __exp = _CCCL_FP64_BIAS * 2 + 1;
                __man = {0, 0};
            }
        }
        else // rd
        {
            if (__sign)
            {
                __exp = _CCCL_FP64_BIAS * 2 + 1;
                __man = {0, 0};
            }
            else
            {
                __exp = _CCCL_FP64_BIAS * 2;
                __man = {0xffffffff, _CCCL_FP64_HI_MANT_MASK};
            }
        }
    } //__fp64_ovfl_sat

    /**
     * @brief Pack floating point components into a 64-bit double
     * 
     * This function combines the sign, exponent, and mantissa components
     * into a 64-bit double precision floating point number according to
     * IEEE-754 format.
     * 
     * @tparam _Acc Accuracy level (full-range special handling only for high)
     * @tparam rm  Rounding mode for overflow/underflow saturation (default: nearest-even)
     * @param sign The sign bit (true for negative)
     * @param exp The exponent value
     * @param man The mantissa as two 32-bit integers
     * @return The packed 64-bit double precision number
     */
    template<fpemu_accuracy   _Acc = fpemu_accuracy::high,
             __fpemu_rounding _Rm  = __fpemu_rounding::rn>
    _CCCL_TRIVIAL_API uint64_t __pack (bool __sign, uint32_t __exp, __uint32x2 __man) noexcept
    {

        bool __zero_mantissa = __man.x[0] == 0 && __man.x[1] == 0;
        bool __is_nan = false;
        // Check for NAN
        if constexpr (_Acc == fpemu_accuracy::high)
        {
            __is_nan =
                __exp >= (0x000047ff - __fpemu_bias - 52 - 1) ||     //NAN * x && NAN + x
                __exp == (0x000017ff - __fpemu_bias - 52 - 1) ||     //INF * 0
                (__exp == 0x000017ff - 64 && __zero_mantissa); //INF + (-INF)

        }
        // A true infinity (inf+x, inf*x, inf+inf/inf*inf, etc.) reaches pack with an
        // inf-magic exponent and is NOT one of the NaN cases above. It must stay
        // infinity regardless of rounding mode. A finite overflow instead arrives
        // with a much smaller exponent (add tops out near 0x7fe; mul at dmax*dmax
        // reaches ~3069) and only trips the man.x[1] += exp carry below, where it is
        // correctly saturated per rounding mode. The two bands are well separated:
        // any inf operand contributes the 0x17ff magic, which lands at pack no lower
        // than (0x17ff - BIAS - 52 - 1) -- the very value the INF*0 NaN test uses --
        // far above the largest finite-overflow exponent. Reuse that magic as the
        // inf floor so the discriminator is operation-independent.
        bool __is_inf = false;
        if constexpr (_Acc == fpemu_accuracy::high)
        {
            __is_inf = (__exp >= (0x000017ff - __fpemu_bias - 52 - 1)) && !__is_nan;
        }
        // Check for exact zero result
        if ( __zero_mantissa && __exp < 0x000007ff) __exp = 0;
    
        if constexpr (_Acc == fpemu_accuracy::high)
        {
            // Convert special value of NAN|INF back to IEEE
            if (__exp >= 0x000007ff) __exp = 0x000007ff;
        }


        // Shift exponent back to high bits
        __exp <<= 20; 

        // Adds up exponent and mantissa to count implicit-bit
        __man.x[1] += __exp;

        // Check for the final overflow
        if (__man.x[1] >= 0x7ff00000) 
        {
            if (__is_nan)
            {
                __man.x[0] = 0;
                __man.x[1] = 0x7fffffff;
            }
            else if (__is_inf)
            {
                // True infinity: emit inf for every rounding mode (do NOT saturate
                // to DBL_MAX as a finite overflow would under rz/ru/rd).
                __man.x[0] = 0;
                __man.x[1] = 0x7ff00000;
            }
            else
            {
                int32_t __sat_exp = 0;
                __fp64_ovfl_sat<_Rm>(__sign, __sat_exp, __man);
                __man.x[1] |= (uint32_t)__sat_exp << _CCCL_FP64_HI_MANT_SHIFT;
            }
        }


        // Pack everything to FP64
        __uint32x2 __res = {__man.x[0], __man.x[1] | (__sign << 31)};

        return __fpemu_bit_cast<uint64_t>(__res);
    } //__pack


  


    /**
     * @brief Convert a 64-bit value to its two's complement
     * 
     * This function computes the two's complement of a 64-bit value represented
     * as two 32-bit integers. The operation is performed by inverting all bits
     * and adding 1.
     * 
     * @param c The 64-bit value to convert as two 32-bit integers
     * @return The two's complement of the input value
     */
    _CCCL_TRIVIAL_API __uint32x2 __two_comp (__uint32x2 __c) noexcept
    {
        uint64_t __c64   = __fpemu_bit_cast<uint64_t>(__c);
        uint64_t __res64 = 0 - __c64; //IMAD.WIDE.U32 Rd, RZ, RZ, -Rc
        return __fpemu_bit_cast<__uint32x2>(__res64);
    } //__imad_wide_sub



    /**
     * @brief Find the position of the most significant set bit in a signed 64-bit value
     * 
     * This function determines the position of the most significant set bit
     * in a 64-bit signed integer represented as two 32-bit integers.
     * 
     * @param x The 64-bit value to analyze as two 32-bit integers
     * @return The position of the most significant set bit (0-based)
     */
    _CCCL_TRIVIAL_API int32_t __flo_s64 (__uint32x2 __x) noexcept
    {
        int64_t __x64 = __fpemu_bit_cast<int64_t>(__x);
        return __internal_clzll(__x64<<1);
    } //__flo_s64

    /**
     * @brief Find the position of the most significant set bit in an unsigned 64-bit value
     * 
     * This function determines the position of the most significant set bit
     * in a 64-bit unsigned integer represented as two 32-bit integers.
     * 
     * @param x The 64-bit value to analyze as two 32-bit integers
     * @return The position of the most significant set bit (0-based)
     */
    _CCCL_TRIVIAL_API int32_t __flo_u64 (__uint32x2 __x) noexcept
    {
        uint64_t __x64 = __fpemu_bit_cast<uint64_t>(__x);
        //Skip sign bit
        return __internal_clzll(__x64 & _CCCL_FPEMU_ABS_64);
    } //__flo_u64

    /**
     * @brief Add two 64-bit unsigned integers
     * 
     * This function adds two 64-bit unsigned integers represented as pairs
     * of 32-bit integers, handling carry propagation between the halves.
     * 
     * @param a First operand as two 32-bit integers
     * @param b Second operand as two 32-bit integers
     * @return The sum as two 32-bit integers
     */
    _CCCL_TRIVIAL_API __uint32x2 __iadd_u64 (__uint32x2 __a, __uint32x2 __b) noexcept
    {
        uint64_t __a64 = __fpemu_bit_cast<uint64_t>(__a);
        uint64_t __b64 = __fpemu_bit_cast<uint64_t>(__b);
        uint64_t __res64 = __a64 + __b64;
        return __fpemu_bit_cast<__uint32x2>(__res64);
    } //__iadd_u64

    /**
     * @brief Subtract two 64-bit unsigned integers
     * 
     * This function subtracts two 64-bit unsigned integers represented as pairs
     * of 32-bit integers, handling carry propagation between the halves.
     * 
     * @param a First operand as two 32-bit integers
     * @param b Second operand as two 32-bit integers
     * @return The difference as two 32-bit integers
     */
     _CCCL_TRIVIAL_API __uint32x2 __isub_u64 (__uint32x2 __a, __uint32x2 __b) noexcept
     {
         uint64_t __a64 = __fpemu_bit_cast<uint64_t>(__a);
         uint64_t __b64 = __fpemu_bit_cast<uint64_t>(__b);
         uint64_t __res64 = __a64 - __b64;
         return __fpemu_bit_cast<__uint32x2>(__res64);
     } //__isub_u64

    /**
     * @brief Round a 64-bit value to a specified number of bits
     * 
     * This function rounds a 64-bit value represented as two 32-bit integers
     * to a specified number of bits according to the specified rounding mode.
     * 
     * @tparam rm Rounding mode (default: nearest-even)
     * @param man The 64-bit value to round as two 32-bit integers
     * @param shift The number of bits to round to
     * @param sign Result sign (used for directed rounding modes ru/rd)
     * @return The rounded value as two 32-bit integers
     */
    template<__fpemu_rounding _Rm = __fpemu_rounding::rn>
    _CCCL_TRIVIAL_API __uint32x2 __round (__uint32x2 __man, const int __shift, bool __sign = false) noexcept
    {
        uint64_t __man64 = __fpemu_bit_cast<uint64_t>(__man);
        const int __rshift = __fpemu_extra_bits + __shift;
        const uint64_t __round_bit = 1ULL << (__rshift - 1);
        const uint64_t __sticky_mask = (__round_bit << 1) - 1;
        const uint64_t __tie_mask    = (__round_bit << 2) - 1;

        if constexpr (_Rm == __fpemu_rounding::rz)
        {
            __man64 >>= __rshift;
        }
        else if constexpr (_Rm == __fpemu_rounding::rn)
        {
            const int __not_sticky = (int)((__man64 & __tie_mask) == __round_bit);
            __man64 = (((__man64 + __round_bit) >> __rshift) - __not_sticky);
        }
        else
        {
            const bool __inexact = (__man64 & __sticky_mask) != 0;
            __man64 >>= __rshift;
            if constexpr (_Rm == __fpemu_rounding::ru)
            {
                if (!__sign && __inexact) __man64++;
            }
            else // rd
            {
                if (__sign && __inexact) __man64++;
            }
        }
        return __fpemu_bit_cast<__uint32x2>(__man64);
    } //__round

    // NOTE: the representation pack/unpack routines
    //   __internal_fp64emu_unpack / __internal_fp64emu_pack
    // live in fpemu_impl_unpack.h so that the common prologue/epilogue shared
    // by every unpacked operation and method lives in a single header.



// ============================================================================
// Unpacked operations (pack / unpack)
// ============================================================================



#ifndef _CCCL_FP64EMU_UNPACKED_OUTPUT_INF
    #define _CCCL_FP64EMU_UNPACKED_OUTPUT_INF 0
#endif

#ifndef EXTRA_BITS
    #define EXTRA_BITS 9
#endif

// ============================================================================
// Shared fixed-point helpers for the native fp64 divide and square root
// implementations (fpemu_impl_div.h / fpemu_impl_sqrt.h).
// ============================================================================

/// @brief High 64 bits of a 64x64 -> 128 unsigned multiply (host/device).
_CCCL_TRIVIAL_API uint64_t __internal_fp64emu_mulhi64 (uint64_t __a, 
                                                       uint64_t __b) noexcept
{
#if defined(__CUDA_ARCH__)
    return __umul64hi(__a, __b);
#elif defined(__SIZEOF_INT128__)
    return (uint64_t)(((unsigned __int128)__a * (unsigned __int128)__b) >> 64);
#else
    uint64_t al = (uint32_t)a, ah = a >> 32;
    uint64_t bl = (uint32_t)b, bh = b >> 32;
    uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    uint64_t mid = (ll >> 32) + (uint32_t)lh + (uint32_t)hl;
    return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
#endif
} // __internal_fp64emu_mulhi64

/// @brief Right shift keeping a sticky bit.
_CCCL_TRIVIAL_API uint64_t __internal_fp64emu_shr_jam64 (uint64_t __a, 
                                                                  uint32_t __dist) noexcept
{
    return (__dist < 63) ? (__a >> __dist) | ((uint64_t)((__a << (-__dist & 63)) != 0)) : (uint64_t)(__a != 0);
} // __internal_fp64emu_shr_jam64

/// @brief Round and pack a result whose 'sig' carries its leading significand
///        bit at bit 62 and whose 'exp' is the biased exponent minus one.
///        Shared by divide and square root (square root passes sign = false).
template<__fpemu_rounding _Rm>
_CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_round_pack (bool     __sign, 
                                                          int32_t  __exp, 
                                                          uint64_t __sig) noexcept
{
    constexpr bool __round_near_even = (_Rm == __fpemu_rounding::rn);
    uint32_t __round_increment = 0x200;
    if      constexpr (_Rm == __fpemu_rounding::rz) __round_increment = 0;
    else if constexpr (_Rm == __fpemu_rounding::ru) __round_increment = __sign ? 0     : 0x3FF;
    else if constexpr (_Rm == __fpemu_rounding::rd) __round_increment = __sign ? 0x3FF : 0;

    uint32_t __round_bits = (uint32_t)(__sig & 0x3FF);

    if ((uint16_t)__exp >= 0x7FD)
    {
        if (__exp < 0)
        {
            __sig        = __internal_fp64emu_shr_jam64(__sig, (uint32_t)(-__exp));
            __exp        = 0;
            __round_bits = (uint32_t)(__sig & 0x3FF);
        }
        else if ((__exp > 0x7FD) || (__sig + __round_increment >= _CCCL_FPEMU_SIGN_64))
        {
            uint64_t __ui64_z = (((uint64_t)__sign << 63) + _CCCL_FPEMU_INF_64) - (uint64_t)(__round_increment == 0 ? 1 : 0);
            return (__fpbits64)__ui64_z;
        }
    }

    __sig = (__sig + __round_increment) >> 10;
    __sig &= ~(uint64_t)((__round_bits == 0x200) && __round_near_even);
    if (!__sig) __exp = 0;

    uint64_t __ui64_z = ((uint64_t)__sign << 63) + ((uint64_t)(uint32_t)__exp << 52) + __sig;
    return (__fpbits64)__ui64_z;
} // __internal_fp64emu_round_pack

// NOTE: the __fpbits64_unpacked pack/unpack routines
//   __internal_fp64emu_unpack / __internal_fp64emu_pack
// were moved to fpemu_impl_unpack.h (shared prologue/epilogue for every op).



} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPEMU_IMPL_UTILS_H

