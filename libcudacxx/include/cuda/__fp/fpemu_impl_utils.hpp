//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_UTILS_H
#define _CUDA___FP_FPEMU_IMPL_UTILS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/**
 * @file fpemu_impl_utils.hpp
 * @brief Common utility functions and definitions for FPEMU library
 *
 * This header provides common utility functions and definitions used throughout
 * the FPEMU library. It defines:
 *
 * - Compile-time assertion macros for type checking
 * - Bit casting utilities for type punning
 * - Platform-independent memory operations
 *
 * The utilities are designed to work across both host and device code through
 * appropriate decorators and provide consistent behavior across different
 * platforms and compilers.
 */
#if !defined(__CUDA_LIBDEVICE__)
    #include <cstdint>
    #include <cstring>
    #include <stdlib.h>
    #include <memory.h>
#endif

#include <cuda/std/__bit/bit_cast.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

// Verify that either __FPEMU_INLINE__ or __FPEMU_BUILD_LIB__/__FPEMU_USE_LIB__ is defined
#if !defined(__FPEMU_INLINE__) && !defined(__FPEMU_BUILD_LIB__) && !defined(__FPEMU_USE_LIB__)
    #error "ERROR: either __FPEMU_INLINE__ or __FPEMU_BUILD_LIB__/__FPEMU_USE_LIB__ must be defined"
#endif

/*
 * Packed-via-unpacked TEST mode (__FPEMU_PACKED_VIA_UNPACKED__) is configured in
 * fpemu_common.hpp and set by the Makefile's PACKED_VIA_UNPACKED=y. When ON, the
 * packed (fpbits64_t) builtins are routed through the combined unpack ->
 * *_unpacked core -> pack pipeline so the packed test harness exercises the
 * unpacked cores. When OFF (default), the legacy fused packed kernels are used
 * unchanged (byte-for-byte). The unpacked fpbits64_unpacked_t ABI builtins
 * (__FPEMU_UNPACKED__) always co-exist with the packed API regardless of this flag.
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
#ifndef __FPEMU_ADD_METHOD__
    #define __FPEMU_ADD_METHOD__   unset
#endif
#ifndef __FPEMU_MUL_METHOD__
    #define __FPEMU_MUL_METHOD__   unset
#endif
#ifndef __FPEMU_FMA_METHOD__
    #define __FPEMU_FMA_METHOD__   unset
#endif
#ifndef __FPEMU_DIV_METHOD__
    #define __FPEMU_DIV_METHOD__   unset
#endif
#ifndef __FPEMU_SQRT_METHOD__
    #define __FPEMU_SQRT_METHOD__  unset
#endif

namespace fpemu
{
    #define FP32_TOTAL_BITS  32
    #define FP32_BIAS        127
    #define FP32_MANT_BITS   23
    #define FP32_EXP_BITS    8
    #define FP32_SIGN_BITS   1
    #define FP32_MANT_MASK   0x007fffff
    #define FP32_LO_EXP_MASK 0x000000FF

    #define FP64_TOTAL_BITS   64
    #define FP64_BIAS         1023
    #define FP64_MANT_BITS    52
    #define FP64_EXP_BITS     11
    #define FP64_SIGN_BITS    1
    #define FP64_HI_MANT_MASK 0x000FFFFF    
    #define FP64_HI_SIGN_MASK 0x80000000 
    #define FP64_LO_EXP_MASK  0x000007FF

    // ---- Full-width IEEE-754 binary64 bit-field masks and canonical values ----
    // Plain ULL literals (64-bit on every supported platform) avoid relying on
    // the UINT64_C() macro, which is not always available in device builds.
    #define __FPEMU_SIGN_64__          0x8000000000000000ULL   // sign bit
    #define __FPEMU_EXP_64__           0x7FF0000000000000ULL   // exponent field (also +infinity)
    #define __FPEMU_MANT_64__          0x000FFFFFFFFFFFFFULL   // 52-bit trailing significand
    #define __FPEMU_ABS_64__           0x7FFFFFFFFFFFFFFFULL   // magnitude (everything but sign)
    #define __FPEMU_HIDDEN_64__        0x0010000000000000ULL   // implicit leading 1 (2^52)
    #define __FPEMU_QNAN_BIT_64__      0x0008000000000000ULL   // quiet bit (significand MSB)
    #define __FPEMU_SNAN_PAYLOAD_64__  0x0007FFFFFFFFFFFFULL   // signaling-NaN payload (low 51 bits)
    #define __FPEMU_INF_64__           0x7FF0000000000000ULL   // +infinity
    #define __FPEMU_QNAN_64__          0x7FF8000000000000ULL   // exponent field + quiet bit (+qNaN)
    #define __FPEMU_DEFNAN_64__        0xFFF8000000000000ULL   // canonical default NaN

    // 1.0f in fp32
    #define FP32_ONE          0x3f800000
    // 2.0f in fp32
    #define FP32_TWO          0x40000000

    // Number of extra bits for precise multiplication of the mantissas
    #define FP64_EXTRA_BITS 9
    
    // The value to shift high part of fp64 mantissa to get the exponent
    #define FP64_HI_MANT_SHIFT         (FP32_TOTAL_BITS - FP64_EXP_BITS - FP64_SIGN_BITS)          // 20
    // The value to shift mantissas product to get the mantissa of the result (to the begin of the mantissa)
    // *2 because the mantissa is 52 bits and the result is 104 bits
    #define FP64_MANT_MUL_SHIFT        (FP64_MANT_BITS - ((FP64_MANT_BITS*2) - FP64_TOTAL_BITS))   // 12
    // The value to shift the mantissa of the result to the high part of the fp32 mantissa
    // to use fp32 computations for the fp64 mantissa
    #define FP64_MANT_TO_FP32_HI_SHIFT (FP64_EXP_BITS - FP32_EXP_BITS)                             // 3
    // The value to shift the mantissa of the result to the low part of the fp32 mantissa
    // to use fp32 computations for the fp64 mantissa
    #define FP64_MANT_TO_FP32_LO_SHIFT (FP32_TOTAL_BITS - FP64_MANT_TO_FP32_HI_SHIFT)              // 29
    // Position of the carry bit in the mantissa of the result
    // to be added to the exponent of the result
    #define FP64_MANT_MUL_CARRY_BIT    (((FP64_MANT_BITS*2) - FP64_TOTAL_BITS) - FP32_TOTAL_BITS)  // 8

    /* Total lengh of the internal representation of the mantissa  */
    //constexpr uint64_t bitwidth = (MANTISSA_WIDTH + EXTRA_BITS);
    /* Exponent bias is 2^(11-1) -1 */    
    /* fp64 mantissa is 52 */
    constexpr uint64_t MANTISSA_WIDTH = 52;
    constexpr uint64_t EXPONENT_MASK  = __FPEMU_EXP_64__;
    constexpr uint64_t MANTISSA_MASK  = __FPEMU_MANT_64__;
    constexpr uint32_t EXTRA_BITS     = 9;
    constexpr uint32_t BIAS           = 1023;
    constexpr uint32_t INF_ZERO       = 0x00007ff0 - BIAS - 2048 - 1 + 0xC;; // - 128

    /*
    // by default route fpemu's internal bit-casts through
    // cuda::std::bit_cast. __FPEMU_BIT_CAST__ is the single switch point --
    // define it before including the fpemu headers for a fast re-map back to the
    // in-house polyfill, e.g.:
    //   #define __FPEMU_BIT_CAST__(To, v) \
    //       ::cuda::experimental::fpemu::__fpemu_builtin_bit_cast<To>(v)
    */
    #ifndef __FPEMU_BIT_CAST__
    #  define __FPEMU_BIT_CAST__(To, v) ::cuda::std::bit_cast<To>(v)
    #endif

    // In-house bit cast polyfill, kept available as the __FPEMU_BIT_CAST__
    // fallback target. Similar to C++20 std::bit_cast.
    template<typename T, typename R>
    __FPEMU_HOST_DEVICE_DECL__ T __fpemu_builtin_bit_cast(const R value) 
    {
        T dst;
    #if defined __DO_NOT_USE_MEMCPY__
        for (unsigned i = 0U; i < sizeof(T); i++)
        {
            unsigned char * ptrSRC = i + (unsigned char *)(&value);
            unsigned char * ptrDST = i + (unsigned char *)(&dst);
            *ptrDST = *ptrSRC;
        }
    #else
        #if !defined(__CUDA_LIBDEVICE__)
            std::memcpy(static_cast<void*>(&dst), static_cast<const void*>(&value),
                        sizeof(T));
        #else
            memcpy(static_cast<void*>(&dst), static_cast<const void*>(&value),
                        sizeof(T));
        #endif
    #endif
            return dst;
    }

    // Internal bit cast utility used throughout the library. Delegates to the
    // __FPEMU_BIT_CAST__ switch macro (cuda::std::bit_cast by default).
    template<typename T, typename R>
    __FPEMU_HOST_DEVICE_DECL__ T bit_cast(const R value)
    {
        return __FPEMU_BIT_CAST__(T, value);
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
    enum struct fpclass_t
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
     * @note `fpclass` carries a default member initializer (`fpclass_t::normal`)
     *       so that any code path producing an `fp64emu_unpacked` that does not
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
    struct fp64emu_unpacked 
    {
        uint64_t mantissa;
        int32_t exponent;
        uint32_t sign;
        fpclass_t fpclass = fpclass_t::normal;
    };

    /**
     * @brief Structure representing a 64-bit integer split into two 32-bit parts
     * 
     * This structure is used to store a 64-bit integer as two 32-bit parts,
     * which is useful for certain operations that require 64-bit arithmetic
     * but can be performed by 32-bit operations.    
     */
    struct uint32x2_t 
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
    struct uint64x2_t 
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
    struct uint32x4_t
    {
        uint32x2_t lo;
        uint32x2_t hi;
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
        __FPEMU_INTERNAL_DECL__ int __internal_clz(int x) 
        {
            if (x == 0) return 32;

            int n = 0;
            unsigned int u = (unsigned int)x;

            if ((u >> 16) == 0) { n += 16; u <<= 16; }
            if ((u >> 24) == 0) { n += 8;  u <<= 8;  }
            if ((u >> 28) == 0) { n += 4;  u <<= 4;  }
            if ((u >> 30) == 0) { n += 2;  u <<= 2;  }
            if ((u >> 31) == 0) { n += 1; }

            return n;
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
        __FPEMU_INTERNAL_DECL__ int __internal_clzll(int64_t x) 
        {
            uint64_t ux = (uint64_t)x;
            if (ux == 0)
                return 64;

            int count = 0;
            for (int i = 63; i >= 0; --i) {
                if (ux & (1ULL << i))
                    break;
                count++;
            }
            return count;
        }
    #else
        __FPEMU_INTERNAL_DECL__ int __internal_clz(int x)       { return __clz(x); }
        __FPEMU_INTERNAL_DECL__ int __internal_clzll(int64_t x) { return __clzll(x); }
    #endif //__CUDA_ARCH__

        #undef  __max_fp64emu
    #if defined(__CUDA_ARCH__) && !defined(__CUDA_LIBDEVICE__)
        // Global-scope qualifier: inside namespace cuda::experimental an
        // unqualified `max` now resolves to the fpmp2_t max() template, which
        // shadows the CUDA device `::max(int, int)` builtin we want here.
        #define __max_fp64emu      ::max
    #else
        #define __max_fp64emu(a, b) ((a) > (b) ? (a) : (b))
    #endif

    #ifndef __FP64EMU_PTX_XOR__
      #define __FP64EMU_PTX_XOR__ 0
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
    __FPEMU_INTERNAL_DECL__ uint32_t __invert_msb(uint32_t sign)
    {
    #if __FP64EMU_PTX_XOR__ == 1
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
        return sign ^ 0x80000000u;
    #endif
    }


    /**
     * @brief Multiply two 64-bit integers and return the high 32 bits of the result
     * 
     * This function performs multiplication of two 64-bit integers represented as pairs
     * of 32-bit integers (uint32x2_t). It returns only the high 32 bits of the 128-bit
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
     * @tparam meth The accuracy level (fp64emu_accuracy)
     * @param a_ First 64-bit multiplicand as two 32-bit integers
     * @param b_ Second 64-bit multiplicand as two 32-bit integers
     * @return The high 32 bits of the multiplication result
     */
    template<fp64emu_accuracy meth = fp64emu_accuracy::high>
    __FPEMU_INTERNAL_DECL__ uint32_t __mul_32(uint32x2_t a_, uint32x2_t b_)
    {
        uint32_t res;
#if defined  __CUDA_ARCH__
        uint32x2_t a64 = fpemu::bit_cast<uint32x2_t>(a_);
        uint32x2_t b64 = fpemu::bit_cast<uint32x2_t>(b_);
        uint32_t res32;
        asm ("{\n\t"
            ".reg .u32 r0, ahi, bhi;\n\t"
            "mov.b32         ahi, %1;   \n\t"
            "mov.b32         bhi, %2;   \n\t"
            "mad.hi.u32      r0, ahi, bhi,  0;\n\t"
            "mov.b32         %0, r0;     \n\t"
            "}"
            : "=r"(res32)
            : "r"(a64.x[1]), "r"(b64.x[1]));
        res = res32;
#else
        // Split inputs into 32-bit parts
        uint32_t ahi = a_.x[1];
        uint32_t bhi = b_.x[1];

        // Compute high 32 bits directly
        uint64_t hi_hi = uint64_t(ahi) * bhi;  // (a.hi * b.hi)
        res = uint32_t(hi_hi >> 32);
#endif
        return res;
    } //__mul_32


    /**
     * @brief Multiply two 64-bit integers and return the high 64 bits of the result
     * 
     * This function performs multiplication of two 64-bit integers represented as pairs
     * of 32-bit integers (uint32x2_t). It returns the high 64 bits of the 128-bit result.
     * 
     * The implementation is optimized differently for CUDA and CPU:
     * - On CUDA: Uses PTX assembly instructions for efficient 64-bit multiplication
     * - On CPU: Uses standard C++ arithmetic with careful handling of carries
     * 
     * @tparam meth The accuracy level (fp64emu_accuracy)
     * @param a_ First 64-bit multiplicand as two 32-bit integers
     * @param b_ Second 64-bit multiplicand as two 32-bit integers
     * @return The high 64 bits of the multiplication result as two 32-bit integers
     */
    template<fp64emu_accuracy meth = fp64emu_accuracy::high>
    __FPEMU_INTERNAL_DECL__ uint32x2_t __mul_64(uint32x2_t a_, uint32x2_t b_)
    {
        uint32x2_t res;
#if defined  __CUDA_ARCH__
        uint64_t a64 = fpemu::bit_cast<uint64_t>(a_);
        uint64_t b64 = fpemu::bit_cast<uint64_t>(b_);
        uint64_t res64;
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
            : "=l"(res64)
            : "l"(a64), "l"(b64));
        res = fpemu::bit_cast<uint32x2_t>(res64);
#else
        // Split inputs into 32-bit parts
        uint32_t alo = a_.x[0], ahi = a_.x[1];
        uint32_t blo = b_.x[0], bhi = b_.x[1];

        // Compute partial products that contribute to high 64 bits
        uint64_t lo_hi = uint64_t(alo) * bhi;  // (a.lo * b.hi)
        uint64_t hi_lo = uint64_t(ahi) * blo;  // (a.hi * b.lo)
        uint64_t hi_hi = uint64_t(ahi) * bhi;  // (a.hi * b.hi)

        // Extract high parts and combine
        uint32_t lo_hi_hi = uint32_t(lo_hi >> 32);
        uint32_t hi_lo_hi = uint32_t(hi_lo >> 32);
        uint32_t hi_hi_lo = uint32_t(hi_hi);
        uint32_t hi_hi_hi = uint32_t(hi_hi >> 32);

        // Combine high parts with carries
        uint64_t hi = uint64_t(lo_hi_hi) + hi_lo_hi + hi_hi_lo;
        
        // Store high result
        res.x[0] = uint32_t(hi);
        res.x[1] = uint32_t(hi >> 32) + hi_hi_hi;
#endif
        return res;
    } //__mul_64


    /**
     * @brief Multiply two 64-bit integers and return the full 128-bit result
     * 
     * This function performs multiplication of two 64-bit integers represented as pairs
     * of 32-bit integers (uint32x2_t). It returns the full 128-bit result stored in
     * a uint32x4_t structure containing both high and low 64 bits.
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
     * @tparam meth The accuracy level (fp64emu_accuracy)
     * @param a_ First 64-bit multiplicand as two 32-bit integers
     * @param b_ Second 64-bit multiplicand as two 32-bit integers
     * @return The full 128-bit multiplication result as four 32-bit integers
     */
    template<fp64emu_accuracy meth = fp64emu_accuracy::high>
    __FPEMU_INTERNAL_DECL__ uint32x4_t __mul_128(uint32x2_t a_, uint32x2_t b_)
    {
        uint32x4_t res;

        // Split inputs into 32-bit parts
        uint32_t alo = a_.x[0], ahi = a_.x[1];
        uint32_t blo = b_.x[0], bhi = b_.x[1];

        // Compute partial products
        uint64_t lo_lo = uint64_t(alo) * blo;  // (a.lo * b.lo)
        uint64_t lo_hi = uint64_t(alo) * bhi;  // (a.lo * b.hi) 
        uint64_t hi_lo = uint64_t(ahi) * blo;  // (a.hi * b.lo)

        // Extract 32-bit parts of intermediate results
        uint32_t lo_lo_lo = uint32_t(lo_lo);
        uint32_t lo_lo_hi = uint32_t(lo_lo >> 32);
        uint32_t lo_hi_lo = uint32_t(lo_hi);
        uint32_t hi_lo_lo = uint32_t(hi_lo);

        // Compute final low result with carries
        uint64_t mid = uint64_t(lo_lo_hi) + lo_hi_lo + hi_lo_lo;
        uint32_t mid_lo = uint32_t(mid);
        
        // Store low result
        res.lo.x[0] = lo_lo_lo;
        res.lo.x[1] = mid_lo;

#if defined  __CUDA_ARCH__
        uint64_t a64 = fpemu::bit_cast<uint64_t>(a_);
        uint64_t b64 = fpemu::bit_cast<uint64_t>(b_);
        uint64_t res64 = __umul64hi(a64, b64);
        res.hi = fpemu::bit_cast<uint32x2_t>(res64);
#else
        uint64_t hi_hi = uint64_t(ahi) * bhi;  // (a.hi * b.hi)
        uint32_t lo_hi_hi = uint32_t(lo_hi >> 32);
        uint32_t hi_lo_hi = uint32_t(hi_lo >> 32);
        uint32_t hi_hi_lo = uint32_t(hi_hi);
        uint32_t mid_hi = uint32_t(mid >> 32);
        uint64_t hi = uint64_t(mid_hi) + lo_hi_hi + hi_lo_hi + hi_hi_lo;
        uint32_t hi_hi_hi = uint32_t(hi_hi >> 32);

        // Store high result
        res.hi.x[0] = uint32_t(hi);
        res.hi.x[1] = uint32_t(hi >> 32) + hi_hi_hi;
#endif
        return res;
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
     __FPEMU_INTERNAL_DECL__ uint32x2_t __shl_64 (uint32x2_t man, int shift)
     {
         uint64_t man64 = fpemu::bit_cast<uint64_t>(man);
         man64 <<= shift;
         return fpemu::bit_cast<uint32x2_t>(man64);
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
     __FPEMU_INTERNAL_DECL__ uint32x2_t __shr_64 (uint32x2_t man, int shift)
     {
         uint64_t man64 = fpemu::bit_cast<uint64_t>(man);
 #ifndef __CUDA_ARCH__
         shift = (shift > 0) ? (shift>64) ? 64 : shift : 0;
 #endif
         man64 = man64 >> shift;
         return fpemu::bit_cast<uint32x2_t>(man64);
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
    template<fpemu::rounding rm = fpemu::rounding::rn>
    __FPEMU_INTERNAL_DECL__ float __fmul_dir (float x, float y)
    {
#if defined(__CUDA_ARCH__)
        if constexpr (rm == fpemu::rounding::rn) return __fmul_rn(x, y);
        else if constexpr (rm == fpemu::rounding::rz) return __fmul_rz(x, y);
        else if constexpr (rm == fpemu::rounding::ru) return __fmul_ru(x, y);
        else return __fmul_rd(x, y);
#else
        (void)rm;
        return x * y;
#endif
    } //__fmul_dir

    /**
     * @brief Single-precision add with directed rounding (fast add paths)
     *
     * Uses CUDA intrinsics on device; host falls back to native add (RN).
     */
    template<fpemu::rounding rm = fpemu::rounding::rn>
    __FPEMU_INTERNAL_DECL__ float __fadd_dir (float x, float y)
    {
#if defined(__CUDA_ARCH__)
        if constexpr (rm == fpemu::rounding::rn) return __fadd_rn(x, y);
        else if constexpr (rm == fpemu::rounding::rz) return __fadd_rz(x, y);
        else if constexpr (rm == fpemu::rounding::ru) return __fadd_ru(x, y);
        else return __fadd_rd(x, y);
#else
        (void)rm;
        return x + y;
#endif
    } //__fadd_dir

     template<fpemu::rounding rm = fpemu::rounding::rn>
     __FPEMU_INTERNAL_DECL__ uint32x2_t __shr_64_rnd (uint32x2_t man, int shift, bool sign = false)
     {
         uint64_t man64 = fpemu::bit_cast<uint64_t>(man);
 #ifndef __CUDA_ARCH__
         shift = (shift > 0) ? (shift>64) ? 64 : shift : 0;
 #endif
         if (shift <= 0) return man;

         const uint64_t discard_mask = (shift >= 64) ? ~0ULL : ((1ULL << shift) - 1);
         const bool inexact = (man64 & discard_mask) != 0;
         man64 >>= shift;

         if constexpr (rm == fpemu::rounding::ru)
         {
             if (!sign && inexact) man64++;
         }
         else if constexpr (rm == fpemu::rounding::rd)
         {
             if (sign && inexact) man64++;
         }
         return fpemu::bit_cast<uint32x2_t>(man64);
     } //__shr_64_rnd

    /**
     * @brief Logical right shift of 128-bit mantissa with directed rounding
     */
    template<fpemu::rounding rm = fpemu::rounding::rn>
    __FPEMU_INTERNAL_DECL__ __uint128_t __shr_128_rnd (__uint128_t man, int shift, bool sign = false)
    {
#ifndef __CUDA_ARCH__
        shift = (shift > 0) ? ((shift > 127) ? 127 : shift) : 0;
#endif
        if (shift <= 0) return man;

        const __uint128_t discard_mask =
            (shift >= 128) ? ~(__uint128_t)0 : ((((__uint128_t)1) << shift) - 1);
        const bool inexact = (man & discard_mask) != 0;
        man >>= shift;

        if constexpr (rm == fpemu::rounding::rn || rm == fpemu::rounding::rz)
        {
            if (inexact) man |= 1;
        }
        else if constexpr (rm == fpemu::rounding::ru)
        {
            if (!sign && inexact) man |= 1;
        }
        else if constexpr (rm == fpemu::rounding::rd)
        {
            if (sign && inexact) man |= 1;
        }
        return man;
    } //__shr_128_rnd

    /**
     * @brief Logical right shift of 128-bit mantissa with jam (sticky) only.
     * Used during FMA alignment; directed rounding is deferred to the pack epilogue.
     */
    __FPEMU_INTERNAL_DECL__ __uint128_t __shr_128_jam (__uint128_t man, int shift)
    {
        return __shr_128_rnd<fpemu::rounding::rn>(man, shift);
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
      template<fp64emu_accuracy meth = fp64emu_accuracy::high>
     __FPEMU_INTERNAL_DECL__ uint32x2_t __sar_64 (uint32x2_t man, int shift)
     {
 #ifndef __CUDA_ARCH__
         shift = (shift > 0) ? (shift>63) ? 63 : shift : 0;
 #endif
         int64_t man64 = fpemu::bit_cast<int64_t>(man);
        man64 = man64 >> shift;
         uint32x2_t res = fpemu::bit_cast<uint32x2_t>(man64);
         return res;
     } //__sar_64_rnd



     /**
      * @brief Arithmetic Shift a 64-bit value right with rounding
      * 
      * This function performs a arithmetic right shift on a 64-bit value represented
      * as two 32-bit integers, with rounding according to the specified mode.
      * The shift amount can be positive or negative.
      * 
      * @tparam meth Accuracy level (sticky-bit preservation applies only for high)
      * @tparam rm  Rounding mode (default: nearest-even)
      * @param man  The 64-bit value to shift as two 32-bit integers
      * @param shift The number of bits to shift (positive for right shift)
      * @param sign Result sign (used for directed rounding modes ru/rd)
      * @return The shifted and rounded value as two 32-bit integers
      */
      template<fp64emu_accuracy meth = fp64emu_accuracy::high,
               fpemu::rounding rm = fpemu::rounding::rn>
     __FPEMU_INTERNAL_DECL__ uint32x2_t __sar_64_rnd (uint32x2_t man, int shift, bool sign = false)
     {
 #ifndef __CUDA_ARCH__
         shift = (shift > 0) ? (shift>63) ? 63 : shift : 0;
 #endif
         int64_t man64 = fpemu::bit_cast<int64_t>(man);
         int64_t man64_res = man64 >> shift;
         uint32x2_t res = fpemu::bit_cast<uint32x2_t>(man64_res);
         if constexpr (meth == fp64emu_accuracy::high)
         {
            uint64_t mask = (1LLU << shift) - 1;
            bool sticky = (man64 & mask) != 0;
            if constexpr (rm == fpemu::rounding::rn)
            {
                res.x[0] |= sticky;
            }
            else if constexpr (rm == fpemu::rounding::rz)
            {
                // truncate toward zero
            }
            else if constexpr (rm == fpemu::rounding::ru)
            {
                if (!sign && sticky) res.x[0] |= 1;
            }
            else if constexpr (rm == fpemu::rounding::rd)
            {
                if (sign && sticky) res.x[0] |= 1;
            }
         }
         return res;
     } //__sar_64_rnd
 

    /**
     * @brief Unpack the exponent from a 64-bit floating point number
     * 
     * This function extracts the exponent from a 64-bit floating point number
     * represented as two 32-bit integers. It handles the exponent extraction
     * according to IEEE-754 double precision format.
     * 
     * @tparam meth Accuracy level (full-range special handling only for high)
     * @param input The input number as two 32-bit integers
     * @return The extracted exponent value
     */
     template<fp64emu_accuracy meth = fp64emu_accuracy::high>
    __FPEMU_INTERNAL_DECL__ int32_t __unpack_exp(uint32x2_t input)
    {
        int32_t exp;

        //remove sign bit
        input.x[1] &= 0x7fffffff;

        //Shift exponent bits to the beggining
        exp = input.x[1] >> 20;

        //Extract mantissa bits
        input.x[1] = input.x[1] & 0x000fffff;

        //Check for NAN and INF
        if constexpr (meth == fp64emu_accuracy::high) 
        {
            if (exp == 0x7ff) 
            {
                if (input.x[0] == 0 && input.x[1] == 0) 
                {
                    exp = 0x000017ff; //Magic value for INF
                } 
                else 
                {
                    exp = 0x000047ff; //Magic value for NAN
                }
            }
        }

        return exp;
    } //__unpack_exp

    /**
     * @brief Unpack the mantissa from a 64-bit floating point number
     * 
     * This function extracts the mantissa from a 64-bit floating point number
     * represented as two 32-bit integers. It handles both normal and denormal
     * numbers according to IEEE-754 double precision format.
     * 
     * @tparam twos_comp_flag Whether to use two's complement representation
     * @tparam meth Accuracy level (full-range special handling only for high)
     * @param sign Pointer to store the sign bit
     * @param input The input number as two 32-bit integers
     * @param is_zero_exp Whether the exponent is zero (denormal case)
     * @return The extracted mantissa as two 32-bit integers
     */
    template<fp64emu_accuracy meth = fp64emu_accuracy::high>
    __FPEMU_INTERNAL_DECL__ uint32x2_t __unpack_mant(bool *sign, uint32x2_t input, bool is_zero_exp)
    {
        uint32x2_t man32x2;

        //Extract sign
        *sign = (input.x[1] & 0x80000000) != 0;

        //Extract mantissa bits
        man32x2.x[0] = input.x[0];
        man32x2.x[1] = input.x[1] & 0x000fffff;

        //Set implicit-bit for normal numbers
        if (!is_zero_exp) man32x2.x[1] |= ( 1 << 20 );

        man32x2 = __shl_64(man32x2, EXTRA_BITS);
        return man32x2;
    } //__unpack_mant

    /**
     * @brief Saturate unbiased exponent and mantissa on fp64 overflow
     *
     * Sets exp (unbiased exponent field, 0..2047) and man (without exponent
     * in man.x[1]) according to rounding mode. Caller folds exp into man via
     * man.x[1] |= (uint32_t)exp << FP64_HI_MANT_SHIFT when needed.
     *
     * @tparam rm   Rounding mode
     * @param sign  Result sign (true for negative)
     * @param exp   Output unbiased exponent field
     * @param man   Output mantissa as two 32-bit integers
     */
    template<fpemu::rounding rm = fpemu::rounding::rn>
    __FPEMU_INTERNAL_DECL__
    void __fp64_ovfl_sat (bool sign, int32_t& exp, uint32x2_t& man)
    {
        if constexpr (rm == fpemu::rounding::rz)
        {
            exp = FP64_BIAS * 2;
            man = {0xffffffff, FP64_HI_MANT_MASK};
        }
        else if constexpr (rm == fpemu::rounding::rn)
        {
            exp = FP64_BIAS * 2 + 1;
            man = {0, 0};
        }
        else if constexpr (rm == fpemu::rounding::ru)
        {
            if (sign)
            {
                exp = FP64_BIAS * 2;
                man = {0xffffffff, FP64_HI_MANT_MASK};
            }
            else
            {
                exp = FP64_BIAS * 2 + 1;
                man = {0, 0};
            }
        }
        else // rd
        {
            if (sign)
            {
                exp = FP64_BIAS * 2 + 1;
                man = {0, 0};
            }
            else
            {
                exp = FP64_BIAS * 2;
                man = {0xffffffff, FP64_HI_MANT_MASK};
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
     * @tparam meth Accuracy level (full-range special handling only for high)
     * @tparam rm  Rounding mode for overflow/underflow saturation (default: nearest-even)
     * @param sign The sign bit (true for negative)
     * @param exp The exponent value
     * @param man The mantissa as two 32-bit integers
     * @return The packed 64-bit double precision number
     */
    template<fp64emu_accuracy meth = fp64emu_accuracy::high,
             fpemu::rounding rm = fpemu::rounding::rn>
    __FPEMU_INTERNAL_DECL__ uint64_t __pack (bool sign, uint32_t exp, uint32x2_t man)
    {

        bool zero_mantissa = man.x[0] == 0 && man.x[1] == 0;
        bool is_nan = false;
        // Check for NAN
        if constexpr (meth == fp64emu_accuracy::high)
        {
            is_nan =
                exp >= (0x000047ff - BIAS - 52 - 1) ||     //NAN * x && NAN + x
                exp == (0x000017ff - BIAS - 52 - 1) ||     //INF * 0
                (exp == 0x000017ff - 64 && zero_mantissa); //INF + (-INF)

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
        bool is_inf = false;
        if constexpr (meth == fp64emu_accuracy::high)
        {
            is_inf = (exp >= (0x000017ff - BIAS - 52 - 1)) && !is_nan;
        }
        // Check for exact zero result
        if ( zero_mantissa && exp < 0x000007ff) exp = 0;
    
        if constexpr (meth == fp64emu_accuracy::high)
        {
            // Convert special value of NAN|INF back to IEEE
            if (exp >= 0x000007ff) exp = 0x000007ff;
        }


        // Shift exponent back to high bits
        exp <<= 20; 

        // Adds up exponent and mantissa to count implicit-bit
        man.x[1] += exp;

        // Check for the final overflow
        if (man.x[1] >= 0x7ff00000) 
        {
            if (is_nan)
            {
                man.x[0] = 0;
                man.x[1] = 0x7fffffff;
            }
            else if (is_inf)
            {
                // True infinity: emit inf for every rounding mode (do NOT saturate
                // to DBL_MAX as a finite overflow would under rz/ru/rd).
                man.x[0] = 0;
                man.x[1] = 0x7ff00000;
            }
            else
            {
                int32_t sat_exp = 0;
                __fp64_ovfl_sat<rm>(sign, sat_exp, man);
                man.x[1] |= (uint32_t)sat_exp << FP64_HI_MANT_SHIFT;
            }
        }


        // Pack everything to FP64
        uint32x2_t res = {man.x[0], man.x[1] | (sign << 31)};

        return fpemu::bit_cast<uint64_t>(res);
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
    __FPEMU_INTERNAL_DECL__ uint32x2_t __two_comp (uint32x2_t c)
    {
        uint64_t c64   = fpemu::bit_cast<uint64_t>(c);
        uint64_t res64 = 0 - c64; //IMAD.WIDE.U32 Rd, RZ, RZ, -Rc
        return fpemu::bit_cast<uint32x2_t>(res64);
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
    __FPEMU_INTERNAL_DECL__ int32_t __flo_s64 (uint32x2_t x)
    {
        int64_t x64 = fpemu::bit_cast<int64_t>(x);
        return __internal_clzll(x64<<1);
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
    __FPEMU_INTERNAL_DECL__ int32_t __flo_u64 (uint32x2_t x)
    {
        uint64_t x64 = fpemu::bit_cast<uint64_t>(x);
        //Skip sign bit
        return __internal_clzll(x64 & __FPEMU_ABS_64__);
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
    __FPEMU_INTERNAL_DECL__ uint32x2_t __iadd_u64 (uint32x2_t a, uint32x2_t b)
    {
        uint64_t a64 = fpemu::bit_cast<uint64_t>(a);
        uint64_t b64 = fpemu::bit_cast<uint64_t>(b);
        uint64_t res64 = a64 + b64;
        return fpemu::bit_cast<uint32x2_t>(res64);
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
     __FPEMU_INTERNAL_DECL__ uint32x2_t __isub_u64 (uint32x2_t a, uint32x2_t b)
     {
         uint64_t a64 = fpemu::bit_cast<uint64_t>(a);
         uint64_t b64 = fpemu::bit_cast<uint64_t>(b);
         uint64_t res64 = a64 - b64;
         return fpemu::bit_cast<uint32x2_t>(res64);
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
    template<fpemu::rounding rm = fpemu::rounding::rn>
    __FPEMU_INTERNAL_DECL__ uint32x2_t __round (uint32x2_t man, const int shift, bool sign = false)
    {
        uint64_t man64 = fpemu::bit_cast<uint64_t>(man);
        const int rshift = EXTRA_BITS + shift;
        const uint64_t round_bit = 1ULL << (rshift - 1);
        const uint64_t sticky_mask = (round_bit << 1) - 1;
        const uint64_t tie_mask    = (round_bit << 2) - 1;

        if constexpr (rm == fpemu::rounding::rz)
        {
            man64 >>= rshift;
        }
        else if constexpr (rm == fpemu::rounding::rn)
        {
            const int not_sticky = (int)((man64 & tie_mask) == round_bit);
            man64 = (((man64 + round_bit) >> rshift) - not_sticky);
        }
        else
        {
            const bool inexact = (man64 & sticky_mask) != 0;
            man64 >>= rshift;
            if constexpr (rm == fpemu::rounding::ru)
            {
                if (!sign && inexact) man64++;
            }
            else // rd
            {
                if (sign && inexact) man64++;
            }
        }
        return fpemu::bit_cast<uint32x2_t>(man64);
    } //__round

    // NOTE: the representation pack/unpack routines
    //   impl::__nv_internal_fp64emu_unpack / impl::__nv_internal_fp64emu_pack
    // live in fpemu_impl_unpack.hpp so that the common prologue/epilogue shared
    // by every unpacked operation and method lives in a single header.
} // namespace fpemu


// ============================================================================
// Unpacked operations (pack / unpack)
// ============================================================================

namespace impl
{

#ifndef __FP64EMU_UNPACKED_OUTPUT_INF__
    #define __FP64EMU_UNPACKED_OUTPUT_INF__ 0
#endif

#ifndef EXTRA_BITS
    #define EXTRA_BITS 9
#endif

// ============================================================================
// Shared fixed-point helpers for the native fp64 divide and square root
// implementations (fpemu_impl_div.hpp / fpemu_impl_sqrt.hpp).
// ============================================================================

/// @brief High 64 bits of a 64x64 -> 128 unsigned multiply (host/device).
__FPEMU_INTERNAL_DECL__ uint64_t __nv_internal_fp64emu_mulhi64 (uint64_t a, 
                                                                uint64_t b)
{
#if defined(__CUDA_ARCH__)
    return __umul64hi(a, b);
#elif defined(__SIZEOF_INT128__)
    return (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) >> 64);
#else
    uint64_t al = (uint32_t)a, ah = a >> 32;
    uint64_t bl = (uint32_t)b, bh = b >> 32;
    uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    uint64_t mid = (ll >> 32) + (uint32_t)lh + (uint32_t)hl;
    return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
#endif
} // __nv_internal_fp64emu_mulhi64

/// @brief Right shift keeping a sticky bit.
__FPEMU_INTERNAL_DECL__ uint64_t __nv_internal_fp64emu_shr_jam64 (uint64_t a, 
                                                                  uint32_t dist)
{
    return (dist < 63) ? (a >> dist) | ((uint64_t)((a << (-dist & 63)) != 0)) : (uint64_t)(a != 0);
} // __nv_internal_fp64emu_shr_jam64

/// @brief Round and pack a result whose 'sig' carries its leading significand
///        bit at bit 62 and whose 'exp' is the biased exponent minus one.
///        Shared by divide and square root (square root passes sign = false).
template<fpemu::rounding rm>
__FPEMU_INTERNAL_DECL__ fpbits64_t __nv_internal_fp64emu_round_pack (bool     sign, 
                                                                     int32_t  exp, 
                                                                     uint64_t sig)
{
    constexpr bool round_near_even = (rm == fpemu::rounding::rn);
    uint32_t round_increment = 0x200;
    if      constexpr (rm == fpemu::rounding::rz) round_increment = 0;
    else if constexpr (rm == fpemu::rounding::ru) round_increment = sign ? 0     : 0x3FF;
    else if constexpr (rm == fpemu::rounding::rd) round_increment = sign ? 0x3FF : 0;

    uint32_t round_bits = (uint32_t)(sig & 0x3FF);

    if ((uint16_t)exp >= 0x7FD)
    {
        if (exp < 0)
        {
            sig        = __nv_internal_fp64emu_shr_jam64(sig, (uint32_t)(-exp));
            exp        = 0;
            round_bits = (uint32_t)(sig & 0x3FF);
        }
        else if ((exp > 0x7FD) || (sig + round_increment >= __FPEMU_SIGN_64__))
        {
            uint64_t ui64_z = (((uint64_t)sign << 63) + __FPEMU_INF_64__) - (uint64_t)(round_increment == 0 ? 1 : 0);
            return (fpbits64_t)ui64_z;
        }
    }

    sig = (sig + round_increment) >> 10;
    sig &= ~(uint64_t)((round_bits == 0x200) && round_near_even);
    if (!sig) exp = 0;

    uint64_t ui64_z = ((uint64_t)sign << 63) + ((uint64_t)(uint32_t)exp << 52) + sig;
    return (fpbits64_t)ui64_z;
} // __nv_internal_fp64emu_round_pack

// NOTE: the fpbits64_unpacked_t pack/unpack routines
//   impl::__nv_internal_fp64emu_unpack / impl::__nv_internal_fp64emu_pack
// were moved to fpemu_impl_unpack.hpp (shared prologue/epilogue for every op).

} // namespace impl

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPEMU_IMPL_UTILS_H

