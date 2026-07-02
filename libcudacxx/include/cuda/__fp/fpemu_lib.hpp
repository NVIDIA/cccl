//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_LIB_H
#define _CUDA___FP_FPEMU_LIB_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/**
 * @file fpemu_lib.hpp
 * @brief Core function APIs for FPEMU floating point scalar emulation library
 *
 * This header provides the core function APIs for the FPEMU library for emulating
 * IEEE-754 double precision floating point scalar operations. It defines:
 *
 * - Arithmetic operation APIs (add, multiply, divide, etc)
 * - Comparison operation APIs (equals, less than, etc) 
 * - Conversion operation APIs (float/double conversions)
 * - Special function APIs (sqrt, fma, mad, dot, cmul, etc)
 *
 * Built-in naming convention (packed fpbits64_t):
 *   __nv_fp64emu_<op>_<rm>            — correctly rounded, full IEEE-754 range
 *   __nv_fp64emu_high_<op>_<rm>   — same as above, explicitly named
 *   __nv_fp64emu_mid_<op>_<rm>        — up to 1-2 least significant mantissa bits of error,
 *                                        limited INF, NaN and subnormal support
 *   __nv_fp64emu_low_<op>_<rm>       — up to half mantissa bits lost,
 *                                        limited INF, NaN and subnormal support
 *
 * Built-in naming convention (unpacked fpbits64_unpacked_t):
 *   __nv_fp64emu_unpacked_<op>_<rm>            — correctly rounded, full IEEE-754 range
 *   __nv_fp64emu_unpacked_high_<op>_<rm>   — same as above, explicitly named
 *   __nv_fp64emu_unpacked_mid_<op>_<rm>        — up to 1-2 least significant mantissa bits of error,
 *                                                  limited INF, NaN and subnormal support
 *   __nv_fp64emu_unpacked_low_<op>_<rm>       — up to half mantissa bits lost,
 *                                                  limited INF, NaN and subnormal support
 *
 * where <op> is the operation (dadd, dmul, dsub, ddiv, dsqrt, fma, mad, dot, cmul)
 * and <rm> is the rounding mode (rn, rz, ru, rd).
 *
 * The APIs are designed to work across both host and device code through
 * appropriate decorators and provide bit-exact results matching hardware
 * floating point units.
 */

#include <cuda/__fp/fpemu_common.hpp>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

/*
// Comparison operations
*/
/**
* @brief Comparison API APIs for 64-bit Floating-Point Operations
* 
* This section provides a set of comparison functions for 64-bit floating-point values (fpbits64_t)
* that follow IEEE-754 floating-point comparison rules. These functions are fundamental building blocks
* used throughout the library for implementing various mathematical operations and are particularly
* important for range checking, conditional operations, and validation of input parameters.
* 
* The comparison functions handle all special cases properly:
* - NaN comparisons (always return false for equality/inequality)
* - Infinities
* - Zero values (both +0 and -0)
* 
* Available comparison operations:
* - __nv_fp64emu_cmp_eq: Equality comparison (x == y)
* - __nv_fp64emu_cmp_ne: Inequality comparison (x != y)
* - __nv_fp64emu_cmp_le: Less than or equal comparison (x <= y)
* - __nv_fp64emu_cmp_lt: Less than comparison (x < y)
* - __nv_fp64emu_cmp_ge: Greater than or equal comparison (x >= y)
* - __nv_fp64emu_cmp_gt: Greater than comparison (x > y)
* 
*/
__FPEMU_BUILTIN_DECL__  bool     __nv_fp64emu_cmp_eq (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  bool     __nv_fp64emu_cmp_ne (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  bool     __nv_fp64emu_cmp_le (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  bool     __nv_fp64emu_cmp_lt (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  bool     __nv_fp64emu_cmp_ge (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  bool     __nv_fp64emu_cmp_gt (fpbits64_t x, fpbits64_t y);

/*
// Conversion operations
*/
/**
* @brief Conversion functions from 64-bit floating-point to double-precision floating-point
* 
* This group of functions converts a 64-bit floating-point value (fpbits64_t) to a double-precision
* floating-point value (double). The conversion follows IEEE-754 floating-point conversion rules.
* 
* Functions:
* - __nv_fp64emu_to_double: Convert from fpbits64_t to double
* - __nv_fp64emu_from_double: Convert from double to fpbits64_t
* - __nv_fp64emu_to_float: Convert from fpbits64_t to float
* - __nv_fp64emu_from_float: Convert from float to fpbits64_t
* - __nv_fp64emu_to_int: Convert from fpbits64_t to int32_t
* 
*/
__FPEMU_BUILTIN_DECL__ double     __nv_fp64emu_to_double          (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_double        (double x);
__FPEMU_BUILTIN_DECL__ float      __nv_fp64emu_to_float           (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_float         (float x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_int           (int32_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_uint          (uint32_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ll            (int64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ull           (uint64_t x);
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_fpbits64_cast_ull  (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_ull_cast_fpbits64  (uint64_t x);


/**
* @brief Conversion functions from 64-bit floating-point to 32-bit integer
* 
* This group of functions converts a 64-bit floating-point value (fpbits64_t) to a 32-bit integer (int32_t)
* by different rounding modes. The conversion follows IEEE-754 floating-point to integer conversion rules.
* 
* Functions:
* - __nv_fp64emu_to_int_rn: Convert with round-to-nearest rounding
* - __nv_fp64emu_to_int_rz: Convert with round-toward-zero rounding
* - __nv_fp64emu_to_int_ru: Convert with round-toward-positive-infinity rounding
* - __nv_fp64emu_to_int_rd: Convert with round-toward-negative-infinity rounding
* 
*/
__FPEMU_BUILTIN_DECL__    int32_t __nv_fp64emu_to_int_rn (fpbits64_t x);
__FPEMU_BUILTIN_DECL__    int32_t __nv_fp64emu_to_int_rz (fpbits64_t x);
__FPEMU_BUILTIN_DECL__    int32_t __nv_fp64emu_to_int_ru (fpbits64_t x);
__FPEMU_BUILTIN_DECL__    int32_t __nv_fp64emu_to_int_rd (fpbits64_t x);

/**
* @brief Conversion functions from 64-bit floating-point to unsigned 32-bit integer
* 
* This group of functions converts a 64-bit floating-point value (fpbits64_t) to an unsigned 32-bit integer (uint32_t)
* by different rounding modes. The conversion follows IEEE-754 floating-point to integer conversion rules.
* 
* Functions:
* - __nv_fp64emu_to_uint_rn: Convert with round-to-nearest rounding
* - __nv_fp64emu_to_uint_rz: Convert with round-toward-zero rounding
* - __nv_fp64emu_to_uint_ru: Convert with round-toward-positive-infinity rounding
* - __nv_fp64emu_to_uint_rd: Convert with round-toward-negative-infinity rounding
* 
*/
__FPEMU_BUILTIN_DECL__    uint32_t __nv_fp64emu_to_uint_rn (fpbits64_t x);
__FPEMU_BUILTIN_DECL__    uint32_t __nv_fp64emu_to_uint_rz (fpbits64_t x);
__FPEMU_BUILTIN_DECL__    uint32_t __nv_fp64emu_to_uint_ru (fpbits64_t x);
__FPEMU_BUILTIN_DECL__    uint32_t __nv_fp64emu_to_uint_rd (fpbits64_t x);

/**
* @brief Conversion functions from 64-bit floating-point to 64-bit integer
* 
* This group of functions converts a 64-bit floating-point value (fpbits64_t) to a 64-bit integer (int64_t)
* by different rounding modes. The conversion follows IEEE-754 floating-point to integer conversion rules.
* 
* Functions:
* - __nv_fp64emu_to_ll_rn: Convert with round-to-nearest rounding
* - __nv_fp64emu_to_ll_rz: Convert with round-toward-zero rounding
* - __nv_fp64emu_to_ll_ru: Convert with round-toward-positive-infinity rounding
* - __nv_fp64emu_to_ll_rd: Convert with round-toward-negative-infinity rounding
* 
*/
__FPEMU_BUILTIN_DECL__    int64_t __nv_fp64emu_to_ll_rn (fpbits64_t x);
__FPEMU_BUILTIN_DECL__    int64_t __nv_fp64emu_to_ll_rz (fpbits64_t x);
__FPEMU_BUILTIN_DECL__    int64_t __nv_fp64emu_to_ll_ru (fpbits64_t x);
__FPEMU_BUILTIN_DECL__    int64_t __nv_fp64emu_to_ll_rd (fpbits64_t x);

/**
* @brief Conversion functions from 64-bit floating-point to unsigned 64-bit integer
* 
* This group of functions converts a 64-bit floating-point value (fpbits64_t) to an unsigned 64-bit integer (uint64_t)
* by different rounding modes. The conversion follows IEEE-754 floating-point to integer conversion rules.
* 
* Functions:
* - __nv_fp64emu_to_ull_rn: Convert with round-to-nearest rounding
* - __nv_fp64emu_to_ull_rz: Convert with round-toward-zero rounding
* - __nv_fp64emu_to_ull_ru: Convert with round-toward-positive-infinity rounding
* - __nv_fp64emu_to_ull_rd: Convert with round-toward-negative-infinity rounding
* 
*/
__FPEMU_BUILTIN_DECL__    uint64_t __nv_fp64emu_to_ull_rn (fpbits64_t x);
__FPEMU_BUILTIN_DECL__    uint64_t __nv_fp64emu_to_ull_rz (fpbits64_t x);
__FPEMU_BUILTIN_DECL__    uint64_t __nv_fp64emu_to_ull_ru (fpbits64_t x);
__FPEMU_BUILTIN_DECL__    uint64_t __nv_fp64emu_to_ull_rd (fpbits64_t x);

/**
* @brief Double-precision floating-point addition kernels
*
* Collection of addition kernels with different accuracy, range, and rounding modes:
* - Accuracy: cr (correct rounding), ha (high accuracy), la (low accuracy)
* - Range: full (IEEE-754), finite (no inf/NaN), normal (no inf/NaN/subnormals)
* - Rounding: rn (nearest), rz (zero), ru (up), rd (down)
*
* Each kernel takes two fpbits64_t operands and returns their sum as fpbits64_t.
* The default kernel __nv_fp64emu_dadd_rn uses correct rounding and full IEEE range.
*/
// IEEE-754 compliant built-ins
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dadd_rn          (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dadd_rz          (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dadd_ru          (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dadd_rd          (fpbits64_t x, fpbits64_t y);
// Custom accuracy built-ins (accurate = IEEE-754 compliant)
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_high_dadd_rn (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_dadd_rn      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_dadd_rz      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_dadd_ru      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_dadd_rd      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_dadd_rn     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_dadd_rz     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_dadd_ru     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_dadd_rd     (fpbits64_t x, fpbits64_t y);

/**
* Collection of multiplication kernels with different accuracy, range, and rounding modes:
* - Accuracy: cr (correct rounding), ha (high accuracy), la (low accuracy)
* - Range: full (IEEE-754), finite (no inf/NaN), normal (no inf/NaN/subnormals)
* - Rounding: rn (nearest), rz (zero), ru (up), rd (down)
*
* Each kernel takes two fpbits64_t operands and returns their product as fpbits64_t.
* The default kernel __nv_fp64emu_dmul_rn uses correct rounding and full IEEE range.
*/
// IEEE-754 compliant built-ins
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dmul_rn          (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dmul_rz          (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dmul_ru          (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dmul_rd          (fpbits64_t x, fpbits64_t y);
// Custom accuracy built-ins (accurate = IEEE-754 compliant)
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_high_dmul_rn (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_dmul_rn      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_dmul_rz      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_dmul_ru      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_dmul_rd      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_dmul_rn     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_dmul_rz     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_dmul_ru     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_dmul_rd     (fpbits64_t x, fpbits64_t y);


/**
* Collection of subtraction kernels with different accuracy, range, and rounding modes:
* - Accuracy: cr (correct rounding), ha (high accuracy), la (low accuracy)
* - Range: full (IEEE-754), finite (no inf/NaN), normal (no inf/NaN/subnormals)
* - Rounding: rn (nearest), rz (zero), ru (up), rd (down)
*
* Each kernel takes two fpbits64_t operands and returns their difference (x-y) as fpbits64_t.
* The default kernel __nv_fp64emu_dsub_rn uses correct rounding and full IEEE range.
*/
// IEEE-754 compliant built-ins
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dsub_rn          (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dsub_rz          (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dsub_ru          (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dsub_rd          (fpbits64_t x, fpbits64_t y);
// Custom accuracy built-ins (accurate = IEEE-754 compliant)
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_high_dsub_rn (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_dsub_rn      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_dsub_rz      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_dsub_ru      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_dsub_rd      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_dsub_rn     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_dsub_rz     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_dsub_ru     (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_dsub_rd     (fpbits64_t x, fpbits64_t y);

// Negate
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_neg (fpbits64_t x);

/**
* Collection of division kernels with different accuracy, range, and rounding modes:
* - Accuracy: cr (correct rounding), ha (high accuracy), la (low accuracy)
* - Range: full (IEEE-754), finite (no inf/NaN), normal (no inf/NaN subnormals)
* - Rounding: rn (nearest), rz (zero), ru (up), rd (down)
*
* Each kernel takes two fpbits64_t operands and returns their quotient as fpbits64_t.
* The default kernel __nv_fp64emu_ddiv_rn uses correct rounding and full IEEE range.
*/
// IEEE-754 compliant built-ins
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_ddiv_rn          (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_ddiv_rz          (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_ddiv_ru          (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_ddiv_rd          (fpbits64_t x, fpbits64_t y);
// Custom accuracy built-ins (accurate = IEEE-754 compliant)
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_high_ddiv_rn (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_ddiv_rn      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_ddiv_rn     (fpbits64_t x, fpbits64_t y);

/**
* Collection of sqrt kernels with different accuracy, range, and rounding modes:
* - Accuracy: cr (correct rounding), ha (high accuracy), la (low accuracy)
* - Range: full (IEEE-754), finite (no inf/NaN), normal (no inf/NaN subnormals)
* - Rounding: rn (nearest), rz (zero), ru (up), rd (down)
*
* Each kernel takes one fpbits64_t operand and returns its square root as fpbits64_t.  
* The default kernel __nv_fp64emu_dsqrt_rn uses correct rounding and full IEEE range.
*/
// IEEE-754 compliant built-ins
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dsqrt_rn          (fpbits64_t x);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dsqrt_rz          (fpbits64_t x);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dsqrt_ru          (fpbits64_t x);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_dsqrt_rd          (fpbits64_t x);
// Custom accuracy built-ins (accurate = IEEE-754 compliant)
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_high_dsqrt_rn  (fpbits64_t x);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_dsqrt_rn       (fpbits64_t x);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_dsqrt_rn      (fpbits64_t x);

/**
* Collection of fused multiply-add (FMA) kernels with different accuracy, range, and rounding modes:
* - Accuracy: cr (correct rounding), ha (high accuracy), la (low accuracy)
* - Range: full (IEEE-754), finite (no inf/NaN), normal (no inf/NaN/subnormals)
* - Rounding: rn (nearest), rz (zero), ru (up), rd (down)
*
* Each kernel takes three fpbits64_t operands and returns (x*y)+z as fpbits64_t.
* The default kernel __nv_fp64emu_fma_rn uses correct rounding and full IEEE range.
*/
// IEEE-754 compliant built-ins
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_fma_rn          (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_fma_rz          (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_fma_ru          (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t  __nv_fp64emu_fma_rd          (fpbits64_t x, fpbits64_t y, fpbits64_t z);
// Custom accuracy built-ins (accurate = IEEE-754 compliant)
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_high_fma_rn  (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_fma_rn       (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_fma_rz       (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_fma_ru       (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_fma_rd       (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_fma_rn      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_fma_rz      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_fma_ru      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_fma_rd      (fpbits64_t x, fpbits64_t y, fpbits64_t z);

/*
* Collection of MAD kernels with different accuracy, range, and rounding modes:
* - Accuracy: cr (correct rounding), ha (high accuracy), la (low accuracy)
* - Range: full (IEEE-754), finite (no inf/NaN), normal (no inf/NaN/subnormals)
* - Rounding: rn (nearest), rz (zero), ru (up), rd (down)
*
* Each kernel takes three fpbits64_t operands and returns (x*y)+z as fpbits64_t.
* The default kernel __nv_fp64emu_mad_rn uses correct rounding and full IEEE range.
*/
// Custom accuracy built-ins
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mad_rn           (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_high_mad_rn  (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_mid_mad_rn       (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__  fpbits64_t __nv_fp64emu_low_mad_rn      (fpbits64_t x, fpbits64_t y, fpbits64_t z);

/**
* Collection of dot product kernels with default and specific accuracy, range, and rounding mode:
* - Accuracy: ha (high accuracy)
* - Range: normal (no inf/NaN/subnormals)
* - Rounding: rn (nearest)
*
* Each kernel takes four fpbits64_t operands and returns their dot product as fpbits64_t.
* The default kernel __nv_fp64emu_dot_rn uses correct rounding and full IEEE range.
*/
// Custom accuracy built-ins
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dot_rn          (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_dot_rn (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dot_rn      (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dot_rn     (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2);

/**
* Collection of complex multiplication kernels with default and specific accuracy, range, and rounding mode:
* - Accuracy: ha (high accuracy)
* - Range: normal (no inf/NaN/subnormals)
* - Rounding: rn (nearest)
*
* Each kernel takes five fpbits64_t operands and returns the product of the two complex numbers as fpbits64_t.
* The default kernel __nv_fp64emu_cmul_rn uses correct rounding and full IEEE range.
*/
// Custom accuracy built-ins
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_cmul_rn          (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im);
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_high_cmul_rn (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im);
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_mid_cmul_rn      (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im);
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_low_cmul_rn     (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im);

#if __FPEMU_UNPACKED__ == 1

    // Unpack (exact, method/rounding-independent)
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpack (fpbits64_t a);
    // Packs (per rounding mode)
    __FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_pack_rn (fpbits64_unpacked_t a);
    __FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_pack_rz (fpbits64_unpacked_t a);
    __FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_pack_ru (fpbits64_unpacked_t a);
    __FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_pack_rd (fpbits64_unpacked_t a);
    // Conversions to other types
    __FPEMU_BUILTIN_DECL__ int32_t  __nv_fp64emu_unpacked_to_int            (fpbits64_unpacked_t x); 
    __FPEMU_BUILTIN_DECL__ uint32_t __nv_fp64emu_unpacked_to_uint           (fpbits64_unpacked_t x); 
    __FPEMU_BUILTIN_DECL__ int64_t  __nv_fp64emu_unpacked_to_ll             (fpbits64_unpacked_t x); 
    __FPEMU_BUILTIN_DECL__ uint64_t __nv_fp64emu_unpacked_to_ull            (fpbits64_unpacked_t x); 
    __FPEMU_BUILTIN_DECL__ float    __nv_fp64emu_unpacked_to_float          (fpbits64_unpacked_t x); 
    __FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_to_double         (fpbits64_unpacked_t x); 
    // Custom accuracy built-ins
    __FPEMU_BUILTIN_DECL__ double __nv_fp64emu_unpacked_high_to_double  (fpbits64_unpacked_t x);
    __FPEMU_BUILTIN_DECL__ double __nv_fp64emu_unpacked_mid_to_double       (fpbits64_unpacked_t x);
    __FPEMU_BUILTIN_DECL__ double __nv_fp64emu_unpacked_low_to_double      (fpbits64_unpacked_t x);
    // Conversions from other types
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_int             (int32_t x); 
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_uint            (uint32_t x); 
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_ll              (int64_t x); 
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_ull             (uint64_t x); 
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_float           (float x);   
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_double          (double x);
    // Custom accuracy built-ins
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_from_double (double x);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_from_double      (double x);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_from_double     (double x);
    // Casts
    __FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_unpacked_fpbits64_cast_ull (fpbits64_unpacked_t x);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_ull_cast_fpbits64 (uint64_t x);
    // Comparisons
    __FPEMU_BUILTIN_DECL__ bool __nv_fp64emu_unpacked_cmp_eq (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    __FPEMU_BUILTIN_DECL__ bool __nv_fp64emu_unpacked_cmp_ne (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    __FPEMU_BUILTIN_DECL__ bool __nv_fp64emu_unpacked_cmp_le (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    __FPEMU_BUILTIN_DECL__ bool __nv_fp64emu_unpacked_cmp_lt (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    __FPEMU_BUILTIN_DECL__ bool __nv_fp64emu_unpacked_cmp_gt (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    __FPEMU_BUILTIN_DECL__ bool __nv_fp64emu_unpacked_cmp_ge (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    // Negate
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_neg (fpbits64_unpacked_t x);
    //Add
    // IEEE-754 compliant built-ins
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dadd          (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    // Custom accuracy built-ins (accurate = IEEE-754 compliant)
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dadd (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dadd      (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dadd     (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    // Mul
    // IEEE-754 compliant built-ins
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dmul          (fpbits64_unpacked_t x, fpbits64_unpacked_t y); 
    // Custom accuracy built-ins (accurate = IEEE-754 compliant)
  __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t   __nv_fp64emu_unpacked_high_dmul (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dmul      (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dmul     (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    // Sub
    // IEEE-754 compliant built-ins
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dsub          (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    // Custom accuracy built-ins (accurate = IEEE-754 compliant)
   __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t  __nv_fp64emu_unpacked_high_dsub (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dsub      (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dsub     (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    // Div
    // IEEE-754 compliant built-ins
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_ddiv          (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    // Custom accuracy built-ins (accurate = IEEE-754 compliant)
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_ddiv (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_ddiv      (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_ddiv     (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
    // Sqrt
    // IEEE-754 compliant built-ins
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dsqrt          (fpbits64_unpacked_t x);
    // Custom accuracy built-ins (accurate = IEEE-754 compliant)
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dsqrt (fpbits64_unpacked_t x);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dsqrt      (fpbits64_unpacked_t x);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dsqrt     (fpbits64_unpacked_t x);
    // Fma
    // IEEE-754 compliant built-ins
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_fma          (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
    // Custom accuracy built-ins (accurate = IEEE-754 compliant)
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_fma (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_fma      (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_fma     (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
    // Mad
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mad          (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_mad (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_mad      (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_mad     (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
    // Dot
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dot          (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dot (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dot      (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2);
    __FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dot     (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2);
    // Cmul
    __FPEMU_BUILTIN_DECL__ void __nv_fp64emu_unpacked_cmul          (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im);
    __FPEMU_BUILTIN_DECL__ void __nv_fp64emu_unpacked_high_cmul (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im);
    __FPEMU_BUILTIN_DECL__ void __nv_fp64emu_unpacked_mid_cmul      (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im);
    __FPEMU_BUILTIN_DECL__ void __nv_fp64emu_unpacked_low_cmul     (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im);
                                           
#endif // __FPEMU_UNPACKED__ == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPEMU_LIB_H
