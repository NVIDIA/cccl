//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_COMMON_H
#define _CUDA___FP_FPEMU_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/**
 * @file fpemu_common.hpp
 * @brief Common definitions, macros, and enumerations for the FPEMU library
 *
 * This header provides the core definitions, macros, and enumerations used throughout 
 * the FPEMU library. It defines:
 *
 * - Macros for controlling inline vs library compilation
 * - Host/device decorators for CUDA compilation
 * - Function declaration macros for different contexts
 * - Rounding modes (nearest, zero, up, down)
 * - Accuracy levels (high, mid, low)
 *
 * The macros handle:
 * - Inline vs library compilation modes
 * - Host vs device code compilation for CUDA
 * - Function visibility and linkage
 *
 * This provides a unified way to handle compilation across different platforms
 * and compilation modes while maintaining consistent behavior.
 */

#if !defined(__CUDA_LIBDEVICE__)
#include <cstdint>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#endif

/*********************************************************************
 * Compilation mode macros
 *********************************************************************/

// FPEMU_LIB: Compilation mode control.
//   1 = link against precompiled library (maps to __FPEMU_USE_LIB__)
//   0 = header-only inline mode (default)
// FPEMU_INLINE is the inverse alias: FPEMU_INLINE=1 is equivalent to FPEMU_LIB=0.
#ifndef FPEMU_LIB
    #ifdef FPEMU_INLINE
        #if FPEMU_INLINE == 1
            #define FPEMU_LIB 0
        #else
            #define FPEMU_LIB 1
        #endif
    #else
        #define FPEMU_LIB 0
    #endif
#endif
#ifndef FPEMU_INLINE
    #if FPEMU_LIB == 1
        #define FPEMU_INLINE 0
    #else
        #define FPEMU_INLINE 1
    #endif
#endif
#if FPEMU_LIB == 1 && !defined(__FPEMU_USE_LIB__)
    #define __FPEMU_USE_LIB__
#endif

// If neither inline nor lib is defined internally, default to inline
#if !defined __FPEMU_INLINE__ && !defined __FPEMU_BUILD_LIB__ && !defined __FPEMU_USE_LIB__
    #define __FPEMU_INLINE__
#endif

// If neither host nor device is defined, default to device
#if defined __CUDACC__
    #undef  __FPEMU_HOST__
    #undef  __FPEMU_DEVICE__
    #define __FPEMU_DEVICE__
#else
    #undef  __FPEMU_DEVICE__
    #undef  __FPEMU_HOST__
    #define __FPEMU_HOST__
#endif

/*
// Custom ABI for builtins in static library
*/
#if ((defined __CUDA_LIBDEVICE__) || (defined __FPEMU_BUILD_LIB__) || (defined __FPEMU_USE_LIB__)) && \
     (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13))
  #ifndef __FPEMU_ABI_PRESERVE_N_DATA__
    #define __FPEMU_ABI_PRESERVE_N_DATA__    -1
  #endif
  #ifndef __FPEMU_ABI_PRESERVE_N_CONTROL__
    #define __FPEMU_ABI_PRESERVE_N_CONTROL__ -1
  #endif
  #if (__FPEMU_ABI_PRESERVE_N_DATA__ != -1) && (__FPEMU_ABI_PRESERVE_N_CONTROL__ != -1)
    #define __FPEMU_ABI_STR1__(x) #x
    #define __FPEMU_ABI_STR__(x) __FPEMU_ABI_STR1__(x)
    #define __FPEMU_ABI_PRAGMA_TEXT__ nv_abi preserve_n_data(__FPEMU_ABI_PRESERVE_N_DATA__) preserve_n_control(__FPEMU_ABI_PRESERVE_N_CONTROL__)
    #define __FPEMU_ABI__ _Pragma(__FPEMU_ABI_STR__(__FPEMU_ABI_PRAGMA_TEXT__))
  #else
    #define __FPEMU_ABI__
  #endif
#else
  #define __FPEMU_ABI__
#endif

/*********************************************************************
 * Declaration macros
 *********************************************************************/

// Define the target device declaration
#if defined __FPEMU_DEVICE__
    #define __FPEMU_INLINE_DECL__      __forceinline__
    #define __FPEMU_HOST_DECL__        __host__
    #define __FPEMU_DEVICE_DECL__      __device__
    #define __FPEMU_HOST_DEVICE_DECL__ __host__ __device__
    #define __FPEMU_MANAGED_DECL__     __managed__
#else // defined __FPEMU_HOST__
    #define __FPEMU_INLINE_DECL__      inline
    #define __FPEMU_HOST_DECL__ 
    #define __FPEMU_DEVICE_DECL__
    #define __FPEMU_HOST_DEVICE_DECL__
    #define __FPEMU_MANAGED_DECL__
#endif

// Define the subkernel declaration
#if defined __CUDA_LIBDEVICE__
    #define __FPEMU_INTERNAL_DECL__  static __FPEMU_INLINE_DECL__ __FPEMU_HOST_DEVICE_DECL__
    #define __FPEMU_IMPL_DECL__      static __FPEMU_INLINE_DECL__ __FPEMU_HOST_DEVICE_DECL__
    #define __FPEMU_BUILTIN_DECL__   __FPEMU_ABI__ extern "C" __FPEMU_HOST_DEVICE_DECL__
    #define __FPEMU_API_DECL__       static __FPEMU_INLINE_DECL__ __FPEMU_HOST_DEVICE_DECL__
#elif defined __FPEMU_INLINE__
    #define __FPEMU_INTERNAL_DECL__  static __FPEMU_INLINE_DECL__ __FPEMU_HOST_DEVICE_DECL__
    #define __FPEMU_IMPL_DECL__      static __FPEMU_INLINE_DECL__ __FPEMU_HOST_DEVICE_DECL__
    #define __FPEMU_BUILTIN_DECL__   static __FPEMU_INLINE_DECL__ __FPEMU_HOST_DEVICE_DECL__  
    #define __FPEMU_API_DECL__       static __FPEMU_INLINE_DECL__ __FPEMU_HOST_DEVICE_DECL__
#else // __FPEMU_BUILD_LIB__ or __FPEMU_USE_LIB__
    #define __FPEMU_INTERNAL_DECL__  static __FPEMU_INLINE_DECL__ __FPEMU_HOST_DEVICE_DECL__
    #define __FPEMU_IMPL_DECL__      static __FPEMU_INLINE_DECL__ __FPEMU_HOST_DEVICE_DECL__
    #define __FPEMU_BUILTIN_DECL__   __FPEMU_ABI__ extern "C" __FPEMU_HOST_DEVICE_DECL__
    #define __FPEMU_API_DECL__       static __FPEMU_INLINE_DECL__ __FPEMU_HOST_DEVICE_DECL__
#endif

/*********************************************************************
 * Default configuration values
 *********************************************************************/

// Define the default values for the enums
#ifndef __FPEMU_DEFAULT_ROUNDING__
    #define __FPEMU_DEFAULT_ROUNDING__ rn
#endif

// Unpacked API presence. Default ON so the packed (fpbits64_t) legacy API and
// the unpacked (fpbits64_unpacked_t) API co-exist in the same package: this
// gates the *_unpacked cores, the universal pack/unpack, and the *_unpacked
// builtins/operators. Set to 0 only to strip the unpacked API entirely.
#ifndef __FPEMU_UNPACKED__
    #define __FPEMU_UNPACKED__ 1
#endif

// Route the PACKED API through the unpack -> *_unpacked core -> pack pipeline
// instead of the legacy fused kernels. Set by the Makefile's PACKED_VIA_UNPACKED=y.
// This is a TESTING knob (it lets the packed test harness exercise the unpacked
// cores); default OFF keeps the packed API on its legacy implementations.
#ifndef __FPEMU_PACKED_VIA_UNPACKED__
    #define __FPEMU_PACKED_VIA_UNPACKED__ 0
#endif

/*********************************************************************
 * Enumerations
 *********************************************************************/

/**
* @brief Bit representation of a double-precision floating point number
*
* This struct provides a simple wrapper around a 64-bit integer value that represents
* the IEEE-754 binary encoding of a double-precision floating point number.
* The value field contains the raw bits including sign, exponent and mantissa.
*/
typedef uint64_t fpbits64_t;

/**
* @brief Unpacked representation of a double-precision floating point number
* 
* This struct represents a double-precision floating point number in an unpacked format.
* It contains the sign, exponent, and mantissa components.
*/
typedef struct 
{
    uint32_t sign;
    uint32_t exponent;
    uint64_t mantissa;
} fpbits64_unpacked_t;

/**
* @brief Accuracy level for floating-point emulation (public).
*
* Defined directly in cuda::experimental (no internal fpemu:: namespace) and named
* fp64emu_accuracy, so callers write e.g. fp64emu_t<fp64emu_accuracy::high>.
* - high: Correctly rounded with full IEEE-754 range (infinities, NaNs, subnormals)
* - mid:  High accuracy (1-2 ULP) with normal range
* - low:  Low accuracy (up to half mantissa) with normal range
* - def:  Default selector; equals high so the default is IEEE-correct.
*/
enum struct fp64emu_accuracy
{
    unset = -1,
    low   =  1,
    mid   =  2,
    high  =  3,
    def   =  3,
};

namespace fpemu
{
    /**
    * @brief Rounding modes for floating point operations
    *
    * Enumeration of supported rounding modes:
    * - rn: Round to nearest (ties to even) - default IEEE-754 rounding
    * - rz: Round toward zero (truncation)
    * - ru: Round toward positive infinity 
    * - rd: Round toward negative infinity
    */
    enum struct rounding
    {
        unset = -1,
        rn    =  0,
        rz    =  1,
        ru    =  2,
        rd    =  3,
        def   = __FPEMU_DEFAULT_ROUNDING__
    };

} // namespace fpemu

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPEMU_COMMON_H
