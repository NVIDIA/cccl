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
 * @file fpemu_common.h
 * @brief Public API surface shared by the FPEMU headers
 *
 * This header carries only the user-facing pieces that both the fpemu class
 * (<cuda/__fp/fpemu.h>) and the emulation cores (<cuda/__fp/fpemu_impl.h> and the
 * per-operation fpemu_impl_<op>.h headers) need to agree on:
 *
 * - The public accuracy selector fpemu_accuracy
 * - The public compile-mode knobs CCCL_FPEMU_LIB / CCCL_FPEMU_INLINE
 *
 * All library-internal machinery (decorator/ABI/declaration macros, the raw-bits
 * vocabulary types __fpbits64/__fpbits64_unpacked, the internal __fpemu_rounding
 * enum, and the helper functions) lives in <cuda/__fp/fpemu_impl.h>. Keeping the
 * public and internal pieces apart lets every FP header compile standalone.
 */

// ---------------------------------------------------------------------------
// User-facing configuration (public compile-mode knobs)
// ---------------------------------------------------------------------------
// CCCL_FPEMU_LIB: Compilation mode control.
//   1 = link against precompiled library (maps to _CCCL_FPEMU_USE_LIB)
//   0 = header-only inline mode (default)
// CCCL_FPEMU_INLINE is the inverse alias: CCCL_FPEMU_INLINE=1 is equivalent to CCCL_FPEMU_LIB=0.
#ifndef CCCL_FPEMU_LIB
    #ifdef CCCL_FPEMU_INLINE
        #if CCCL_FPEMU_INLINE == 1
            #define CCCL_FPEMU_LIB 0
        #else
            #define CCCL_FPEMU_LIB 1
        #endif
    #else
        #define CCCL_FPEMU_LIB 0
    #endif
#endif
#ifndef CCCL_FPEMU_INLINE
    #if CCCL_FPEMU_LIB == 1
        #define CCCL_FPEMU_INLINE 0
    #else
        #define CCCL_FPEMU_INLINE 1
    #endif
#endif
#if CCCL_FPEMU_LIB == 1 && !defined(_CCCL_FPEMU_USE_LIB)
    #define _CCCL_FPEMU_USE_LIB
#endif

// The prologue/epilogue pair and the standard-library include are skipped in
// __CUDA_LIBDEVICE__ builds. The namespace and the
// enum below are plain C++ and are always emitted so the braces stay balanced and
// the emulation cores (which take fpemu_accuracy as a template parameter) can see
// it in every build.
#if !defined(__CUDA_LIBDEVICE__)
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>
#endif

namespace cuda::experimental
{

/**
* @brief Accuracy level for floating-point emulation (public).
*
* Named fpemu_accuracy, so callers write e.g. fpemu<double, fpemu_accuracy::high>.
* - high: Correctly rounded with full IEEE-754 range (infinities, NaNs, subnormals)
* - mid:  High accuracy (1-2 ULP) with normal range
* - low:  Low accuracy (up to half mantissa) with normal range
* - def:  Default selector; equals high so the default is IEEE-correct.
*/
enum struct fpemu_accuracy
{
    unset = -1,
    low   =  1,
    mid   =  2,
    high  =  3,
    def   =  3,
};

} // namespace cuda::experimental

#if !defined(__CUDA_LIBDEVICE__)
#include <cuda/std/__cccl/epilogue.h>
#endif

#endif // _CUDA___FP_FPEMU_COMMON_H
