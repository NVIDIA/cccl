//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_COMMON_H
#define _CUDA___FP_FPMP_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_common.h - Public API surface shared by the FPMP headers
    ======================================================================================================
    This header carries only the user-facing pieces that both the fpmp2 class
    (<cuda/__fp/fpmp.h>) and the arithmetic cores (<cuda/__fp/fpmp_impl.h>) need to
    agree on:

      - The public accuracy selector fpmp2_accuracy
      - The public compile-mode / behavior knobs:
          * CCCL_FPMP_LIB / CCCL_FPMP_INLINE            (header-only vs library linkage)
          * CCCL_FPMP_EXPLICIT_CASTS                    (strict narrowing conversions)
          * CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP          (integer-only double -> fpmp2)
          * CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE          (integer-only fpmp2 -> double)
          * CCCL_FPMP_FP128_MATH_FALLBACK               (binary128 fp64mp2 math)
        together with their mapping to the internal switches (_CCCL_FPMP_USE_LIB,
        _CCCL_FPMP_EXPLICIT, _CCCL_FPMP_USE_OPT_FROM_DOUBLE, _CCCL_FPMP_USE_OPT_TO_DOUBLE,
        _CCCL_FPMP_FP128_MATH_FALLBACK).

    All library-internal machinery (decorator/ABI/declaration macros, the fp128
    detection/typedef, the bit-cast plumbing, the tuning knobs, and the __fpmp_*
    helper functions) lives in <cuda/__fp/fpmp_impl.h>. Keeping the public and
    internal pieces apart lets every FP header compile standalone.
*/

// ---------------------------------------------------------------------------
// User-facing configuration (public knobs)
// ---------------------------------------------------------------------------
// CCCL_FPMP_LIB: Compilation mode control.
//   1 = link against precompiled library (maps to _CCCL_FPMP_USE_LIB)
//   0 = header-only inline mode (default)
// CCCL_FPMP_INLINE is the inverse alias: CCCL_FPMP_INLINE=1 is equivalent to CCCL_FPMP_LIB=0.
#ifndef CCCL_FPMP_LIB
#  ifdef CCCL_FPMP_INLINE
#    if CCCL_FPMP_INLINE == 1
#      define CCCL_FPMP_LIB 0
#    else
#      define CCCL_FPMP_LIB 1
#    endif
#  else
#    define CCCL_FPMP_LIB 0
#  endif
#endif
#ifndef CCCL_FPMP_INLINE
#  if CCCL_FPMP_LIB == 1
#    define CCCL_FPMP_INLINE 0
#  else
#    define CCCL_FPMP_INLINE 1
#  endif
#endif
#if CCCL_FPMP_LIB == 1 && !defined(_CCCL_FPMP_USE_LIB)
#  define _CCCL_FPMP_USE_LIB
#endif

// CCCL_FPMP_EXPLICIT_CASTS controls whether lossy/narrowing conversions INTO fpmp2
// are explicit. It gates only the constructors:
//   - double      -> fp32mp2   (narrowing)
//   - fp64mp2     -> fp32mp2   (narrowing)
//   - __float128  -> fp64mp2   (narrowing)
//   - int32_t / uint32_t -> fpmp2
//   - int64_t / uint64_t -> fpmp2
// The conversion OUT to double (operator double()) is always implicit and is NOT
// affected by this macro (it is a value-preserving widening conversion).
//
// Default is 1 (lossy casts explicit), matching CCCL's strict-cast conventions.
//
// Set to 0 (fully-implicit model) when it eases adoption more than strictness
// helps, e.g.:
//   - migrating a large existing codebase where `double`/`float` are being
//     replaced by fpmp2 as a near drop-in: implicit casts let the existing call
//     sites and mixed-type expressions compile unchanged instead of requiring an
//     explicit cast at every narrowing boundary, and
//   - rapid prototyping, where minimizing edit churn speeds up the process.
// Warning: with implicit casts the compiler will silently perform narrowing
// conversions INTO fpmp2 (double/fp64mp2/__float128 -> smaller fpmp2, integers
// -> fpmp2). This can:
//   - silently drop precision at conversions you did not intend (e.g. a stray
//     `double` in an expression pulls an fp32mp2 value down to double-float), and
//   - introduce unintended conversions / round-trips that hurt accuracy or
//     performance (e.g. accidental FP64 use) without any diagnostic.
// Keep the default 1 unless you have weighed these risks; prefer explicit casts
// at the boundaries you actually intend once migration settles.
#ifndef CCCL_FPMP_EXPLICIT_CASTS
#  define CCCL_FPMP_EXPLICIT_CASTS 1
#endif
#if CCCL_FPMP_EXPLICIT_CASTS == 1
#  define _CCCL_FPMP_EXPLICIT explicit
#else
#  define _CCCL_FPMP_EXPLICIT
#endif

// CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP / CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE: use integer
// bit manipulation instead of FP64 arithmetic for the double<->fpmp2 conversions.
// This avoids the slow FP64 pipeline on GPUs with limited double-precision throughput
// (e.g. consumer GPUs with a 1:64 ratio), which is the common target for the FP32-based
// fp32mp2 double-word type. (The optimization applies to the float-based fp32mp2
// conversions; fp64mp2 conversions are inherently FP64 either way.)
//
// Both default to 1 (integer conversions on the hot path). Set either to 0 to fall back
// to the plain cast-based conversions (static_cast + FP64 add) if you hit:
//   - register pressure / reduced occupancy from the extra integer work, or
//   - a GPU with high FP64 throughput (e.g. datacenter A100/H100, ~1:2), where the FP64
//     path is already cheap and the integer path adds instructions for little gain.
// e.g. compile with -DCCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE=0 (and/or ..._DOUBLE_TO_FPMP=0).
#ifndef CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP
#  define CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP 1
#endif
#ifndef CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE
#  define CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE 1
#endif
#ifndef _CCCL_FPMP_USE_OPT_FROM_DOUBLE
#  define _CCCL_FPMP_USE_OPT_FROM_DOUBLE CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP
#endif
#ifndef _CCCL_FPMP_USE_OPT_TO_DOUBLE
#  define _CCCL_FPMP_USE_OPT_TO_DOUBLE CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE
#endif

// CCCL_FPMP_FP128_MATH_FALLBACK: whether the fp64mp2 math functions compute in binary128
// (~113-bit) instead of double. Unset by default, in which case the library decides per
// compilation pass: a host-only build takes the quad path wherever fp128 is available,
// while in a CUDA compilation only the device pass does, and only where every targeted
// architecture can run fp128 (sm_100 and later). A .cu file therefore does not silently
// acquire a libquadmath dependency its host-only counterpart never had.
//
// Set it to 1 to put both passes on the quad path, which is what a program wants when the
// host and the device halves of a computation have to agree to the last bits:
//
//   nvcc -arch=sm_100 -DCCCL_FPMP_FP128_MATH_FALLBACK=1 app.cu -lquadmath
//
// -lquadmath is the host side of that bargain on x86_64 GCC, where the quad entry points
// (expq, sinq, ...) live in libquadmath; hosts whose long double is IEEE binary128
// (AArch64, PPC64LE, s390x) call libm's *l entry points and need nothing extra. Setting it
// to 1 for a target whose device cannot run fp128 makes the device pass fail to compile,
// since its bodies then ask for quad arithmetic the architecture does not have. Setting it
// to 0 keeps every pass on double.
//
// Every translation unit in a program has to agree on the value, as does the library build
// in library mode: it selects which implementation the fp64mp2 entry points get. The
// derivation, including the automatic case, lives with those bodies in
// <cuda/__fp/fpmp_math_impl.h>.
#ifdef CCCL_FPMP_FP128_MATH_FALLBACK
#  ifndef _CCCL_FPMP_FP128_MATH_FALLBACK
#    define _CCCL_FPMP_FP128_MATH_FALLBACK CCCL_FPMP_FP128_MATH_FALLBACK
#  endif
#endif

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
/*
// Accuracy level for fpmp arithmetic (public). Named fpmp2_accuracy, so
// callers write e.g. fpmp2<float, fpmp2_accuracy::high>.
// mid is the Dekker-based split and error accumulation technique
// high is the Thall-based split and error accumulation technique
// low is the fast arithmetic operation without re-normalizations
// def is the default selector; equals mid.
*/
enum struct fpmp2_accuracy
{
  unset = -1,
  low   = 1,
  mid   = 2,
  high  = 3,
  def   = mid,
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_COMMON_H
