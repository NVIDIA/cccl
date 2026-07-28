//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_IMPL_H
#define _CUDA___FP_FPMP_IMPL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/*
    fpmp_impl.h - Core Multi-Precision Arithmetic Operations
    ======================================================================================================
    This header provides the core low-level (C-style) API for fpmp2 arithmetic.
    It implements fundamental operations using error-free transformations and supports multiple accuracy
    levels (low/mid/high; default == mid). The same templates are used for float-based (fp32mp2) and, when enabled,
    double-based (fp64mp2) variants.

    Supported Operations:
    -------------------------------------------------------------------------
    - Type Conversions:
        * __fpmp2_from_double, __fpmp2_from_int, __fpmp2_from_uint
        * __fpmp2_from_ll, __fpmp2_from_ull
        * __fpmp2_to_double, __fpmp2_to_float
        * __fpmp2_to_int, __fpmp2_to_uint, __fpmp2_to_ll, __fpmp2_to_ull
        * __fpmp2_from_double supports CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP (integer-only, no FP64)
        * __fpmp2_to_double supports CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE (integer-only, no FP64)

    - Basic Arithmetic:
        * Addition: __fpmp2_add, __fpmp2_low_add, __fpmp2_high_add
        * Subtraction: __fpmp2_sub, __fpmp2_low_sub, __fpmp2_high_sub
        * Accumulate: __fpmp2_acc, __fpmp2_low_acc, __fpmp2_high_acc (optimized single-component add)
        * Multiplication: __fpmp2_mul, __fpmp2_low_mul, __fpmp2_high_mul (if _CCCL_FPMP_USE_ACCURATE_MUL == 1)
        * Division: __fpmp2_div, __fpmp2_low_div, __fpmp2_high_div (if _CCCL_FPMP_USE_ACCURATE_DIV == 1)
        * Negation: __fpmp2_neg
        * Renormalization: __fpmp2_renormalize

    - Advanced Operations:
        * Square Root: __fpmp2_sqrt (Newton-Raphson iteration)
        * Reciprocal Square Root: __fpmp2_rsqrt (Karp-Markstein algorithm)
        * Fused Multiply-Add: __fpmp2_fma, __fpmp2_low_fma
        * Multiply-Add with Rounding: __fpmp2_mad

    - Comparison Operations:
        * __fpmp2_cmp_eq, __fpmp2_cmp_ne
        * __fpmp2_cmp_lt, __fpmp2_cmp_gt
        * __fpmp2_cmp_le, __fpmp2_cmp_ge

    - Utility Operations:
        * __fpmp2_bit_cast : IEEE-754 format bit representation

    - Atomic Operations (CUDA device only):
        * __fpmp2_atomicAdd : Atomic addition with CAS loop
        * __fpmp2_atomicSub : Atomic subtraction with CAS loop

    Implementation Details:
    -------------------------------------------------------------------------
    - Uses Dekker's error-free transformation algorithms (2Sum, 2Mult)
    - Supports three accuracy levels: low, mid (default), and high
    - Template-based for both float (fp32) and double (fp64) precision
    - Provides both inline implementations and library declarations
    - All operations maintain the (hi, lo) representation invariant
    - __fpmp2_from_double uses integer bit manipulation when CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP == 1

    Accuracy Levels:
    -------------------------------------------------------------------------
    - mid (Dekker, default): Balanced accuracy and performance
    - low: Minimizes renormalization steps for speed
    - high (Thall): Maximum precision with additional refinement steps
    - (Optional) FPAN-style normalization for addition in high mode (compile-time selection)

    Reference Papers:
    -------------------------------------------------------------------------
    [1] Dekker, T. (1971). A floating-point technique for extending available precision.
    [2] Karp & Markstein (1997). High Precision Division and Square Root. ACM TOMS.
    [3] Thall, A. Extended-Precision Floating-Point Numbers for GPU Computation.
    [4] Nagai et al. (2008). Fast Quadruple Precision Arithmetic Library. ICCS '08.
    [5] Fukuda et al. (2010). FPAN: A Fast Pairwise Addition Normalization Algorithm. SC '10.
*/

#include <cuda/__fp/fpmp_common.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/native_type.h> // __fpmp_fp128 is CCCL's native binary128 type
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cfloat> // LDBL_* , to recognize a binary128 long double
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>

#include <iostream>
#include <string>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// ===================================================================
// Internal implementation vocabulary and helpers (moved from fpmp_common.h;
// fpmp_common.h now carries only the public API surface).
// ===================================================================
/*
// Require C++11 minimum for alignas support
*/
#if !defined(__cplusplus) || __cplusplus < 201103L
#  error "This header requires C++11 or later (for alignas, constexpr, etc.)"
#endif

/*
// Check for if constexpr support (C++17)
*/
#if __cplusplus < 201703L
#  warning "This header works best with C++17 or later for if constexpr support"
#endif

/*
// External configuration macros
*/

// The public knobs CCCL_FPMP_LIB / CCCL_FPMP_INLINE / CCCL_FPMP_EXPLICIT_CASTS /
// CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP / CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE are
// defined in the user-facing entry point <cuda/__fp/fpmp.h>, which also maps them
// to the internal switches (_CCCL_FPMP_USE_LIB, _CCCL_FPMP_EXPLICIT,
// _CCCL_FPMP_USE_OPT_FROM_DOUBLE, _CCCL_FPMP_USE_OPT_TO_DOUBLE) used below. The
// standalone libcufp build TU includes this header directly and drives the mode
// via _CCCL_FPMP_BUILD_LIB, so it does not need the public knobs.

/*
// Define if double precision based types (double-double) are enabled
*/

/*
// libquadmath availability detection
// -----------------------------------
// libquadmath ships only with GCC's runtime on x86 (32/64-bit) and provides
// <quadmath.h>, the *q math suite (expq, sinq, sqrtq, ...), and the
// __float128 type. It is NOT available on:
//   - Non-x86 architectures (ARM, ARM64, RISC-V, PowerPC, ...)
//   - MSVC (no GCC runtime)
//   - Most Windows toolchains (MinGW configurations vary)
//
// Override at compile time with -DFPMP_HOST_SUPPORTS_LIBQUADMATH=1 (force enable) or
// -DFPMP_HOST_SUPPORTS_LIBQUADMATH=0 (force disable) when the auto-detection is wrong
// for your environment.
*/
#ifndef _CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH
#  if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && !defined(_MSC_VER) \
    && !defined(_WIN32)
#    define _CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH 1
#  else
#    define _CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH 0
#  endif
#endif

/*
// Detection of systems where 'long double' is 128-bit IEEE 754 quadruple precision
// ---------------------------------------------------------------------------------
// On these platforms, 'long double' has the same binary format as __float128,
// enabling the use of standard <cmath> functions (expl, sinl, sqrtl, logl, etc.)
// instead of libquadmath (*q functions like expq, sinq, sqrtq, logq).
//
// The test is on the format rather than on a list of architectures, and is the same one
// CCCL applies in <cuda/std/__floating_point/format.h> to classify long double as
// __fp_format::__binary128. It is true on aarch64, s390x and PowerPC built with
// -mabi=ieeelongdouble, and false wherever long double is x87 80-bit (x86) or plain
// binary64 (Windows, including ARM64) - no special case needed for either.
//
// It would be preferable to ask __fp_has_native_type_v<__binary128> directly, but that is
// a constexpr variable and this decision has to be made by the preprocessor; the
// static_assert next to __fpmp_fp128 below keeps the two answers in agreement.
//
// _CCCL_HAS_LONG_DOUBLE() additionally excludes CUDA compilation, where long double is
// not a usable device type, and honors CCCL_DISABLE_LONG_DOUBLE_SUPPORT.
*/
#ifndef _CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128
#  if _CCCL_HAS_LONG_DOUBLE() && LDBL_MIN_EXP == -16381 && LDBL_MAX_EXP == 16384 && LDBL_MANT_DIG == 113
#    define _CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128 1
#  else
#    define _CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128 0
#  endif
#endif

/*
// Automatic detection of FP128 support based on platform capabilities
// --------------------------------------------------------------------
// FP128 is enabled when the platform provides 128-bit IEEE 754 quadruple
// precision arithmetic, either via:
//   - __float128, as reported by _CCCL_HAS_FLOAT128()
//   - 128-bit long double (aarch64, s390x, PowerPC with IEEE long double)
//
// _CCCL_HAS_FLOAT128() is CCCL's own detection and already accounts for everything that
// decides whether the type can even be named: a host compiler that provides it, Linux,
// a non-ARM64 host, __int128 support, and - during device compilation - NVCC/NVRTC 12.8+
// targeting sm_100+ (Blackwell). Asking it instead of re-deriving the platform matrix
// here is what keeps ARM64 and Windows working.
*/
#ifndef _CCCL_FPMP_FP128_ENABLE
#  if defined(__CUDACC__)
// Under CUDA the quad conversions are device-only, as they were before.
#    if _CCCL_DEVICE_COMPILATION() && _CCCL_HAS_FLOAT128()
#      define _CCCL_FPMP_FP128_ENABLE 1
#    else
#      define _CCCL_FPMP_FP128_ENABLE 0
#    endif
#  else
#    if _CCCL_HAS_FLOAT128() || (_CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128 == 1)
#      define _CCCL_FPMP_FP128_ENABLE 1
#    else
#      define _CCCL_FPMP_FP128_ENABLE 0
#    endif
#  endif
#endif

/*
// fp128 math functions fallback to system implementation enabling
*/
#ifndef _CCCL_FPMP_FP128_MATH_FALLBACK
#  if (_CCCL_FPMP_FP128_ENABLE == 1)
#    define _CCCL_FPMP_FP128_MATH_FALLBACK 1
#  else
#    define _CCCL_FPMP_FP128_MATH_FALLBACK 0
#  endif
#endif

/*
// Internal 128-bit floating-point type definition
// ------------------------------------------------
// __fpmp_fp128 is the library's internal quad-precision type. Which type that is on a
// given platform is CCCL's decision, not ours: __fp_native_type_t resolves binary128 to
// __float128 where _CCCL_HAS_FLOAT128() and to a 128-bit long double otherwise (aarch64,
// s390x, PowerPC with IEEE long double).
//
// The static_assert catches the case where the preprocessor conditions that set
// _CCCL_FPMP_FP128_ENABLE - or a hand override of it - claim a quad type that CCCL does
// not actually have, which would otherwise show up as __fpmp_fp128 being void.
//
// Only defined when _CCCL_FPMP_FP128_ENABLE == 1.
*/
#if (_CCCL_FPMP_FP128_ENABLE == 1)
static_assert(::cuda::std::__fp_has_native_type_v<::cuda::std::__fp_format::__binary128>,
              "_CCCL_FPMP_FP128_ENABLE=1 but CCCL reports no native 128-bit float type");
using __fpmp_fp128 = ::cuda::std::__fp_native_type_t<::cuda::std::__fp_format::__binary128>;
#endif

/*
// Internal macro definitions
*/

/*
// Custom ABI for builtins in static library
*/
#if ((defined __CUDA_LIBDEVICE__) || (defined _CCCL_FPMP_BUILD_LIB) || (defined _CCCL_FPMP_USE_LIB)) \
  && (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13))
#  ifndef _CCCL_FPMP_ABI_PRESERVE_N_DATA
#    define _CCCL_FPMP_ABI_PRESERVE_N_DATA -1
#  endif
#  ifndef _CCCL_FPMP_ABI_PRESERVE_N_CONTROL
#    define _CCCL_FPMP_ABI_PRESERVE_N_CONTROL -1
#  endif
#  if (_CCCL_FPMP_ABI_PRESERVE_N_DATA != -1) && (_CCCL_FPMP_ABI_PRESERVE_N_CONTROL != -1)
#    define _CCCL_FPMP_ABI_STR1(x) #x
#    define _CCCL_FPMP_ABI_STR(x)  _CCCL_FPMP_ABI_STR1(x)
#    define _CCCL_FPMP_ABI_PRAGMA_TEXT \
      nv_abi preserve_n_data(_CCCL_FPMP_ABI_PRESERVE_N_DATA) preserve_n_control(_CCCL_FPMP_ABI_PRESERVE_N_CONTROL)
#    define _CCCL_FPMP_ABI _Pragma(_CCCL_FPMP_ABI_STR(_CCCL_FPMP_ABI_PRAGMA_TEXT))
#  else
#    define _CCCL_FPMP_ABI
#  endif
#else
#  define _CCCL_FPMP_ABI
#endif

/*
// Builtin declaration macros.
//
// Public and internal functions are decorated at each call site with CCCL
// visibility macros directly (_CCCL_API inline, _CCCL_DEVICE_API inline,
// _CCCL_TRIVIAL_API, _CCCL_TRIVIAL_DEVICE_API); these expand correctly for both
// CUDA and host-only compilation. The only decorators that still need dedicated
// macros are the extern-"C" ABI symbols used when building or linking the
// standalone libcufp library.
*/
#if (defined _CCCL_FPMP_BUILD_LIB) || (defined _CCCL_FPMP_USE_LIB)
#  define _CCCL_FPMP_BUILTIN_DECL        _CCCL_FPMP_ABI extern "C" _CCCL_HOST_DEVICE
#  define _CCCL_FPMP_BUILTIN_DEVICE_DECL _CCCL_FPMP_ABI extern "C" _CCCL_DEVICE
#else
#  define _CCCL_FPMP_BUILTIN_DECL        _CCCL_TRIVIAL_API
#  define _CCCL_FPMP_BUILTIN_DEVICE_DECL _CCCL_TRIVIAL_DEVICE_API
#endif

/*
// Core per-type implementation linkage.
//
// The generic per-type kernels (e.g. __fpmp2_add<float>) share their mangled name
// with the consumer-side forwarder templates emitted in _CCCL_FPMP_USE_LIB mode
// (which just call the extern-"C" __fp32mp2_* ABI symbols). When the standalone
// libcufp library is built (_CCCL_FPMP_BUILD_LIB) and later linked into consumer
// code with device LTO (-dlto), those two same-named weak_odr definitions -- the
// real implementation in the library and the forwarder in the consumer -- are
// ODR-merged by nvlink. The merge fuses __fp32mp2_add <-> __fpmp2_add<float> into
// mutual recursion that the optimizer then eliminates, producing empty kernels
// (0 correct bits, unstable timing) for every op routed through the library.
//
// Giving the core templates INTERNAL linkage in library-build mode removes them
// from ODR merging, so the extern-"C" wrapper keeps its own real body. In every
// other mode (pure inline consumer, or the consumer side of a library build) the
// decoration is unchanged, preserving CCCL styling and inline-mode codegen (adding
// static in inline mode caps registers and hurts performance -- see the note in
// fpmp_math_impl_special.h).
*/
#if defined(_CCCL_FPMP_BUILD_LIB)
#  define _CCCL_FPMP_CORE_API        static _CCCL_TRIVIAL_API
#  define _CCCL_FPMP_CORE_DEVICE_API static _CCCL_DEVICE_API
#else
#  define _CCCL_FPMP_CORE_API        _CCCL_TRIVIAL_API
#  define _CCCL_FPMP_CORE_DEVICE_API _CCCL_DEVICE_API
#endif

/*
// Optional function qualifiers for portable API annotation.
// _CCCL_FPMP_CONSTEXPR can be used only on functions whose bodies are valid
// constant-evaluation code across all supported toolchains.
*/
#ifndef _CCCL_FPMP_CONSTEXPR
#  define _CCCL_FPMP_CONSTEXPR constexpr
#endif

#ifndef _CCCL_FPMP_NOEXCEPT
#  define _CCCL_FPMP_NOEXCEPT noexcept
#endif

/*
// fp32mp2 large-argument trig: fallback to system fp64 sin/cos (1) or use dedicated Payne-Hanek reduction (0)
*/
#ifndef _CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK
#  define _CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK 0
#endif

/*
// C++20 is_constant_evaluated() compatibility.
// NVCC, GCC, and Clang provide __builtin_is_constant_evaluated() which works
// in __host__ __device__ context without warnings.  std::is_constant_evaluated()
// is a __host__-only constexpr function under NVCC, triggering warning #20015-D.
// Fall back to std:: only for compilers that lack the built-in (e.g., MSVC).
*/
#if defined(__CUDACC__) || defined(__GNUC__) || defined(__clang__)
#  define _CCCL_FPMP_IS_CONSTEVAL() __builtin_is_constant_evaluated()
#else
#  define _CCCL_FPMP_IS_CONSTEVAL() std::is_constant_evaluated()
#endif

/*
// Internal macro for inline assembly support
// for reciprocal and reciprocal square root operations when available (CUDA)
// When _CCCL_FPMP_USE_INLINE_ASM_RSQRT is 1, the inline assembly is used
// When _CCCL_FPMP_USE_INLINE_ASM_RSQRT is 0, the inline assembly is not used
// When _CCCL_FPMP_USE_INLINE_ASM_RCP is 1, the inline assembly is used
// When _CCCL_FPMP_USE_INLINE_ASM_RCP is 0, the inline assembly is not used
// The default is to use inline assembly
// This is the fastest option, but may cause accuracy loss in subtle domains
// close to denormals or large numbers.
*/
#ifndef _CCCL_FPMP_USE_INLINE_ASM_RSQRT
#  define _CCCL_FPMP_USE_INLINE_ASM_RSQRT 1
#endif
#ifndef _CCCL_FPMP_USE_INLINE_ASM_RCP
#  define _CCCL_FPMP_USE_INLINE_ASM_RCP 1
#endif
/*
// _CCCL_FPMP_USE_INLINE_ASM_EX2_LG2 controls the implementation of the single-precision
// fast exp2 / log2 helpers used by the fp32mp2 transcendental kernels (cbrt, ...).
// When 1 (default on CUDA): emit ex2.approx.ftz.f32 / lg2.approx.ftz.f32 inline asm.
// When 0: fall back to the __exp2f / __log2f device intrinsics.
*/
#ifndef _CCCL_FPMP_USE_INLINE_ASM_EX2_LG2
#  define _CCCL_FPMP_USE_INLINE_ASM_EX2_LG2 1
#endif
/*
// Internal macro for accurate multiplication & division support
// When _CCCL_FPMP_USE_ACCURATE_MUL is 1, the accurate multiplication is used
// When _CCCL_FPMP_USE_ACCURATE_MUL is 0, the accurate multiplication is not used
// When _CCCL_FPMP_USE_ACCURATE_DIV is 1, the accurate division is used
// When _CCCL_FPMP_USE_ACCURATE_DIV is 0, the accurate division is not used
// Defaults: accurate multiplication is OFF, accurate division is ON. The
// accurate division routes fpmp2_accuracy::high through __fpmp2_high_div, whose
// branch-free exponent scaling keeps the reciprocal in range at BOTH ends of
// the exponent axis (small operands near denormal AND large divisors whose
// reciprocal would otherwise underflow to a denormal and be flushed to 0 by
// FTZ). It costs about 1.5x over the plain Nagai division; def/low are unaffected.
*/
#ifndef _CCCL_FPMP_USE_ACCURATE_MUL
#  define _CCCL_FPMP_USE_ACCURATE_MUL 0
#endif
#ifndef _CCCL_FPMP_USE_ACCURATE_DIV
#  define _CCCL_FPMP_USE_ACCURATE_DIV 1
#endif

/*********************************************************************
 * Internal utilities
 *********************************************************************/
// NOTE: the public fpmp2_accuracy enum lives in <cuda/__fp/fpmp_common.h>
// (the public API header), which is included at the top of this file.

/*
// Which IEEE-754 format an fpmp2 element type is: binary32 (double-float) or binary64
// (double-double), asked in one place instead of restated at every dispatch site.
//
// Exact type identity, not a format comparison against ::cuda::std::__fp_format_of_v:
// the math layer specializes on exactly float / double (__fpmp2_sin<double> and its
// siblings) with the fp32 implementation as the primary template, so any other type
// sharing a format is silently routed into the fp32 kernels. That excludes long double,
// which is binary64 wherever LDBL_MANT_DIG == 53 (MSVC, some AArch64 ABIs), and
// std::float32_t / std::float64_t, which are distinct types from float / double.
*/
template <typename _Tp>
inline constexpr bool __fpmp2_is_fp32_v = ::cuda::std::is_same_v<_Tp, float>;

template <typename _Tp>
inline constexpr bool __fpmp2_is_fp64_v = ::cuda::std::is_same_v<_Tp, double>;

// Element types accepted by fpmp2: exactly the two formats above.
template <typename _Tp>
inline constexpr bool __fpmp2_is_supported_fp_v = __fpmp2_is_fp32_v<_Tp> || __fpmp2_is_fp64_v<_Tp>;

/*
// Internal basic arith operations
// dispatched to the appropriate built-in for host and device
// if not available, use the appropriate fallback
// the fallback is the appropriate arithmetic operation
*/
#ifdef __CUDA_ARCH__
_CCCL_TRIVIAL_API float __fpmp_internal_fabs(float __x) noexcept
{
  return fabsf(__x);
}
_CCCL_TRIVIAL_API bool __fpmp_internal_isnan(float __x) noexcept
{
  return ::isnan(__x);
}
_CCCL_TRIVIAL_API float __fpmp_add_rn(float __x, float __y) noexcept
{
  return __fadd_rn(__x, __y);
}
_CCCL_TRIVIAL_API float __fpmp_add_rz(float __x, float __y) noexcept
{
  return __fadd_rz(__x, __y);
}
_CCCL_TRIVIAL_API float __fpmp_sub_rn(float __x, float __y) noexcept
{
  return __fsub_rn(__x, __y);
}
_CCCL_TRIVIAL_API float __fpmp_mul_rn(float __x, float __y) noexcept
{
  return __fmul_rn(__x, __y);
}
_CCCL_TRIVIAL_API float __fpmp_fma_rn(float __x, float __y, float __z) noexcept
{
  return __fmaf_ieee_rn(__x, __y, __z);
}
#  if _CCCL_FPMP_USE_INLINE_ASM_RCP == 1
_CCCL_TRIVIAL_API float __fpmp_rcp_rn(float __x) noexcept
{
  float __r;
  asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(__r) : "f"(__x));
  return __r;
}
#  else
_CCCL_TRIVIAL_API float __fpmp_rcp_rn(float __x) noexcept
{
  return __frcp_rn(__x);
}
#  endif
#  if _CCCL_FPMP_USE_INLINE_ASM_RSQRT == 1
_CCCL_TRIVIAL_API float __fpmp_rsqrt_rn(float __x) noexcept
{
  float __r;
  asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(__r) : "f"(__x));
  return __r;
}
#  else
_CCCL_TRIVIAL_API float __fpmp_rsqrt_rn(float __x) noexcept
{
  return __frsqrt_rn(__x);
}
#  endif
// Fast single-precision base-2 exp / log mapped to the FP32 SFU
// approximation units (ex2.approx / lg2.approx). These are not
// correctly rounded; they are used as initial estimates for
// higher-precision Newton/Halley refinement.
#  if _CCCL_FPMP_USE_INLINE_ASM_EX2_LG2 == 1
_CCCL_TRIVIAL_API float __fpmp_fast_exp2(float __x) noexcept
{
  float __r;
  asm("ex2.approx.ftz.f32 %0,%1;" : "=f"(__r) : "f"(__x));
  return __r;
}
_CCCL_TRIVIAL_API float __fpmp_fast_log2(float __x) noexcept
{
  float __r;
  asm("lg2.approx.ftz.f32 %0,%1;" : "=f"(__r) : "f"(__x));
  return __r;
}
#  else
_CCCL_TRIVIAL_API float __fpmp_fast_exp2(float __x) noexcept
{
  return __exp2f(__x);
}
_CCCL_TRIVIAL_API float __fpmp_fast_log2(float __x) noexcept
{
  return __log2f(__x);
}
#  endif
_CCCL_TRIVIAL_API int32_t __fpmp_fp2int_rz(float __x) noexcept
{
  return __float2int_rz(__x);
}
_CCCL_TRIVIAL_API int32_t __fpmp_fp2int_rn(float __x) noexcept
{
  return __float2int_rn(__x);
}
_CCCL_TRIVIAL_API uint32_t __fpmp_fp2uint_rz(float __x) noexcept
{
  return __float2uint_rz(__x);
}
_CCCL_TRIVIAL_API int64_t __fpmp_fp2ll_rz(float __x) noexcept
{
  return __float2ll_rz(__x);
}
_CCCL_TRIVIAL_API uint64_t __fpmp_fp2ull_rz(float __x) noexcept
{
  return __float2ull_rz(__x);
}

template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_int2fp_rn(int32_t __x) noexcept
{
  return static_cast<_FpType>(__int2float_rn(__x));
}
template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_int2fp_rz(int32_t __x) noexcept
{
  return static_cast<_FpType>(__int2float_rz(__x));
}
template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_uint2fp_rz(uint32_t __x) noexcept
{
  return static_cast<_FpType>(__uint2float_rz(__x));
}
template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_ll2fp_rz(int64_t __x) noexcept
{
  return static_cast<_FpType>(__ll2float_rz(__x));
}
template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_ull2fp_rz(uint64_t __x) noexcept
{
  return static_cast<_FpType>(__ull2float_rz(__x));
}
#else // !__CUDA_ARCH__
_CCCL_TRIVIAL_API float __fpmp_internal_fabs(float __x) noexcept
{
  return fabsf(__x);
}
_CCCL_TRIVIAL_API bool __fpmp_internal_isnan(float __x) noexcept
{
  return std::isnan(__x);
}
_CCCL_TRIVIAL_API float __fpmp_add_rn(float __x, float __y) noexcept
{
  return __x + __y;
}
_CCCL_TRIVIAL_API float __fpmp_add_rz(float __x, float __y) noexcept
{
  float __sum = __x + __y;
  if (__sum == 0.0f)
  {
    return __sum;
  }
  float __error = fmaf(-1.0f, __sum, __x) + __y;
  if (__error == 0.0f)
  {
    return __sum;
  }
  if ((__sum > 0.0f && __error < 0.0f) || (__sum < 0.0f && __error > 0.0f))
  {
    // Rounded away from zero - need to adjust mantissa toward zero
    uint32_t __bits = ::cuda::std::bit_cast<uint32_t>(__sum);
    // Decrement mantissa (moves toward zero for both positive and negative)
    __bits--;
    __sum = ::cuda::std::bit_cast<float>(__bits);
  }
  return __sum;
}
_CCCL_TRIVIAL_API float __fpmp_sub_rn(float __x, float __y) noexcept
{
  return __x - __y;
}
_CCCL_TRIVIAL_API float __fpmp_mul_rn(float __x, float __y) noexcept
{
  return __x * __y;
}
_CCCL_TRIVIAL_API float __fpmp_fma_rn(float __x, float __y, float __z) noexcept
{
  return fmaf(__x, __y, __z);
}
_CCCL_TRIVIAL_API float __fpmp_rcp_rn(float __x) noexcept
{
  return 1.0f / __x;
}
_CCCL_TRIVIAL_API float __fpmp_rsqrt_rn(float __x) noexcept
{
  return 1.0f / sqrtf(__x);
}
// Host fallback for the fast SFU-style exp2 / log2; uses the libm
// single-precision routines.  Same use case as the device path:
// a low-cost initial estimate for Newton/Halley refinement.
_CCCL_TRIVIAL_API float __fpmp_fast_exp2(float __x) noexcept
{
  return ::exp2f(__x);
}
_CCCL_TRIVIAL_API float __fpmp_fast_log2(float __x) noexcept
{
  return ::log2f(__x);
}
_CCCL_TRIVIAL_API int32_t __fpmp_fp2int_rz(float __x) noexcept
{
  return static_cast<int32_t>(__x);
}
_CCCL_TRIVIAL_API int32_t __fpmp_fp2int_rn(float __x) noexcept
{
  return static_cast<int32_t>(roundf(__x));
}
_CCCL_TRIVIAL_API uint32_t __fpmp_fp2uint_rz(float __x) noexcept
{
  return static_cast<uint32_t>(__x);
}
_CCCL_TRIVIAL_API int64_t __fpmp_fp2ll_rz(float __x) noexcept
{
  return static_cast<int64_t>(__x);
}
_CCCL_TRIVIAL_API uint64_t __fpmp_fp2ull_rz(float __x) noexcept
{
  return static_cast<uint64_t>(__x);
}

template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_int2fp_rn(int32_t __x) noexcept
{
  return static_cast<_FpType>(roundf(__x));
}

/*
// Round-toward-zero (truncation) versions for integer constructors
// For CPU: implement round-to-zero by checking if round-to-nearest went away from zero,
// then use nextafter to get the next representable float toward zero
// Using double for exact comparison (double has 53 bits, enough for int32_t's 32 bits)
// Template versions for both float and double
*/
template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_int2fp_rz(int32_t __x) noexcept
{
  _FpType __f    = static_cast<_FpType>(__x);
  double __exact = static_cast<double>(__x);
  if ((__x > 0 && __f > __exact) || (__x < 0 && __f < __exact))
  {
    __f = __fpmp2_is_fp32_v<_FpType> ? nextafterf(__f, 0.0f) : nextafter(__f, 0.0);
  }
  return __f;
}
template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_uint2fp_rz(uint32_t __x) noexcept
{
  _FpType __f    = static_cast<_FpType>(__x);
  double __exact = static_cast<double>(__x);
  if (__f > __exact)
  {
    __f = __fpmp2_is_fp32_v<_FpType> ? nextafterf(__f, 0.0f) : nextafter(__f, 0.0);
  }
  return __f;
}
template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_ll2fp_rz(int64_t __x) noexcept
{
  _FpType __f    = static_cast<_FpType>(__x);
  double __exact = static_cast<double>(__x);
  if ((__x > 0 && __f > __exact) || (__x < 0 && __f < __exact))
  {
    __f = __fpmp2_is_fp32_v<_FpType> ? nextafterf(__f, 0.0f) : nextafter(__f, 0.0);
  }
  return __f;
}
template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_ull2fp_rz(uint64_t __x) noexcept
{
  _FpType __f    = static_cast<_FpType>(__x);
  double __exact = static_cast<double>(__x);
  if (__f > __exact)
  {
    __f = __fpmp2_is_fp32_v<_FpType> ? nextafterf(__f, 0.0f) : nextafter(__f, 0.0);
  }
  return __f;
}
#endif // __CUDA_ARCH__

#ifdef __CUDA_ARCH__
_CCCL_TRIVIAL_API double __fpmp_internal_fabs(double __x) noexcept
{
  return ::fabs(__x);
}
_CCCL_TRIVIAL_API bool __fpmp_internal_isnan(double __x) noexcept
{
  return ::isnan(__x);
}
_CCCL_TRIVIAL_API double __fpmp_add_rn(double __x, double __y) noexcept
{
  return __dadd_rn(__x, __y);
}
_CCCL_TRIVIAL_API double __fpmp_add_rz(double __x, double __y) noexcept
{
  return __dadd_rz(__x, __y);
}
_CCCL_TRIVIAL_API double __fpmp_sub_rn(double __x, double __y) noexcept
{
  return __dsub_rn(__x, __y);
}
_CCCL_TRIVIAL_API double __fpmp_mul_rn(double __x, double __y) noexcept
{
  return __dmul_rn(__x, __y);
}
_CCCL_TRIVIAL_API double __fpmp_fma_rn(double __x, double __y, double __z) noexcept
{
  return __fma_rn(__x, __y, __z);
}
_CCCL_TRIVIAL_API double __fpmp_rcp_rn(double __x) noexcept
{
  return __drcp_rn(__x);
}
_CCCL_TRIVIAL_API double __fpmp_rsqrt_rn(double __x) noexcept
{
  return rsqrt(__x);
}
_CCCL_TRIVIAL_API int32_t __fpmp_fp2int_rz(double __x) noexcept
{
  return __double2int_rz(__x);
}
_CCCL_TRIVIAL_API int32_t __fpmp_fp2int_rn(double __x) noexcept
{
  return __double2int_rn(__x);
}
_CCCL_TRIVIAL_API uint32_t __fpmp_fp2uint_rz(double __x) noexcept
{
  return __double2uint_rz(__x);
}
_CCCL_TRIVIAL_API int64_t __fpmp_fp2ll_rz(double __x) noexcept
{
  return __double2ll_rz(__x);
}
_CCCL_TRIVIAL_API uint64_t __fpmp_fp2ull_rz(double __x) noexcept
{
  return __double2ull_rz(__x);
}
// int32_t and uint32_t always fit exactly in double (52-bit mantissa vs 32-bit values)
template <>
_CCCL_API inline double __fpmp_int2fp_rn<double>(int32_t __x) noexcept
{
  return __int2double_rn(__x);
}
template <>
_CCCL_API inline double __fpmp_int2fp_rz<double>(int32_t __x) noexcept
{
  return static_cast<double>(__x);
}
template <>
_CCCL_API inline double __fpmp_uint2fp_rz<double>(uint32_t __x) noexcept
{
  return static_cast<double>(__x);
}
// int64_t and uint64_t: use CUDA intrinsics for round-toward-zero
template <>
_CCCL_API inline double __fpmp_ll2fp_rz<double>(int64_t __x) noexcept
{
  return __ll2double_rz(__x);
}
template <>
_CCCL_API inline double __fpmp_ull2fp_rz<double>(uint64_t __x) noexcept
{
  return __ull2double_rz(__x);
}
#else // !__CUDA_ARCH__
_CCCL_TRIVIAL_API double __fpmp_internal_fabs(double __x) noexcept
{
  return ::fabs(__x);
}
_CCCL_TRIVIAL_API bool __fpmp_internal_isnan(double __x) noexcept
{
  return std::isnan(__x);
}
_CCCL_TRIVIAL_API double __fpmp_add_rn(double __x, double __y) noexcept
{
  return __x + __y;
}
_CCCL_TRIVIAL_API double __fpmp_add_rz(double __x, double __y) noexcept
{
  double __sum = __x + __y;
  if (__sum == 0.0)
  {
    return __sum;
  }
  double __error = fma(-1.0, __sum, __x) + __y;
  if (__error == 0.0)
  {
    return __sum;
  }
  if ((__sum > 0.0 && __error < 0.0) || (__sum < 0.0 && __error > 0.0))
  {
    // Rounded away from zero - need to adjust mantissa toward zero
    uint64_t __bits = ::cuda::std::bit_cast<uint64_t>(__sum);
    // Decrement mantissa (moves toward zero for both positive and negative)
    __bits--;
    __sum = ::cuda::std::bit_cast<double>(__bits);
  }
  return __sum;
}
_CCCL_TRIVIAL_API double __fpmp_sub_rn(double __x, double __y) noexcept
{
  return __x - __y;
}
_CCCL_TRIVIAL_API double __fpmp_mul_rn(double __x, double __y) noexcept
{
  return __x * __y;
}
_CCCL_TRIVIAL_API double __fpmp_fma_rn(double __x, double __y, double __z) noexcept
{
  return fma(__x, __y, __z);
}
_CCCL_TRIVIAL_API double __fpmp_rcp_rn(double __x) noexcept
{
  return 1.0 / __x;
}
_CCCL_TRIVIAL_API double __fpmp_rsqrt_rn(double __x) noexcept
{
  return 1.0 / sqrt(__x);
}
_CCCL_TRIVIAL_API int32_t __fpmp_fp2int_rz(double __x) noexcept
{
  return static_cast<int32_t>(__x);
}
_CCCL_TRIVIAL_API int32_t __fpmp_fp2int_rn(double __x) noexcept
{
  return static_cast<int32_t>(round(__x));
}
_CCCL_TRIVIAL_API uint32_t __fpmp_fp2uint_rz(double __x) noexcept
{
  return static_cast<uint32_t>(__x);
}
_CCCL_TRIVIAL_API int64_t __fpmp_fp2ll_rz(double __x) noexcept
{
  return static_cast<int64_t>(__x);
}
_CCCL_TRIVIAL_API uint64_t __fpmp_fp2ull_rz(double __x) noexcept
{
  return static_cast<uint64_t>(__x);
}
/*
// Round-toward-zero (truncation) versions for integer-to-double constructors
// For double, we need to use long double for exact comparison where possible
// Template specializations for double type
*/
template <>
_CCCL_API inline double __fpmp_int2fp_rn<double>(int32_t __x) noexcept
{
  return round(__x);
}
template <>
_CCCL_API inline double __fpmp_int2fp_rz<double>(int32_t __x) noexcept
{
  return static_cast<double>(__x);
}
template <>
_CCCL_API inline double __fpmp_uint2fp_rz<double>(uint32_t __x) noexcept
{
  return static_cast<double>(__x);
}
template <>
_CCCL_API inline double __fpmp_ll2fp_rz<double>(int64_t __x) noexcept
{
  // int64_t may not fit exactly in double
  double __d          = static_cast<double>(__x);
  long double __exact = static_cast<long double>(__x);
  if ((__x > 0 && __d > __exact) || (__x < 0 && __d < __exact))
  {
    __d = nextafter(__d, 0.0);
  }
  return __d;
}
template <>
_CCCL_API inline double __fpmp_ull2fp_rz<double>(uint64_t __x) noexcept
{
  // uint64_t may not fit exactly in double
  double __d          = static_cast<double>(__x);
  long double __exact = static_cast<long double>(__x);
  if (__d > __exact)
  {
    __d = nextafter(__d, 0.0);
  }
  return __d;
}
#endif // __CUDA_ARCH__

/*
// Scalar rounding helpers (host + device)
// Intentionally __fpmp_-prefixed and used by fpmp_math.h
// dedicated fp32mp2 rounding implementations.
*/
template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_internal_trunc(const _FpType __x) noexcept
{
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    const _FpType __abs_x = __fpmp_internal_fabs(__x);
    if (__abs_x >= _FpType(0x1.0p23f))
    {
      return __x;
    }
#if defined(__CUDA_ARCH__)
    const int32_t __xi = __float2int_rz(__x);
    return __int2float_rz(__xi);
#else
    const int32_t __xi = __fpmp_fp2int_rz(__x);
    return __fpmp_int2fp_rz<_FpType>(__xi);
#endif
  }
  else
  {
    return static_cast<_FpType>(::trunc(static_cast<double>(__x)));
  }
}

template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_internal_floor(const _FpType __x) noexcept
{
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    const _FpType __abs_x = __fpmp_internal_fabs(__x);
    if (__abs_x >= _FpType(0x1.0p23f))
    {
      return __x;
    }
#if defined(__CUDA_ARCH__)
    const int32_t __xi = __float2int_rd(__x);
    return __int2float_rn(__xi);
#else
    return floorf(__x);
#endif
  }
  else
  {
    return static_cast<_FpType>(::floor(static_cast<double>(__x)));
  }
}

template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_internal_ceil(const _FpType __x) noexcept
{
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    const _FpType __abs_x = __fpmp_internal_fabs(__x);
    if (__abs_x >= _FpType(0x1.0p23f))
    {
      return __x;
    }
#if defined(__CUDA_ARCH__)
    const int32_t __xi = __float2int_ru(__x);
    return __int2float_rn(__xi);
#else
    return ceilf(__x);
#endif
  }
  else
  {
    return static_cast<_FpType>(::ceil(static_cast<double>(__x)));
  }
}

/*
// Internal operations for 2-precision arithmetic
*/
// Multiply 2 floats exactly, assuming no over/underflow.
template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_two_mult_fma(const _FpType __x, const _FpType __y, _FpType* const __res_lo) noexcept
{
  _FpType __res_hi = __fpmp_mul_rn(__x, __y);
  *__res_lo        = __fpmp_fma_rn(__x, __y, -__res_hi);
  return __res_hi;
}

// Add 2 floats, returning the answer exactly in 'hi' and 'lo' parts.
// Assumes the exponent of 'x' is >= exponent of 'y'.
// (Usually we just check if |x| >= |y|).
// If this is not known use the function below.
template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_fast_two_sum(const _FpType __x, const _FpType __y, _FpType* const __res_lo) noexcept
{
  _FpType __res_hi = __fpmp_add_rn(__x, __y);
  _FpType __diff   = __fpmp_sub_rn(__res_hi, __x);
  *__res_lo        = __fpmp_sub_rn(__y, __diff);
  return __res_hi;
}

// Add 2 floats, returning the answer exactly in 'hi' and 'lo' parts.
// This makes no assumptions on the magnitudes of |x| and |y|.
template <typename _FpType = float>
_CCCL_TRIVIAL_API _FpType __fpmp_two_sum(const _FpType __x, const _FpType __y, _FpType* const __res_lo) noexcept
{
  _FpType __res_hi  = __fpmp_add_rn(__x, __y);
  _FpType __a_prime = __fpmp_sub_rn(__res_hi, __y);
  _FpType __b_prime = __fpmp_sub_rn(__res_hi, __a_prime);
  _FpType __delta_a = __fpmp_sub_rn(__x, __a_prime);
  _FpType __delta_b = __fpmp_sub_rn(__y, __b_prime);
  *__res_lo         = __fpmp_add_rn(__delta_a, __delta_b);
  return __res_hi;
}

// double -> (hi, lo) conversions (plain versions)
// only for the C++ class below to be optimized in compile-time
_CCCL_API _CCCL_FPMP_CONSTEXPR void
__fpmp_from_double(const double __x, float* __res_hi, float* __res_lo) _CCCL_FPMP_NOEXCEPT
{
  *__res_hi = (float) __x;
  *__res_lo = (float) (__x - (double) (float) __x);
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

// ---------------------------------------------------------------------------
// This header is the shared base for the per-operation implementation families
// (fpmp_impl_{cvt,muladd,divsqrt,cmp,atomic}.h): the internal macros, the fp128
// vocabulary type, and the __fpmp_* primitives above. It deliberately does NOT
// include the family headers -- that would create a self-referential cycle
// (family -> base -> family) that breaks standalone (internal_headers)
// compilation. Instead each family includes this base plus the specific sibling
// families it depends on, and the aggregators <cuda/__fp/fpmp.h> and
// <cuda/__fp/fpmp_lib.h> pull in the full family set. (Mirrors the fpemu layout,
// where fpemu.h aggregates the fpemu_impl_*.h families.)
//
// NOTE: the freestanding fpmp2 atomics (atomicAdd/atomicSub) and warp-shuffle
// helpers live in <cuda/__fp/fpmp.h>, after the fpmp2 class definition, since
// they are public class-dependent free functions rather than internal impl.
// ---------------------------------------------------------------------------

#endif // _CUDA___FP_FPMP_IMPL_H
