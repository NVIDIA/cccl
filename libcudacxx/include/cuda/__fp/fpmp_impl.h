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
        * Multiplication: __fpmp2_mul, __fpmp2_low_mul
        * Division: __fpmp2_div, __fpmp2_low_div, __fpmp2_high_div
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
#include <cuda/std/__cccl/preprocessor.h> // _CCCL_PP_FOR_EACH, to fold __CUDA_ARCH_LIST__
#include <cuda/std/__concepts/concept_macros.h>
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

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// ===================================================================
// Internal implementation vocabulary and helpers (moved from fpmp_common.h;
// fpmp_common.h now carries only the public API surface).
// ===================================================================
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
// The architecture is necessary but not sufficient, hence the __has_include: <quadmath.h>
// lives in GCC's own include directory, which clang does not search even when GCC is
// installed alongside it, so an x86_64 clang host generally cannot reach the header. Such
// hosts fall back to binary64 for the fp64mp2 math functions; to give them the quad path,
// put the header on the include path and build with -D_CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH=1
// (-D...=0 forces it off when the detection is wrong the other way).
*/
#ifndef _CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH
#  if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && !defined(_MSC_VER) \
    && !defined(_WIN32) && __has_include(<quadmath.h>)
#    define _CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH 1
#  else
#    define _CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH 0
#  endif
#endif

/*
// Availability of the __float128 type
// ------------------------------------
// fpmp needs the __float128 *type* only: the quad conversions and the reference math take
// and return values, they never write q/Q literals.
//
// This is therefore deliberately broader than _CCCL_HAS_FLOAT128(), which also requires the
// literal suffixes and so reports 0 in cases where the type is perfectly usable:
//   - GCC under __STRICT_ANSI__, i.e. every -std=c++NN build, unless the user passes
//     -fext-numeric-literals and defines CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS
//   - nvcc device passes below sm_100, even where the toolchain does provide device fp128
// Both are configurations fpmp supports, so the test below asks the compiler directly
// whether it provides the type. ARM64 stays excluded, as in CCCL: no ARM64 toolchain
// provides __float128 (aarch64 GCC does not even define __SIZEOF_FLOAT128__), and nvc++
// there rejects the name outright - such hosts take the 128-bit long double path below.
*/
#ifndef _CCCL_FPMP_HAS_FLOAT128_TYPE
#  if _CCCL_HAS_FLOAT128()
#    define _CCCL_FPMP_HAS_FLOAT128_TYPE 1
#  elif (defined(__SIZEOF_FLOAT128__) || defined(__FLOAT128__) || defined(__CUDACC_RTC_FLOAT128__)) \
    && !_CCCL_HOST_ARCH(ARM64)
#    define _CCCL_FPMP_HAS_FLOAT128_TYPE 1
#  else
#    define _CCCL_FPMP_HAS_FLOAT128_TYPE 0
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
// Deliberately not gated on _CCCL_HAS_LONG_DOUBLE(): that macro is 0 for the whole of a
// CUDA compilation, host pass included, because long double is not a device type. The
// question here is what the *host* ABI makes long double, which the LDBL_* limits answer
// in either pass, and getting 0 for it under nvcc would leave ARM64 - where long double is
// the only 128-bit type - with no quad type at all.
*/
#ifndef _CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128
#  if LDBL_MIN_EXP == -16381 && LDBL_MAX_EXP == 16384 && LDBL_MANT_DIG == 113
#    define _CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128 1
#  else
#    define _CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128 0
#  endif
#endif

/*
// Automatic detection of FP128 support based on platform capabilities
// --------------------------------------------------------------------
// Two separate questions, deliberately answered by two macros:
//
//   _CCCL_FPMP_FP128_ENABLE     - does fpmp2 declare its fp128 constructor and conversion?
//   _CCCL_FPMP_FP128_DEVICE_OPS - may those members, and the quad math fallback, run on the
//                                 device?
//
// The first one must answer identically in the host pass and in every device pass of a CUDA
// compilation: it decides the *class definition*, and a class whose member set depends on
// the pass is an ODR violation between the two halves of the same .cu file. It is therefore
// derived only from host-toolchain properties - the type detection above, which nvcc reports
// the same way in both passes - and never from __CUDA_ARCH__. That is also what gives host
// code inside a .cu file the fp128 interchange it is entitled to, whatever the target
// architecture.
//
// The second one is the question __CUDA_ARCH__ used to be asked, answered instead from
// __CUDA_ARCH_LIST__, which nvcc exposes in both passes. It is all-or-nothing for the whole
// compilation: a fatbin that also targets an architecture below sm_100 keeps the fp128
// members host-only, because nvcc rejects a __host__ __device__ signature naming __float128
// there. Declaring it anyway and failing at link time - the way the sm_90 atomics do - is
// not available when the signature itself cannot be parsed.
//
// Only nvcc gets this treatment. Clang's NVPTX target has no 128-bit floating-point type at
// all and rejects the name even in a __host__-only declaration, while still defining
// __SIZEOF_FLOAT128__ in the device pass, so the detection above cannot see the difference;
// nvc++ is unverified. Both keep fp128 out of the class in both passes, as they did before.
// NVRTC compiles a single device pass, so it has no symmetry to preserve and usability
// remains its gate.
//
// A toolchain that provides device fp128 on earlier architectures can still opt in with
// -D_CCCL_FPMP_FP128_DEVICE_OPS=1, and either macro can be forced off.
*/
#ifndef _CCCL_FPMP_FP128_ENABLE
#  if _CCCL_COMPILER(NVRTC)
#    if (_CCCL_FPMP_HAS_FLOAT128_TYPE == 1) && (_CCCL_PTX_ARCH() >= 1000)
#      define _CCCL_FPMP_FP128_ENABLE 1
#    else
#      define _CCCL_FPMP_FP128_ENABLE 0
#    endif
#  elif _CCCL_CUDA_COMPILATION() && !_CCCL_CUDA_COMPILER(NVCC)
#    define _CCCL_FPMP_FP128_ENABLE 0
#  elif (_CCCL_FPMP_HAS_FLOAT128_TYPE == 1) || (_CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128 == 1)
#    define _CCCL_FPMP_FP128_ENABLE 1
#  else
#    define _CCCL_FPMP_FP128_ENABLE 0
#  endif
#endif

#ifndef _CCCL_FPMP_FP128_DEVICE_OPS
#  if (_CCCL_FPMP_FP128_ENABLE == 0) || !_CCCL_CUDA_COMPILATION()
#    define _CCCL_FPMP_FP128_DEVICE_OPS 0
#  elif _CCCL_COMPILER(NVRTC)
#    define _CCCL_FPMP_FP128_DEVICE_OPS 1
#  elif defined(__CUDA_ARCH_LIST__)
// Folds the list into one conjunction: 800,1000 becomes 1 &&(800 >= 1000) &&(1000 >= 1000).
#    define _CCCL_FPMP_FP128_ARCH_IS_SM100(_Arch) &&((_Arch) >= 1000)
#    if 1 _CCCL_PP_FOR_EACH(_CCCL_FPMP_FP128_ARCH_IS_SM100, __CUDA_ARCH_LIST__)
#      define _CCCL_FPMP_FP128_DEVICE_OPS 1
#    else
#      define _CCCL_FPMP_FP128_DEVICE_OPS 0
#    endif
#    undef _CCCL_FPMP_FP128_ARCH_IS_SM100
#  else
#    define _CCCL_FPMP_FP128_DEVICE_OPS 0
#  endif
#endif

// The third fp128 knob, _CCCL_FPMP_FP128_MATH_FALLBACK, selects the bodies of the fp64mp2
// math functions rather than anything in the class, so it lives with them in
// <cuda/__fp/fpmp_math_impl.h> and is derived from the two macros above.

/*
// Internal 128-bit floating-point type definition
// ------------------------------------------------
// __fpmp_fp128 is the library's internal quad-precision type, mapped to whichever 128-bit
// IEEE 754 type the platform provides:
//
//   - __float128 wherever the compiler offers it (see _CCCL_FPMP_HAS_FLOAT128_TYPE)
//   - long double on aarch64, s390x and PowerPC with IEEE long double
//
// Forcing _CCCL_FPMP_FP128_ENABLE=1 on a platform with neither is a hard error rather than
// a type that silently degrades to void, and the static_assert additionally rejects a
// 128-bit type that turns out not to be 128 bits wide.
//
// Only defined when _CCCL_FPMP_FP128_ENABLE == 1.
*/
#if (_CCCL_FPMP_FP128_ENABLE == 1)
#  if (_CCCL_FPMP_HAS_FLOAT128_TYPE == 1)
using __fpmp_fp128 = __float128;
#  elif (_CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128 == 1)
using __fpmp_fp128 = long double;
#  else
#    error "_CCCL_FPMP_FP128_ENABLE=1 but this platform provides no 128-bit floating-point type"
#  endif
static_assert(sizeof(__fpmp_fp128) == 16, "__fpmp_fp128 must be a 128-bit floating-point type");
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
// visibility macros directly (_CCCL_HOST_DEVICE_API inline, _CCCL_DEVICE_API inline,
// _CCCL_TRIVIAL_HOST_DEVICE_API, _CCCL_TRIVIAL_DEVICE_API); these expand correctly for both
// CUDA and host-only compilation. The only decorators that still need dedicated
// macros are the extern-"C" ABI symbols used when building or linking the
// standalone libcufp library.
*/
#if (defined _CCCL_FPMP_BUILD_LIB) || (defined _CCCL_FPMP_USE_LIB)
#  define _CCCL_FPMP_BUILTIN_DECL        _CCCL_FPMP_ABI extern "C" _CCCL_HOST_DEVICE
#  define _CCCL_FPMP_BUILTIN_DEVICE_DECL _CCCL_FPMP_ABI extern "C" _CCCL_DEVICE
#else
#  define _CCCL_FPMP_BUILTIN_DECL        _CCCL_TRIVIAL_HOST_DEVICE_API
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
// static in inline mode caps registers and hurts performance).
*/
#if defined(_CCCL_FPMP_BUILD_LIB)
#  define _CCCL_FPMP_CORE_API        static _CCCL_TRIVIAL_HOST_DEVICE_API
#  define _CCCL_FPMP_CORE_DEVICE_API static _CCCL_DEVICE_API
#else
#  define _CCCL_FPMP_CORE_API        _CCCL_TRIVIAL_HOST_DEVICE_API
#  define _CCCL_FPMP_CORE_DEVICE_API _CCCL_DEVICE_API
#endif

/*
// Execution space of the fp128 interchange entry points.
//
// The fp128 members and the conversions behind them are declared wherever the 128-bit type
// can be named (_CCCL_FPMP_FP128_ENABLE), which under nvcc includes every device pass, but
// they carry __device__ only where fp128 arithmetic is device-callable
// (_CCCL_FPMP_FP128_DEVICE_OPS). Below that, they are host functions that the device pass
// parses and discards, so both passes see the same class and device code that reaches for
// quad precision is diagnosed at the call site.
*/
#if (_CCCL_FPMP_FP128_DEVICE_OPS == 1) || !_CCCL_CUDA_COMPILATION()
#  define _CCCL_FPMP_FP128_API          _CCCL_HOST_DEVICE_API
#  define _CCCL_FPMP_FP128_CORE_API     _CCCL_FPMP_CORE_API
#  define _CCCL_FPMP_FP128_BUILTIN_DECL _CCCL_FPMP_BUILTIN_DECL
#else
#  define _CCCL_FPMP_FP128_API _CCCL_HOST_API
#  if defined(_CCCL_FPMP_BUILD_LIB)
#    define _CCCL_FPMP_FP128_CORE_API static _CCCL_TRIVIAL_HOST_API
#  else
#    define _CCCL_FPMP_FP128_CORE_API _CCCL_TRIVIAL_HOST_API
#  endif
#  if (defined _CCCL_FPMP_BUILD_LIB) || (defined _CCCL_FPMP_USE_LIB)
#    define _CCCL_FPMP_FP128_BUILTIN_DECL _CCCL_FPMP_ABI extern "C" _CCCL_HOST
#  else
#    define _CCCL_FPMP_FP128_BUILTIN_DECL _CCCL_TRIVIAL_HOST_API
#  endif
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
//
// The CUDA intrinsics are called as ::__dadd_rn etc. because these wrappers live in
// cuda::experimental, where <cuda/fpemu> declares same-named overloads for its own
// types: unqualified lookup would stop there and never reach the global scope.
*/
_CCCL_TRIVIAL_HOST_DEVICE_API float __fpmp_internal_fabs(float __x) noexcept
{
  return fabsf(__x);
}
_CCCL_TRIVIAL_HOST_DEVICE_API bool __fpmp_internal_isnan(float __x) noexcept
{
  return ::cuda::std::isnan(__x);
}
_CCCL_TRIVIAL_HOST_DEVICE_API float __fpmp_add_rn(float __x, float __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__fadd_rn(__x, __y);), (return __x + __y;))
}
// The host has no round-toward-zero add, so round to nearest and step one ULP toward
// zero when the exact residual shows the sum was rounded away from zero.
_CCCL_TRIVIAL_HOST_DEVICE_API float __fpmp_add_rz(float __x, float __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__fadd_rz(__x, __y);), ({
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
                        // Rounded away from zero: decrementing the mantissa moves the magnitude
                        // toward zero for both signs.
                        uint32_t __bits = ::cuda::std::bit_cast<uint32_t>(__sum);
                        __bits--;
                        __sum = ::cuda::std::bit_cast<float>(__bits);
                      }
                      return __sum;
                    }))
}
_CCCL_TRIVIAL_HOST_DEVICE_API float __fpmp_sub_rn(float __x, float __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__fsub_rn(__x, __y);), (return __x - __y;))
}
_CCCL_TRIVIAL_HOST_DEVICE_API float __fpmp_mul_rn(float __x, float __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__fmul_rn(__x, __y);), (return __x * __y;))
}
_CCCL_TRIVIAL_HOST_DEVICE_API float __fpmp_fma_rn(float __x, float __y, float __z) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__fmaf_ieee_rn(__x, __y, __z);), (return fmaf(__x, __y, __z);))
}
// On device the approximate SFU reciprocal / reciprocal square root are emitted as
// inline asm rather than through __frcp_rn / __frsqrt_rn: they are the fastest option
// and only ever feed Newton refinement, at the cost of accuracy in subtle domains
// close to denormals or large numbers. The host divides instead.
_CCCL_TRIVIAL_HOST_DEVICE_API float __fpmp_rcp_rn(float __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    ({
                      float __r;
                      asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(__r) : "f"(__x));
                      return __r;
                    }),
                    (return 1.0f / __x;))
}
_CCCL_TRIVIAL_HOST_DEVICE_API float __fpmp_rsqrt_rn(float __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    ({
                      float __r;
                      asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(__r) : "f"(__x));
                      return __r;
                    }),
                    (return 1.0f / sqrtf(__x);))
}
// Fast single-precision base-2 exp / log mapped to the FP32 SFU approximation units
// (ex2.approx / lg2.approx) on device and to the libm single-precision routines on the
// host. Neither is correctly rounded; both only serve as the initial estimate for
// higher-precision Newton/Halley refinement.
_CCCL_TRIVIAL_HOST_DEVICE_API float __fpmp_fast_exp2(float __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    ({
                      float __r;
                      asm("ex2.approx.ftz.f32 %0,%1;" : "=f"(__r) : "f"(__x));
                      return __r;
                    }),
                    (return ::exp2f(__x);))
}
_CCCL_TRIVIAL_HOST_DEVICE_API float __fpmp_fast_log2(float __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    ({
                      float __r;
                      asm("lg2.approx.ftz.f32 %0,%1;" : "=f"(__r) : "f"(__x));
                      return __r;
                    }),
                    (return ::log2f(__x);))
}
_CCCL_TRIVIAL_HOST_DEVICE_API int32_t __fpmp_fp2int_rz(float __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__float2int_rz(__x);), (return static_cast<int32_t>(__x);))
}
_CCCL_TRIVIAL_HOST_DEVICE_API int32_t __fpmp_fp2int_rn(float __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__float2int_rn(__x);), (return static_cast<int32_t>(roundf(__x));))
}
_CCCL_TRIVIAL_HOST_DEVICE_API uint32_t __fpmp_fp2uint_rz(float __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__float2uint_rz(__x);), (return static_cast<uint32_t>(__x);))
}
_CCCL_TRIVIAL_HOST_DEVICE_API int64_t __fpmp_fp2ll_rz(float __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__float2ll_rz(__x);), (return static_cast<int64_t>(__x);))
}
_CCCL_TRIVIAL_HOST_DEVICE_API uint64_t __fpmp_fp2ull_rz(float __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__float2ull_rz(__x);), (return static_cast<uint64_t>(__x);))
}

// The host cast is already the round-to-nearest conversion the name promises, so it needs
// no rounding call on top of it.
template <typename _FpType>
_CCCL_TRIVIAL_HOST_DEVICE_API _FpType __fpmp_int2fp_rn(int32_t __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return static_cast<_FpType>(::__int2float_rn(__x));), (return static_cast<_FpType>(__x);))
}

/*
// Round-toward-zero (truncation) versions for the integer constructors.
// The device has rz conversion intrinsics. The host rounds to nearest, then checks
// whether that went away from zero -- comparing against double, which holds int32_t
// exactly -- and if so steps one representable value back toward zero via nextafter.
// The templates serve both float and double.
*/
template <typename _FpType>
_CCCL_TRIVIAL_HOST_DEVICE_API _FpType __fpmp_int2fp_rz(int32_t __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return static_cast<_FpType>(::__int2float_rz(__x));), ({
                      _FpType __f    = static_cast<_FpType>(__x);
                      double __exact = static_cast<double>(__x);
                      if ((__x > 0 && __f > __exact) || (__x < 0 && __f < __exact))
                      {
                        if constexpr (__fpmp2_is_fp32_v<_FpType>)
                        {
                          __f = nextafterf(__f, 0.0f);
                        }
                        else
                        {
                          __f = nextafter(__f, 0.0);
                        }
                      }
                      return __f;
                    }))
}
template <typename _FpType>
_CCCL_TRIVIAL_HOST_DEVICE_API _FpType __fpmp_uint2fp_rz(uint32_t __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return static_cast<_FpType>(::__uint2float_rz(__x));), ({
                      _FpType __f    = static_cast<_FpType>(__x);
                      double __exact = static_cast<double>(__x);
                      if (__f > __exact)
                      {
                        if constexpr (__fpmp2_is_fp32_v<_FpType>)
                        {
                          __f = nextafterf(__f, 0.0f);
                        }
                        else
                        {
                          __f = nextafter(__f, 0.0);
                        }
                      }
                      return __f;
                    }))
}
template <typename _FpType>
_CCCL_TRIVIAL_HOST_DEVICE_API _FpType __fpmp_ll2fp_rz(int64_t __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return static_cast<_FpType>(::__ll2float_rz(__x));), ({
                      _FpType __f    = static_cast<_FpType>(__x);
                      double __exact = static_cast<double>(__x);
                      if ((__x > 0 && __f > __exact) || (__x < 0 && __f < __exact))
                      {
                        if constexpr (__fpmp2_is_fp32_v<_FpType>)
                        {
                          __f = nextafterf(__f, 0.0f);
                        }
                        else
                        {
                          __f = nextafter(__f, 0.0);
                        }
                      }
                      return __f;
                    }))
}
template <typename _FpType>
_CCCL_TRIVIAL_HOST_DEVICE_API _FpType __fpmp_ull2fp_rz(uint64_t __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return static_cast<_FpType>(::__ull2float_rz(__x));), ({
                      _FpType __f    = static_cast<_FpType>(__x);
                      double __exact = static_cast<double>(__x);
                      if (__f > __exact)
                      {
                        if constexpr (__fpmp2_is_fp32_v<_FpType>)
                        {
                          __f = nextafterf(__f, 0.0f);
                        }
                        else
                        {
                          __f = nextafter(__f, 0.0);
                        }
                      }
                      return __f;
                    }))
}
_CCCL_TRIVIAL_HOST_DEVICE_API double __fpmp_internal_fabs(double __x) noexcept
{
  return ::fabs(__x);
}
_CCCL_TRIVIAL_HOST_DEVICE_API bool __fpmp_internal_isnan(double __x) noexcept
{
  return ::cuda::std::isnan(__x);
}
_CCCL_TRIVIAL_HOST_DEVICE_API double __fpmp_add_rn(double __x, double __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__dadd_rn(__x, __y);), (return __x + __y;))
}
// As in the fp32 case, the host emulates round-toward-zero from the round-to-nearest
// sum plus the exact residual.
_CCCL_TRIVIAL_HOST_DEVICE_API double __fpmp_add_rz(double __x, double __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__dadd_rz(__x, __y);), ({
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
                        // Rounded away from zero: decrementing the mantissa moves the magnitude
                        // toward zero for both signs.
                        uint64_t __bits = ::cuda::std::bit_cast<uint64_t>(__sum);
                        __bits--;
                        __sum = ::cuda::std::bit_cast<double>(__bits);
                      }
                      return __sum;
                    }))
}
_CCCL_TRIVIAL_HOST_DEVICE_API double __fpmp_sub_rn(double __x, double __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__dsub_rn(__x, __y);), (return __x - __y;))
}
_CCCL_TRIVIAL_HOST_DEVICE_API double __fpmp_mul_rn(double __x, double __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__dmul_rn(__x, __y);), (return __x * __y;))
}
_CCCL_TRIVIAL_HOST_DEVICE_API double __fpmp_fma_rn(double __x, double __y, double __z) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__fma_rn(__x, __y, __z);), (return fma(__x, __y, __z);))
}
_CCCL_TRIVIAL_HOST_DEVICE_API double __fpmp_rcp_rn(double __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__drcp_rn(__x);), (return 1.0 / __x;))
}
_CCCL_TRIVIAL_HOST_DEVICE_API double __fpmp_rsqrt_rn(double __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return rsqrt(__x);), (return 1.0 / sqrt(__x);))
}
_CCCL_TRIVIAL_HOST_DEVICE_API int32_t __fpmp_fp2int_rz(double __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__double2int_rz(__x);), (return static_cast<int32_t>(__x);))
}
_CCCL_TRIVIAL_HOST_DEVICE_API int32_t __fpmp_fp2int_rn(double __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__double2int_rn(__x);), (return static_cast<int32_t>(round(__x));))
}
_CCCL_TRIVIAL_HOST_DEVICE_API uint32_t __fpmp_fp2uint_rz(double __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__double2uint_rz(__x);), (return static_cast<uint32_t>(__x);))
}
_CCCL_TRIVIAL_HOST_DEVICE_API int64_t __fpmp_fp2ll_rz(double __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__double2ll_rz(__x);), (return static_cast<int64_t>(__x);))
}
_CCCL_TRIVIAL_HOST_DEVICE_API uint64_t __fpmp_fp2ull_rz(double __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__double2ull_rz(__x);), (return static_cast<uint64_t>(__x);))
}
// int32_t and uint32_t always fit exactly in double (52-bit mantissa vs 32-bit values)
template <>
_CCCL_HOST_DEVICE_API inline double __fpmp_int2fp_rn<double>(int32_t __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__int2double_rn(__x);), (return static_cast<double>(__x);))
}
template <>
_CCCL_HOST_DEVICE_API inline double __fpmp_int2fp_rz<double>(int32_t __x) noexcept
{
  return static_cast<double>(__x);
}
template <>
_CCCL_HOST_DEVICE_API inline double __fpmp_uint2fp_rz<double>(uint32_t __x) noexcept
{
  return static_cast<double>(__x);
}
// int64_t and uint64_t may not fit exactly in double: the device has round-toward-zero
// conversion intrinsics, while the host detects a round away from zero by comparing
// against long double and steps one value back.
template <>
_CCCL_HOST_DEVICE_API inline double __fpmp_ll2fp_rz<double>(int64_t __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__ll2double_rz(__x);), ({
                      double __d          = static_cast<double>(__x);
                      long double __exact = static_cast<long double>(__x);
                      if ((__x > 0 && __d > __exact) || (__x < 0 && __d < __exact))
                      {
                        __d = nextafter(__d, 0.0);
                      }
                      return __d;
                    }))
}
template <>
_CCCL_HOST_DEVICE_API inline double __fpmp_ull2fp_rz<double>(uint64_t __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__ull2double_rz(__x);), ({
                      double __d          = static_cast<double>(__x);
                      long double __exact = static_cast<long double>(__x);
                      if (__d > __exact)
                      {
                        __d = nextafter(__d, 0.0);
                      }
                      return __d;
                    }))
}

/*
// Scalar rounding helpers (host + device)
// Intentionally __fpmp_-prefixed: they back the dedicated fp32mp2 rounding
// implementations rather than being part of the public surface.
*/
template <typename _FpType>
_CCCL_TRIVIAL_HOST_DEVICE_API _FpType __fpmp_internal_trunc(const _FpType __x) noexcept
{
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    const _FpType __abs_x = __fpmp_internal_fabs(__x);
    if (__abs_x >= _FpType(0x1.0p23f))
    {
      return __x;
    }
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      ({
        const int32_t __xi = ::__float2int_rz(__x);
        return ::__int2float_rz(__xi);
      }),
      ({
        const int32_t __xi = __fpmp_fp2int_rz(__x);
        return __fpmp_int2fp_rz<_FpType>(__xi);
      }))
  }
  else
  {
    return static_cast<_FpType>(::trunc(static_cast<double>(__x)));
  }
}

template <typename _FpType>
_CCCL_TRIVIAL_HOST_DEVICE_API _FpType __fpmp_internal_floor(const _FpType __x) noexcept
{
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    const _FpType __abs_x = __fpmp_internal_fabs(__x);
    if (__abs_x >= _FpType(0x1.0p23f))
    {
      return __x;
    }
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      ({
                        const int32_t __xi = ::__float2int_rd(__x);
                        return ::__int2float_rn(__xi);
                      }),
                      (return floorf(__x);))
  }
  else
  {
    return static_cast<_FpType>(::floor(static_cast<double>(__x)));
  }
}

template <typename _FpType>
_CCCL_TRIVIAL_HOST_DEVICE_API _FpType __fpmp_internal_ceil(const _FpType __x) noexcept
{
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    const _FpType __abs_x = __fpmp_internal_fabs(__x);
    if (__abs_x >= _FpType(0x1.0p23f))
    {
      return __x;
    }
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      ({
                        const int32_t __xi = ::__float2int_ru(__x);
                        return ::__int2float_rn(__xi);
                      }),
                      (return ceilf(__x);))
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
template <typename _FpType>
_CCCL_TRIVIAL_HOST_DEVICE_API _FpType
__fpmp_two_mult_fma(const _FpType __x, const _FpType __y, _FpType* const __res_lo) noexcept
{
  _FpType __res_hi = __fpmp_mul_rn(__x, __y);
  *__res_lo        = __fpmp_fma_rn(__x, __y, -__res_hi);
  return __res_hi;
}

// Add 2 floats, returning the answer exactly in 'hi' and 'lo' parts.
// Assumes the exponent of 'x' is >= exponent of 'y'.
// (Usually we just check if |x| >= |y|).
// If this is not known use the function below.
template <typename _FpType>
_CCCL_TRIVIAL_HOST_DEVICE_API _FpType
__fpmp_fast_two_sum(const _FpType __x, const _FpType __y, _FpType* const __res_lo) noexcept
{
  _FpType __res_hi = __fpmp_add_rn(__x, __y);
  _FpType __diff   = __fpmp_sub_rn(__res_hi, __x);
  *__res_lo        = __fpmp_sub_rn(__y, __diff);
  return __res_hi;
}

// Add 2 floats, returning the answer exactly in 'hi' and 'lo' parts.
// This makes no assumptions on the magnitudes of |x| and |y|.
template <typename _FpType>
_CCCL_TRIVIAL_HOST_DEVICE_API _FpType
__fpmp_two_sum(const _FpType __x, const _FpType __y, _FpType* const __res_lo) noexcept
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
_CCCL_HOST_DEVICE_API constexpr void __fpmp_from_double(const double __x, float* __res_hi, float* __res_lo) noexcept
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
// families it depends on, and the aggregator <cuda/__fp/fpmp.h> pulls in the full
// family set. (Mirrors the fpemu layout, where fpemu.h aggregates the
// fpemu_impl_*.h families.)
//
// NOTE: the freestanding fpmp2 atomics (atomicAdd/atomicSub) and warp-shuffle
// helpers live in <cuda/__fp/fpmp.h>, after the fpmp2 class definition, since
// they are public class-dependent free functions rather than internal impl.
// ---------------------------------------------------------------------------

#endif // _CUDA___FP_FPMP_IMPL_H
