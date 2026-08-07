//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_H
#define _CUDA___FP_FPMP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/*
    fpmp.h - Multi-Precision Floating-Point Types and Core Operations (double-float / double-double)
    ======================================================================================================
    This header defines the primary public types and core operations for multi-component floating-point
    arithmetic using pairs of IEEE-754 floating-point values. It supports both "double-float" (fp32mp2)
    and "double-double" (fp64mp2) representations and can be used from both CPU and GPU (CUDA) code.

    Linkage note:
    - In header mode (default, CCCL_FPMP_INLINE=1), built-in entry points are defined as inline/static.
    - In library mode (CCCL_FPMP_LIB=1), built-in entry points are provided by a
      separately compiled object/library, while the C++ API remains header-based.

    Supported Formats:
    -------------------------------------------------------------------------
    - Double-Float (fp32mp2): Pairs of single-precision floats (2×32-bit), providing up to ~46 bits of effective
   mantissa (algorithm/method dependent; between IEEE-754 float and double). Ideal for consumer GPUs where FP64
   performance is limited (often 1/32 of FP32 throughput).
    - Double-Double (fp64mp2): Pairs of double-precision floats (2×64-bit), providing up to ~104 bits of effective
   mantissa (algorithm/method dependent; between IEEE-754 double and binary128).

    Key Features:
    -------------------------------------------------------------------------
    - Data Types
        - fp32mp2 : Double-float type using two single-precision floats (hi, lo), representing a number as the
   unevaluated sum hi + lo.
        - fpmp2<FpType, fpmp2_accuracy> : C++ class template providing operator overloading and type safety.
          * FpType=float for double-float, FpType=double for double-double.

    - Operations Supported
        * Conversion: Safe conversion from/to double, float, int32_t, uint32_t, int64_t, uint64_t.
        * Normalization: Ensures the two-float representation maintains strict ordering and non-overlapping components.
        * Basic Arithmetic: Addition, subtraction, multiplication, negation, and division for double-float.
        * Accuracy-Explicit Arithmetic: Free functions add<m>, sub<m>, mul<m>, div<m>, fma<m>, mad<m>
          that override the arithmetic method for a single operation without changing the type.
        * Advanced: Square root, reciprocal square root, fused multiply-add (FMA), multiply-add (MAD).
        * Utility: Renormalization, error-aware summation and multiplication with or without FMA.
        * Comparison: Supports all common relational operators (==, !=, <, <=, >, >=).
        * Atomic Operations: Supports atomic addition and subtraction on multi-precision floating point numbers (CUDA
   only).
        * Warp Shuffle: Overloads of CUDA's __shfl_sync family for fpmp2 pairs (CUDA only, header-only,
          declared in this header; not exposed as library ABI symbols).
        * GPU & Host Compatibility: All operations and members are decorated for both device and host use.

    - Implementation Aspects
        * Header-based C++ API with optional library-provided built-in symbols (see linkage note above).
        * Multiple accuracy levels: mid (default, Dekker-based), low (minimal renormalization), and high (Thall-based).
        * Template-based design allows compile-time selection of arithmetic precision/speed trade-offs.
        * Error-free transformations using Dekker's 2Sum and 2Mult algorithms.

    - Usage Scenarios
        * Double-float (fp32mp2): Useful in numerical algorithms requiring more accuracy than float, without the full
   cost of FP64. Particularly suitable for GPUs with limited FP64 performance (consumer GPUs).
        * Double-double (fp64mp2): For applications requiring quad-like accuracy (extended precision beyond IEEE
   double). Ideal for scientific computing, high-precision simulations, and financial calculations. Suitable when
   expensive FP128 operations are not required and GPU .
        * Both formats are suitable for high-performance computing, GPGPU kernels requiring extended precision.
        * Applications needing reproducible results across different hardware platforms.

    Example Usage:
    -------------------------------------------------------------------------
        #include <cuda/fpmp>

        // Basic arithmetic with double-float precision
        fp32mp2 a = 1.23456789123456789;       // double-float precision from double value
        fp32mp2 b = 9.87654321987654321;       // double-float precision from double value
        auto sum = a + b;                        // High-precision addition
        auto product = a * b;                    // High-precision multiplication
        auto result  = fma(a, b, sum);           // Fused multiply-add: a*b + sum
        auto root = sqrt(a);                     // High-precision square root
        double d = static_cast<double>(result);  // Convert to double
        float f  = static_cast<float>(result);   // Convert to float (high part only)
        float hi = result.hi(), lo = result.lo(); // The two components, to inspect or store

        // Accuracy-explicit operations: override arithmetic accuracy for a single operation
        fp32mp2_low x = ..., y = ...;
        auto diff = sub<fpmp2_accuracy::high>(x, y); // Accurate subtraction, result stays fp32mp2_low

    Naming Convention:
    -------------------------------------------------------------------------
    - Built-in functions (C-style API):
        * __fpmp2_add, __fpmp2_sub, __fpmp2_mul, __fpmp2_div : Basic arithmetic operations
        * __fpmp2_acc, __fpmp2_low_acc, __fpmp2_high_acc : Optimized single-component accumulate
        * __fpmp2_low_add, __fpmp2_high_add : Method-specific variants
        * __fpmp2_fma, __fpmp2_mad : Fused multiply-add and multiply-add operations
        * __fpmp2_sqrt, __fpmp2_rsqrt : Square root and reciprocal square root
        * __fpmp2_from_double, __fpmp2_to_double : Type conversions
        * __fpmp2_cmp_eq, __fpmp2_cmp_lt, etc. : Comparison operations
        * __fpmp2_atomicAdd, __fpmp2_atomicSub : Atomic operations (CUDA only, slower than hardware atomics)
        * __shfl_sync, __shfl_xor_sync, __shfl_down_sync, __shfl_up_sync :
          Warp shuffle overloads for fpmp2 pairs (CUDA only, header-only).

    - C++ class template:
        * fpmp2<FpType, fpmp2_accuracy> : Template class with operator overloading
        * fp32mp2, fp32mp2_low, fp32mp2_high : Double-float type aliases
        * fp64mp2, fp64mp2_low, fp64mp2_high : Double-double type aliases

    - Accuracy-explicit free functions:
        * add<fpmp2_accuracy::m>(x, y) : Addition with explicit accuracy override
        * sub<fpmp2_accuracy::m>(x, y) : Subtraction with explicit accuracy override
        * mul<fpmp2_accuracy::m>(x, y) : Multiplication with explicit accuracy override
        * div<fpmp2_accuracy::m>(x, y) : Division with explicit accuracy override
        * fma<fpmp2_accuracy::m>(x, y, z) : Fused multiply-add with explicit accuracy override
        * mad<fpmp2_accuracy::m>(x, y, z) : Multiply-add with explicit accuracy override
        where m is one of: def, low, mid, high
        Each operation also has a mixed-type overload: any operand may be
        a built-in arithmetic scalar as long as at least one operand is
        fpmp2. The scalar is converted to the fpmp2 side's type.
        Example:  ffloat r = sub<fpmp2_accuracy::high>(a, 1.0f);

    Reference Papers:
    -------------------------------------------------------------------------
    [1] Dekker, T. (1971). A floating-point technique for extending the available precision. Numerische Mathematik, 18,
   224–242. [2] Karp, A. H., & Markstein, P. (1997). High Precision Division and Square Root. ACM Transactions on
   Mathematical Software, 23(4), 561–589. [3] Thall, Andrew. Extended-Precision Floating-Point Numbers for GPU
   Computation. (http://andrewthall.org/papers/df64_qf128.pdf) [4] Nagai et al. (2008). Fast Quadruple Precision
   Arithmetic Library on Parallel Computer SR11000/J2. ICCS '08.

    Configuration Macros:
    -------------------------------------------------------------------------
    - CCCL_FPMP_EXPLICIT_CASTS: When 1 (default), lossy/narrowing conversions INTO fpmp2 (e.g., double
      to fp32mp2, fp64mp2 to fp32mp2, and integer (int32/uint32/int64/uint64) to fpmp2) require explicit casts, matching
      CCCL's strict-cast conventions. The widening conversion OUT to double (operator double()) is
      always implicit and is not affected by this macro. Set to 0 to restore the fully-implicit model
      (all conversions implicit) for easier migration of existing code from standard types. Setting 0
      helps when fpmp2 is a near drop-in for double/float in a large codebase (existing call sites and
      mixed-type expressions compile unchanged instead of needing an explicit cast at every narrowing
      boundary) or for rapid prototyping. Warning: implicit casts let the compiler silently narrow
      INTO fpmp2, which can drop precision at unintended conversions or introduce accidental
      round-trips / FP64 use (accuracy/perf) with no diagnostic; keep the default 1 unless the
      migration benefit outweighs that risk.
    - CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP*: When 1 (default), double-to-fpmp2 conversion uses integer
      bit manipulation instead of FP64 casts. Avoids the slow FP64 pipeline on GPUs with limited
      double-precision throughput (e.g. consumer GPUs with a 1:64 ratio). When 0, uses standard casts.
    - CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE*: When 1 (default), fpmp2-to-double conversion reconstructs
      the double bit pattern using integer arithmetic (no FP64 ops). When 0, uses (double)hi + (double)lo.
    * Tuning the double<->fp32mp2 conversions: both macros above default to 1 because the integer
      path is a large win on FP64-throttled GPUs (measured several-x faster than the FP64 casts on
      an L40S) and applies to the FP32-based fp32mp2 (fp64mp2 conversions are inherently FP64 either
      way). Set either to 0 to fall back to the plain FP64 casts if you hit: (a) register pressure /
      reduced occupancy from the extra integer work in large kernels, or (b) a GPU with high FP64
      throughput (e.g. datacenter A100/H100, ~1:2) where the FP64 path is already cheap.
      e.g. -DCCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE=0 (and/or -DCCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP=0).

    The _CCCL_FPMP_* switches those knobs map to, and the fp128 switches the library detects on
    its own, are documented where they are defined: fpmp_impl.h for the ones that shape the class
    (_CCCL_FPMP_FP128_ENABLE, _CCCL_FPMP_FP128_DEVICE_OPS) and fpmp_math_impl.h for the one that
    only picks math bodies (_CCCL_FPMP_FP128_MATH_FALLBACK).

    Important Notes:
    -------------------------------------------------------------------------
    - The error-free transformations (e.g., 2Sum, 2Mult) are essential for guaranteeing that hi/lo have the
   non-overlapping property.
    - It is possible to mix and match routines according to desired accuracy, speed, and hardware FMA support.
    - The library requires C++11 minimum (for alignas, constexpr) and works best with C++17 or later (for if constexpr).
    - For CUDA code, the library requires CUDA Toolkit 11.0 or later.
    - All operations are fully inlined for optimal performance when not using library mode.
*/

// The public API surface (the fpmp2_accuracy selector and the CCCL_FPMP_LIB /
// CCCL_FPMP_INLINE / CCCL_FPMP_EXPLICIT_CASTS / CCCL_FPMP_OPTIMIZED_* knobs, plus
// their mapping to the internal switches) lives in fpmp_common.h; all
// library-internal machinery (decorator/ABI/declaration macros, the fp128
// plumbing, and the __fpmp_* helpers) lives in fpmp_impl.h.
#include <cuda/__fp/fpmp_common.h>
#include <cuda/__fp/fpmp_impl.h>
// Per-operation implementation families (see fpmp_impl.h). Aggregated here (not
// by the base) to avoid a family->base->family include cycle. Each family is
// self-contained and pulls the siblings it needs (divsqrt/atomic -> muladd).
#include <cuda/__fp/fpmp_impl_atomic.h>
#include <cuda/__fp/fpmp_impl_cmp.h>
#include <cuda/__fp/fpmp_impl_cvt.h>
#include <cuda/__fp/fpmp_impl_divsqrt.h>
#include <cuda/__fp/fpmp_impl_muladd.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// The public fpmp2_accuracy selector is defined in <cuda/__fp/fpmp_common.h>; the
// internal element-format predicates (__fpmp2_is_fp32_v / __fpmp2_is_fp64_v /
// __fpmp2_is_supported_fp_v) in <cuda/__fp/fpmp_impl.h>. Both are included above, so
// this class and the arithmetic cores agree while every FP header stays self-contained.
//
// The atomics the class befriends are declared in <cuda/__fp/fpmp_impl_atomic.h>,
// also included above, since a friend declaration needs them visible beforehand;
// they are defined at the end of this header.

/*********************************************************************
 * Multi-precision 32-bit floating-point emulation type (double-float)
 *********************************************************************/
//! @brief Multi-precision 32-bit floating-point emulation type (double-float)
//!
//! The `fpmp2` class provides a flexible, efficient, and accurate software-emulated
//! floating-point type supporting extended precision beyond standard IEEE 754 single-precision (float).
//! It is designed for GPU and CPU usage with double-float precision (hi, lo components),
//! with templates to control the underlying accuracy level (via the `fpmp2_accuracy` parameter).
//!
//! ## Template Parameters
//! - `fpmp2_accuracy`: The arithmetic accuracy level. Usually one of:
//!     - `fpmp2_accuracy::low` (favor speed, possibly at minor cost in accuracy),
//!     - `fpmp2_accuracy::def`, `fpmp2_accuracy::mid`, `fpmp2_accuracy::high`
//!
//! ## Features and Operations
//! - **Arithmetic**: Supports standard operators (+, -, *, /), fused multiply-add (fma), square roots (rsqrt, sqrt).
//! - **Renormalization**: Supports renormalization of the result of the arithmetic operations (useful for fast mode).
//! - **Construction**: Can be constructed from float, double, int32_t, uint32_t, int64_t, uint64_t.
//! - **Conversion**: Provides explicit and implicit conversion to standard C++ scalar types.
//! - **Comparison**: Supports all common relational operators (==, !=, <, <=, >, >=)
//! - **GPU & Host Compatibility**: All operations and members are decorated for both device and host use.
//!
//! ## Internal Representation
//! The class internally stores its value in `val`, a structure `fp32m2_t`
//! encoding the high and low float components following the multi-precision scheme.
//!
//! **Example Usage**:
//! @code
//! fp32mp2 a = 1.0f; // fpmp2<float>, the double-float type
//! fp32mp2 b = 2.0f;
//! auto c = a + b; // High-precision addition
//! float f = static_cast<float>(c); // Convert back to float
//! float hi = c.hi(), lo = c.lo(); // The two components that make up the value
//! @endcode
//!
//! ## Motivation
//! This class is intended for scenarios requiring higher precision than float offers,
//! for example in scientific computing, GPU linear algebra, or when porting algorithms requiring
//! quad/double-scalar emulation to platforms where native double/quad is slow or unavailable.
//!
//! ## Thread Safety
//! - Each instance manages its own state and is safe for concurrent use in different threads.
//!
//! ## Limitations
//! - Denormals, NaN, and Inf handling may differ from IEEE 754 strict standards, depending on accuracy level.
//! - Performance depends on template parameters and underlying hardware.

// fpmp2 class template
// _FpType: the component type, float (double-float) or double (double-double).
//     Not defaulted: neither precision is the natural fallback for the other, so
//     the choice is always spelled out -- use the fp32mp2 / fp64mp2 aliases below.
// met: arithmetic accuracy level
//     - fpmp2_accuracy::mid (default): Dekker-based split and error accumulation technique
//     - fpmp2_accuracy::high: Thall-based and other techniques
//     - fpmp2_accuracy::low: fast arithmetic operation without re-normalizations
template <typename _FpType, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
class alignas(2 * alignof(_FpType)) fpmp2
{
public:
  // float selects the double-float representation (fp32mp2) and double the
  // double-double one (fp64mp2); see __fpmp2_is_supported_fp_v for why the set is
  // exactly these two types.
  static_assert(__fpmp2_is_supported_fp_v<_FpType>,
                "cuda::experimental::fpmp2 supports only _FpType == float (double-float) "
                "or double (double-double)");

  /*
  // Accessor functions for hi and lo fields.
  // constexpr so the cross-method converting constructor below (and any
  // other context that needs (hi, lo) at compile time) can stay constexpr.
  */
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _FpType hi() const noexcept
  {
    return __mp2_hi_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _FpType lo() const noexcept
  {
    return __mp2_lo_;
  }

  /*
  // Accessors for volatile objects, so that reading a limb does not require copying
  // the whole value out first. Not constexpr: a volatile read is never a constant
  // expression.
  */
  [[nodiscard]] _CCCL_HOST_DEVICE_API _FpType hi() const volatile noexcept
  {
    return __mp2_hi_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API _FpType lo() const volatile noexcept
  {
    return __mp2_lo_;
  }

  /*
  // Basic constructors
  */
  // Default constructor
  _CCCL_HIDE_FROM_ABI fpmp2() = default;

  // Constructor from hi and lo floats (direct initialization).
  // constexpr so constant `fpmp2` arrays can live in constexpr
  // context
  _CCCL_HOST_DEVICE_API constexpr fpmp2(_FpType __hi, _FpType __lo) noexcept
      : __mp2_hi_{__hi}
      , __mp2_lo_{__lo}
  {}

  /*
  // Defaulted copy constructor (trivially copyable)
  // Note: NVCC implicitly makes defaulted special members __host__ __device__
  */
  _CCCL_HIDE_FROM_ABI fpmp2(const fpmp2& __other) = default;

  /*
  // Volatile support: the constructor and assignment operators below, plus the
  // volatile hi() / lo() accessors above, cover storage only, i.e. load, store and
  // (hi, lo)-preserving round-trip, which is what the legacy pattern of keeping
  // shared-memory scalars in volatile variables needs.
  //
  // A volatile object cannot be an operand of arithmetic,
  // comparison or the math API: those take const fpmp2&, and a volatile lvalue never
  // binds to it, not even through the converting constructor below, because
  // reference-related types are required to bind directly. Copy into a non-volatile
  // local, compute there, store the result back.
  */

  /*
  // Copy constructor from volatile fpmp2
  // Template so it is NOT a copy constructor per the C++ standard.
  // The volatile overloads are wrapped in dummy templates
  // so that the C++ standard does not consider them copy constructors/assignment
  // operators (a template is never a copy constructor or copy assignment operator),
  // preserving trivial copyability while retaining volatile access support.
  */
  template <typename _Dummy = void>
  _CCCL_HOST_DEVICE_API fpmp2(const volatile fpmp2& __other) noexcept
      : __mp2_hi_{__other.__mp2_hi_}
      , __mp2_lo_{__other.__mp2_lo_}
  {}

  // Defaulted copy assignment operator (trivially copyable)
  _CCCL_HIDE_FROM_ABI constexpr fpmp2& operator=(const fpmp2& __other) = default;

  /*
  // Assignment operator to volatile fpmp2
  // Template so it is NOT a copy assignment operator per the C++ standard
  // Returns void to avoid C++20 -Wvolatile (deprecated volatile return)
  */
  template <typename _Dummy = void>
  _CCCL_HOST_DEVICE_API void operator=(const fpmp2& __other) volatile noexcept
  {
    __mp2_hi_ = __other.__mp2_hi_;
    __mp2_lo_ = __other.__mp2_lo_;
  }

  /*
  // Assignment operator from volatile fpmp2
  // Template so it is NOT a copy assignment operator per the C++ standard
  */
  template <typename _Dummy = void>
  _CCCL_HOST_DEVICE_API fpmp2& operator=(const volatile fpmp2& __other) noexcept
  {
    __mp2_hi_ = __other.__mp2_hi_;
    __mp2_lo_ = __other.__mp2_lo_;
    return *this;
  }

  /*
  // Assignment operator from volatile to volatile fpmp2, e.g. a shared-memory to
  // shared-memory copy
  // Template so it is NOT a copy assignment operator per the C++ standard
  // Returns void to avoid C++20 -Wvolatile (deprecated volatile return)
  */
  template <typename _Dummy = void>
  _CCCL_HOST_DEVICE_API void operator=(const volatile fpmp2& __other) volatile noexcept
  {
    __mp2_hi_ = __other.__mp2_hi_;
    __mp2_lo_ = __other.__mp2_lo_;
  }

  /*
  // Cross-method converting constructor (same FpType, different method tag).
  //
  // Different fpmp2 specializations with the same FpType share an
  // identical (hi, lo) representation; only the method tag (which selects
  // the algorithm used by downstream arithmetic) differs. Without this
  // overload, a direct-init like
  //     fp32mp2_low c(b);   // b is fp32mp2_high
  // would silently route through fpmp2(double), i.e. operator
  // double() + ctor(double), an expensive round trip — particularly bad on
  // GPUs with limited FP64 throughput. With this overload, the same
  // direct-init becomes a plain (hi, lo) copy.
  //
  // Marked `explicit` on purpose: copy-initialization
  //     fp32mp2_low d = b;   // ill-formed
  // and copy-assignment
  //     a = b;                  // ill-formed
  // continue to fail to compile (a single implicit conversion sequence
  // can't chain two user-defined conversions). To opt in, write
  //     fp32mp2_low c(b);              // direct-init
  //     a = fp32mp2_low(b);            // explicit conversion + assign
  //     a = static_cast<fp32mp2_low>(b);
  //
  // SFINAE excludes met2 == met to avoid clashing with the defaulted
  // copy constructor.
  */
  _CCCL_TEMPLATE(fpmp2_accuracy _TypeAcc2)
  _CCCL_REQUIRES((_TypeAcc2 != _TypeAcc))
  _CCCL_HOST_DEVICE_API constexpr explicit fpmp2(const fpmp2<_FpType, _TypeAcc2>& __other) noexcept
      : __mp2_hi_{__other.hi()}
      , __mp2_lo_{__other.lo()}
  {}

  /*
  // Cross-precision converting constructor + assignment (upconvert): fp32mp2 -> fp64mp2.
  //
  // Enabled only when FpType == double. The conversion is exact and lossless:
  //   - Each float component casts losslessly to double (float values are a
  //     subset of double values), so (d_hi, d_lo) = ((double)hi, (double)lo)
  //     is an exact representation of the original mathematical value.
  //   - The pair is then renormalized with fast_two_sum so the result is a
  //     valid fp64mp2 (i.e. |out.lo| <= ulp_double(out.hi)/2).
  //
  // Why renormalize instead of just collapsing to (d_hi + d_lo, 0.0):
  //   In a renormalized fp32mp2 only |lo| <= ulp_float(hi)/2 = 2^-24*|hi|
  //   is guaranteed, but lo may be far smaller — e.g. (1.0f, 2^-100f) is a
  //   valid pair representing 1 + 2^-100. That value is NOT representable
  //   in a single double (2^-100 falls below the 53-bit precision of 1.0),
  //   but it IS representable as the fp64mp2 pair (1.0, 2^-100) because
  //   fp64mp2's renormalization bound is 2^-53*|hi|. fast_two_sum captures
  //   exactly this residual, so no precision is ever lost.
  //
  // Implicit only while the method tag is preserved: that form mirrors the
  // IEEE-754 float -> double widening (no precision loss). A widening that
  // also switches the method tag is `explicit`, matching the same-precision
  // cross-method constructor above: the tag selects the algorithm used by
  // downstream arithmetic, so changing it stays an opt-in rather than
  // something that rides along with a precision conversion. Note that
  // fpmp2_accuracy::def aliases mid, so fp32mp2 -> fp64mp2 (and
  // fp32mp2 -> fp64mp2_mid) is the implicit form.
  */
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  _CCCL_HOST_DEVICE_API fpmp2(const fpmp2<float, _TypeAcc>& __src) noexcept
  {
    const double __d_hi_in = static_cast<double>(__src.hi());
    const double __d_lo_in = static_cast<double>(__src.lo());
    // Renormalized fp32mp2 has |hi| >= |lo|, so fast_two_sum is safe.
    __mp2_hi_ = __fpmp_fast_two_sum(__d_hi_in, __d_lo_in, &__mp2_lo_);
  }

  // Widening across method tags: reinterpret the (hi, lo) pair under the
  // destination tag, then delegate to the widening constructor above.
  _CCCL_TEMPLATE(typename _Up = _FpType, fpmp2_accuracy _TypeAcc2)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up> _CCCL_AND(_TypeAcc2 != _TypeAcc))
  _CCCL_HOST_DEVICE_API explicit fpmp2(const fpmp2<float, _TypeAcc2>& __src) noexcept
      : fpmp2{fpmp2<float, _TypeAcc>{__src.hi(), __src.lo()}}
  {}

  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  _CCCL_HOST_DEVICE_API fpmp2& operator=(const fpmp2<float, _TypeAcc>& __src) noexcept
  {
    const double __d_hi_in = static_cast<double>(__src.hi());
    const double __d_lo_in = static_cast<double>(__src.lo());
    __mp2_hi_              = __fpmp_fast_two_sum(__d_hi_in, __d_lo_in, &__mp2_lo_);
    return *this;
  }

  /*
  // Cross-precision converting constructor + assignment (downconvert): fp64mp2 -> fp32mp2.
  //
  // Enabled only when FpType == float. The conversion is lossy (double's
  // 53 bits do not fit in float's 24), so it is performed via:
  //   1. Split src.hi() into a (float, float) pair: (a_hi, a_lo).
  //   2. Split src.lo() into a (float, float) pair: (b_hi, b_lo).
  //   3. Sum the two fp32mp2 pairs with __fpmp2_add<float> to obtain a
  //      renormalized fp32mp2 result.
  // This typically preserves ~48 bits of effective precision (the fp32mp2
  // limit), losing only ~5 bits relative to the fp64mp2 input.
  //
  // Marked _CCCL_FPMP_EXPLICIT (matches the existing double -> fp32mp2
  // narrowing constructor) so callers must opt in via static_cast or
  // direct-init, mirroring the IEEE-754 double -> float narrowing.
  // Narrowing that also switches the method tag is always `explicit`,
  // independent of the CCCL_FPMP_EXPLICIT_CASTS knob, so that a tag change
  // is opt-in in every direction.
  // The companion assignment operator is provided for symmetry; both
  // perform the same precision-preserving 2-pair add.
  */
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  _CCCL_HOST_DEVICE_API _CCCL_FPMP_EXPLICIT fpmp2(const fpmp2<double, _TypeAcc>& __src) noexcept
  {
    float __a_hi;
    float __a_lo;
    float __b_hi;
    float __b_lo;
    __fpmp2_from_double<float>(__src.hi(), &__a_hi, &__a_lo);
    __fpmp2_from_double<float>(__src.lo(), &__b_hi, &__b_lo);
    __fpmp2_add<float>(__a_hi, __a_lo, __b_hi, __b_lo, &__mp2_hi_, &__mp2_lo_);
  }

  // Narrowing across method tags: reinterpret the (hi, lo) pair under the
  // source precision with the destination tag, then delegate above.
  _CCCL_TEMPLATE(typename _Up = _FpType, fpmp2_accuracy _TypeAcc2)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up> _CCCL_AND(_TypeAcc2 != _TypeAcc))
  _CCCL_HOST_DEVICE_API explicit fpmp2(const fpmp2<double, _TypeAcc2>& __src) noexcept
      : fpmp2{fpmp2<double, _TypeAcc>{__src.hi(), __src.lo()}}
  {}

  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  _CCCL_HOST_DEVICE_API fpmp2& operator=(const fpmp2<double, _TypeAcc>& __src) noexcept
  {
    float __a_hi;
    float __a_lo;
    float __b_hi;
    float __b_lo;
    __fpmp2_from_double<float>(__src.hi(), &__a_hi, &__a_lo);
    __fpmp2_from_double<float>(__src.lo(), &__b_hi, &__b_lo);
    __fpmp2_add<float>(__a_hi, __a_lo, __b_hi, __b_lo, &__mp2_hi_, &__mp2_lo_);
    return *this;
  }

  /*
  // Conversion operators
  */
  // ==== Conversions from other types to fpmp2:
  // Implicit conversion from a single FpType (lo == 0).
  // constexpr so float/double constants flow into constexpr coefficient
  // tables without forcing callers to materialise the (hi, lo) pair.
  _CCCL_HOST_DEVICE_API constexpr fpmp2(_FpType __f) noexcept
      : __mp2_hi_{__f}
      , __mp2_lo_{(_FpType) 0}
  {}

  /*
  // Constructor from double (only for FpType == float)
  // Compile-time evaluation uses plain float casts, run time delegates to
  // __fpmp2_from_double. When FpType is double, use the regular FpType
  // constructor instead.
  //
  // The split is computed by a static helper the constructor delegates to,
  // rather than assigned in the constructor body: through C++17 a constexpr
  // constructor has to initialize every member in its mem-initializer list, so
  // a body that assigns cannot be used in a constant expression there.
  */
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  _CCCL_HOST_DEVICE_API constexpr _CCCL_FPMP_EXPLICIT fpmp2(double __d) noexcept
      : fpmp2(__split_double(__d))
  {}

//  __fpmp_fp128  operations, both directions restricted to FpType == double
// declared wherever the 128-bit type can be named (_CCCL_FPMP_FP128_ENABLE), which is a
// property of the host toolchain and so holds in both passes of a CUDA compilation; under
// nvcc they are device-callable only from sm_100 up (_CCCL_FPMP_FP128_DEVICE_OPS)
#if _CCCL_FPMP_FP128_ENABLE == 1
  /*
  // fp128 is the interchange type for quad-precision host code (libquadmath,
  // Fortran real(16)), and the pair that carries meaning is double-double against
  // binary128: 106 significand bits against 113. A double-float holds ~48 bits,
  // fewer than a double, so its interchange type is double -- which is why both
  // directions below are restricted to FpType == double rather than only the
  // constructor. Library mode declares the fp64 entry points alone
  // (__fp64mp2_from_quad / __fp64mp2_to_quad), so an fp32 path would not link
  // there either.
  //
  // A caller who wants the quad image of an fp32mp2 asks for it through double,
  // which is exact for any pair within the double-float contract
  // (|lo| <= 1/2 ulp(hi)): (__fpmp_fp128) (double) x.
  */
  // Constructor from __fpmp_fp128.
  // Compile-time evaluation uses plain casts; at run time the split goes through
  // __fpmp2_from_quad. Delegates to a static helper for the same reason as the
  // double constructor above.
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  _CCCL_FPMP_FP128_API constexpr _CCCL_FPMP_EXPLICIT fpmp2(__fpmp_fp128 __d) noexcept
      : fpmp2(__split_quad(__d))
  {}
  // Explicit conversion to __fpmp_fp128
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  [[nodiscard]] _CCCL_FPMP_FP128_API explicit operator __fpmp_fp128() const noexcept
  {
    return __fpmp2_to_quad(__mp2_hi_, __mp2_lo_);
  }

  // fp32mp2 has no fp128 interchange in either direction, deleted rather than
  // merely absent so the diagnostic names the rule. Without these, the constructor
  // would report an ambiguity (quad converts to both float and double equally
  // well) and the conversion would silently succeed through operator double().
  // Callers who want the double image can spell it: (__fpmp_fp128) (double) x.
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  _CCCL_FPMP_FP128_API _CCCL_FPMP_EXPLICIT fpmp2(__fpmp_fp128) = delete;
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  _CCCL_FPMP_FP128_API explicit operator __fpmp_fp128() const = delete;
#endif // _CCCL_FPMP_FP128_ENABLE == 1

  // Constructor from any standard integer type (int / long / long long + unsigned).
  // The value is canonicalized to the fixed-width builtin: the target width comes from
  // __num_bits_v and the signedness-correct fixed-width type from __make_nbit_int_t, so
  // the static_cast selects the matching overloaded setter (signed vs unsigned) below.
  // Every integer type is thus handled unambiguously and portably (LP64 and LLP64).
  // bool / character types are excluded by __cccl_is_integer_v.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_HOST_DEVICE_API _CCCL_FPMP_EXPLICIT fpmp2(_Tp __i) noexcept
  {
    if constexpr (::cuda::std::__num_bits_v<_Tp> <= 32)
    {
      __set_from_int32(static_cast<::cuda::std::__make_nbit_int_t<32, ::cuda::std::is_signed_v<_Tp>>>(__i));
    }
    else
    {
      __set_from_int64(static_cast<::cuda::std::__make_nbit_int_t<64, ::cuda::std::is_signed_v<_Tp>>>(__i));
    }
  }
  // bool and character types are excluded from __cccl_is_integer_v, but `1.0 + true`
  // and `1.0 + 'a'` are valid for double, so mirror that behavior by delegating to the
  // integer constructor above. The widened type keeps the source's signedness, which
  // matters for char32_t: it is unsigned and as wide as int32_t, so a plain cast to
  // int32_t would turn values above 2^31 - 1 into negative ones.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND(!::cuda::std::__cccl_is_integer_v<_Tp>))
  _CCCL_HOST_DEVICE_API _CCCL_FPMP_EXPLICIT fpmp2(_Tp __i) noexcept
      : fpmp2(static_cast<::cuda::std::__make_nbit_int_t<(::cuda::std::__num_bits_v<_Tp> <= 32) ? 32 : 64,
                                                         ::cuda::std::is_signed_v<_Tp>>>(__i))
  {}
#if _CCCL_HAS_INT128()
  // 128-bit integers would silently truncate to 64 bits, so they are deleted until
  // real 128-bit support is added (tracking issue: extended-precision fp <-> __int128).
  // Mirror the integer ctor's explicitness so copy-init overload sets are unchanged.
  _CCCL_HOST_DEVICE_API _CCCL_FPMP_EXPLICIT fpmp2(__int128_t)  = delete;
  _CCCL_HOST_DEVICE_API _CCCL_FPMP_EXPLICIT fpmp2(__uint128_t) = delete;
#endif // _CCCL_HAS_INT128()

  /*
  // ==== Conversion from fpmp2 to other types:
  // Conversion to double follows the value: implicit out of fp32mp2, explicit out of
  // fp64mp2. Neither is gated by CCCL_FPMP_EXPLICIT_CASTS.
  //
  // For a double-float the pair sums into a double exactly -- two 24-bit
  // significands that do not overlap fit inside 53 bits -- so the conversion is a
  // widening one, the analog of the implicit IEEE-754 float -> double, and stays
  // implicit for ergonomics. For a double-double it is not: 106 significand bits are
  // being asked to fit in 53, and the low word is simply dropped. That is a narrowing
  // conversion and says so, which is the same reason operator float() is explicit for
  // both.
  //
  // The specifier cannot depend on _FpType directly before C++20, hence the
  // constrained templates. They come with a limitation worth knowing: a conversion
  // function template only enters overload resolution when the target type matches
  // its conversion-type-id exactly, so double is the only floating-point sink these
  // reach. Feeding an fpmp2 to a long double or a 128-bit sink needs hi() / lo(), or
  // the fp128 conversion below, which is exact where a widening through double would
  // not be. Once C++20 is the baseline, explicit(!__fpmp2_is_fp32_v<_FpType>) on a
  // single non-template operator expresses the same intent without the limitation.
  //
  // Implicitness here does not put FP64 into fpmp<->fpmp conversions or fpmp
  // arithmetic: cross-method conversions copy (hi, lo) directly, and mixed
  // fpmp/scalar operators promote the scalar up to fpmp. It only takes effect when an
  // fp32mp2 value is fed into a double-typed sink.
  */
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  _CCCL_HOST_DEVICE_API operator double() const noexcept
  {
    return __fpmp2_to_double(__mp2_hi_, __mp2_lo_);
  }
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  _CCCL_HOST_DEVICE_API operator double() const volatile noexcept
  {
    return __fpmp2_to_double(__mp2_hi_, __mp2_lo_);
  }
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  _CCCL_HOST_DEVICE_API explicit operator double() const noexcept
  {
    return __fpmp2_to_double(__mp2_hi_, __mp2_lo_);
  }
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  _CCCL_HOST_DEVICE_API explicit operator double() const volatile noexcept
  {
    return __fpmp2_to_double(__mp2_hi_, __mp2_lo_);
  }

  // Explicit conversions to other types
  // Conversion to float
  _CCCL_HOST_DEVICE_API explicit operator float() const noexcept
  {
    return __fpmp2_to_float(__mp2_hi_, __mp2_lo_);
  }
  _CCCL_HOST_DEVICE_API explicit operator float() const volatile noexcept
  {
    return __fpmp2_to_float(__mp2_hi_, __mp2_lo_);
  }

  // Conversion to any standard integer type (int / long / long long + unsigned).
  // The target width comes from __num_bits_v and the signedness-correct fixed-width
  // type from __make_nbit_int_t, selecting the matching overloaded __to_integer helper
  // below; excludes bool / character types. Provided for both const and const volatile
  // objects.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_HOST_DEVICE_API explicit operator _Tp() const noexcept
  {
    using _Up =
      ::cuda::std::__make_nbit_int_t<(::cuda::std::__num_bits_v<_Tp> <= 32) ? 32 : 64, ::cuda::std::is_signed_v<_Tp>>;
    return static_cast<_Tp>(__to_integer(_Up{}, __mp2_hi_, __mp2_lo_));
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_HOST_DEVICE_API explicit operator _Tp() const volatile noexcept
  {
    using _Up =
      ::cuda::std::__make_nbit_int_t<(::cuda::std::__num_bits_v<_Tp> <= 32) ? 32 : 64, ::cuda::std::is_signed_v<_Tp>>;
    return static_cast<_Tp>(__to_integer(_Up{}, __mp2_hi_, __mp2_lo_));
  }
#if _CCCL_HAS_INT128()
  // See the deleted 128-bit constructors above: avoid silent 64-bit truncation.
  _CCCL_HOST_DEVICE_API explicit operator __int128_t() const           = delete;
  _CCCL_HOST_DEVICE_API explicit operator __uint128_t() const          = delete;
  _CCCL_HOST_DEVICE_API explicit operator __int128_t() const volatile  = delete;
  _CCCL_HOST_DEVICE_API explicit operator __uint128_t() const volatile = delete;
#endif // _CCCL_HAS_INT128()

  // (renormalize)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2 renormalize(const fpmp2& __x) noexcept
  {
    fpmp2 __res;
    __fpmp2_renormalize(__x.__mp2_hi_, __x.__mp2_lo_, &__res.__mp2_hi_, &__res.__mp2_lo_);
    return __res;
  }

  /*
  // Arithmetic operations:
  */
  // (+)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2 operator+(const fpmp2& __x, const fpmp2& __y) noexcept
  {
    fpmp2 __res;
    if constexpr (_TypeAcc == fpmp2_accuracy::low)
    {
      __fpmp2_low_add(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_, &__res.__mp2_hi_, &__res.__mp2_lo_);
    }
    else if constexpr (_TypeAcc == fpmp2_accuracy::high)
    {
      __fpmp2_high_add(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_, &__res.__mp2_hi_, &__res.__mp2_lo_);
    }
    else
    {
      __fpmp2_add(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_, &__res.__mp2_hi_, &__res.__mp2_lo_);
    }
    return __res;
  }

  // (-)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2 operator-(const fpmp2& __x, const fpmp2& __y) noexcept
  {
    fpmp2 __res;
    if constexpr (_TypeAcc == fpmp2_accuracy::low)
    {
      __fpmp2_low_sub(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_, &__res.__mp2_hi_, &__res.__mp2_lo_);
    }
    else if constexpr (_TypeAcc == fpmp2_accuracy::high)
    {
      __fpmp2_high_sub(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_, &__res.__mp2_hi_, &__res.__mp2_lo_);
    }
    else
    {
      __fpmp2_sub(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_, &__res.__mp2_hi_, &__res.__mp2_lo_);
    }
    return __res;
  }

  // (*)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2 operator*(const fpmp2& __x, const fpmp2& __y) noexcept
  {
    fpmp2 __res;
    if constexpr (_TypeAcc == fpmp2_accuracy::low)
    {
      __fpmp2_low_mul(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_, &__res.__mp2_hi_, &__res.__mp2_lo_);
    }
    else
    {
      __fpmp2_mul(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_, &__res.__mp2_hi_, &__res.__mp2_lo_);
    }
    return __res;
  }

  // (/)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2 operator/(const fpmp2& __x, const fpmp2& __y) noexcept
  {
    fpmp2 __res;
    if constexpr (_TypeAcc == fpmp2_accuracy::low)
    {
      __fpmp2_low_div(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_, &__res.__mp2_hi_, &__res.__mp2_lo_);
    }
    else if constexpr (_TypeAcc == fpmp2_accuracy::high)
    {
      __fpmp2_high_div(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_, &__res.__mp2_hi_, &__res.__mp2_lo_);
    }
    else
    {
      __fpmp2_div(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_, &__res.__mp2_hi_, &__res.__mp2_lo_);
    }
    return __res;
  }

  /*
  // Optimized compound assignment for single-component operands (accumulate)
  // Uses specialized __fpmp2_acc functions which are more efficient than
  // full mp2+mp2 addition (saves ~6 operations by avoiding low-part 2Sum).
  */
  _CCCL_HOST_DEVICE_API fpmp2& operator+=(const _FpType __c) noexcept
  {
    if constexpr (_TypeAcc == fpmp2_accuracy::low)
    {
      __fpmp2_low_acc(__c, &__mp2_hi_, &__mp2_lo_);
    }
    else if constexpr (_TypeAcc == fpmp2_accuracy::high)
    {
      __fpmp2_high_acc(__c, &__mp2_hi_, &__mp2_lo_);
    }
    else
    {
      __fpmp2_acc(__c, &__mp2_hi_, &__mp2_lo_);
    }
    return *this;
  }
  _CCCL_HOST_DEVICE_API fpmp2& operator-=(const _FpType __c) noexcept
  {
    if constexpr (_TypeAcc == fpmp2_accuracy::low)
    {
      __fpmp2_low_acc(-__c, &__mp2_hi_, &__mp2_lo_);
    }
    else if constexpr (_TypeAcc == fpmp2_accuracy::high)
    {
      __fpmp2_high_acc(-__c, &__mp2_hi_, &__mp2_lo_);
    }
    else
    {
      __fpmp2_acc(-__c, &__mp2_hi_, &__mp2_lo_);
    }
    return *this;
  }

  // (neg)
  [[nodiscard]] _CCCL_HOST_DEVICE_API fpmp2 operator-() const noexcept
  {
    fpmp2 __res;
    __fpmp2_neg(__mp2_hi_, __mp2_lo_, &__res.__mp2_hi_, &__res.__mp2_lo_);
    return __res;
  }

  /*
  // Comparison operators:
  */
  // equality (==)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator==(const fpmp2& __x, const fpmp2& __y) noexcept
  {
    return __fpmp2_cmp_eq(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_);
  }
  // inequality (!=)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator!=(const fpmp2& __x, const fpmp2& __y) noexcept
  {
    return __fpmp2_cmp_ne(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_);
  }
  // less than (<)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<(const fpmp2& __x, const fpmp2& __y) noexcept
  {
    return __fpmp2_cmp_lt(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_);
  }
  // greater than (>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>(const fpmp2& __x, const fpmp2& __y) noexcept
  {
    return __fpmp2_cmp_gt(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_);
  }
  // less than or equal to (<=)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<=(const fpmp2& __x, const fpmp2& __y) noexcept
  {
    return __fpmp2_cmp_le(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_);
  }
  // greater than or equal to (>=)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>=(const fpmp2& __x, const fpmp2& __y) noexcept
  {
    return __fpmp2_cmp_ge(__x.__mp2_hi_, __x.__mp2_lo_, __y.__mp2_hi_, __y.__mp2_lo_);
  }

  // Prefix increment/decrement
  _CCCL_HOST_DEVICE_API fpmp2& operator++() noexcept
  {
    *this = *this + fpmp2(1.0f);
    return *this;
  }
  _CCCL_HOST_DEVICE_API fpmp2& operator--() noexcept
  {
    *this = *this - fpmp2(1.0f);
    return *this;
  }
  // Postfix increment/decrement
  _CCCL_HOST_DEVICE_API fpmp2 operator++(int) noexcept
  {
    fpmp2 __temp(*this);
    *this = *this + fpmp2(1.0f);
    return __temp;
  }
  _CCCL_HOST_DEVICE_API fpmp2 operator--(int) noexcept
  {
    fpmp2 __temp(*this);
    *this = *this - fpmp2(1.0f);
    return __temp;
  }
  // Compound assignment operators (multi-precision operand)
  _CCCL_HOST_DEVICE_API fpmp2& operator+=(const fpmp2& __other) noexcept
  {
    *this = *this + __other;
    return *this;
  }
  _CCCL_HOST_DEVICE_API fpmp2& operator-=(const fpmp2& __other) noexcept
  {
    *this = *this - __other;
    return *this;
  }
  _CCCL_HOST_DEVICE_API fpmp2& operator*=(const fpmp2& __other) noexcept
  {
    *this = *this * __other;
    return *this;
  }
  _CCCL_HOST_DEVICE_API fpmp2& operator/=(const fpmp2& __other) noexcept
  {
    *this = *this / __other;
    return *this;
  }

  /*
  // Mixed types arithmetic operations
  // Support for mixed arithmetic and emulation types
  */
  // === mul ===
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2> || ::cuda::std::is_same_v<_T2, fpmp2>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2 operator*(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2(__x) * fpmp2(__y);
  }
  // === div ===
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2> || ::cuda::std::is_same_v<_T2, fpmp2>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2 operator/(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2(__x) / fpmp2(__y);
  }
  // === add ===
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2> || ::cuda::std::is_same_v<_T2, fpmp2>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2 operator+(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2(__x) + fpmp2(__y);
  }
  // === sub ===
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2> || ::cuda::std::is_same_v<_T2, fpmp2>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2 operator-(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2(__x) - fpmp2(__y);
  }
  // equality (==)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2> || ::cuda::std::is_same_v<_T2, fpmp2>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator==(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2(__x) == fpmp2(__y);
  }
  // inequality (!=)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2> || ::cuda::std::is_same_v<_T2, fpmp2>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator!=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2(__x) != fpmp2(__y);
  }
  // less than (<)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2> || ::cuda::std::is_same_v<_T2, fpmp2>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2(__x) < fpmp2(__y);
  }
  // greater than (>)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2> || ::cuda::std::is_same_v<_T2, fpmp2>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2(__x) > fpmp2(__y);
  }
  // less than or equal to (<=)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2> || ::cuda::std::is_same_v<_T2, fpmp2>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2(__x) <= fpmp2(__y);
  }
  // greater than or equal to (>=)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2> || ::cuda::std::is_same_v<_T2, fpmp2>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2(__x) >= fpmp2(__y);
  }

private:
#if _CCCL_CUDA_COMPILATION()
  // See the note on the forward declarations above: the atomics pass the addresses
  // of __mp2_hi_ and __mp2_lo_ to the flat (hi*, lo*) interface.
  template <typename _Up, fpmp2_accuracy _Acc>
  friend _CCCL_DEVICE_API fpmp2<_Up, _Acc> atomicAdd(fpmp2<_Up, _Acc>*, const fpmp2<_Up, _Acc>&) noexcept;
  template <typename _Up, fpmp2_accuracy _Acc>
  friend _CCCL_DEVICE_API fpmp2<_Up, _Acc> atomicSub(fpmp2<_Up, _Acc>*, const fpmp2<_Up, _Acc>&) noexcept;
#endif // _CCCL_CUDA_COMPILATION()

  // Wider-than-FpType splits, used by the delegating constructors above. Plain casts
  // during constant evaluation, the arithmetic primitive at run time. Where the
  // compiler cannot tell the two apart (no __builtin_is_constant_evaluated, so
  // pre-GCC-9 hosts and tile mode) _CCCL_IF_CONSTEVAL_DEFAULT selects the cast and
  // discards the run-time branch, which GCC 7 needs in order to accept the split as
  // constexpr at all: it rejects the function as soon as the body mentions a
  // non-constexpr call, reachable or not.
  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr fpmp2 __split_double(double __d) noexcept
  {
    _CCCL_IF_CONSTEVAL_DEFAULT
    {
      return fpmp2{(_FpType) __d, (_FpType) (__d - (double) (_FpType) __d)};
    }
    else
    {
      _FpType __hi{};
      _FpType __lo{};
      __fpmp2_from_double(__d, &__hi, &__lo);
      return fpmp2{__hi, __lo};
    }
  }

#if _CCCL_FPMP_FP128_ENABLE == 1
  [[nodiscard]] _CCCL_FPMP_FP128_API static constexpr fpmp2 __split_quad(__fpmp_fp128 __d) noexcept
  {
    _CCCL_IF_CONSTEVAL_DEFAULT
    {
      return fpmp2{(_FpType) __d, (_FpType) (__d - (__fpmp_fp128) (_FpType) __d)};
    }
    else
    {
      _FpType __hi{};
      _FpType __lo{};
      __fpmp2_from_quad(__d, &__hi, &__lo);
      return fpmp2{__hi, __lo};
    }
  }
#endif // _CCCL_FPMP_FP128_ENABLE == 1

  // Signedness-overloaded integer setters: the width-canonical value produced by the
  // integer constructor selects the matching signed/unsigned fixed-width builtin, so
  // the constructor body needs no signedness branch.
  _CCCL_HOST_DEVICE_API void __set_from_int32(int32_t __i) noexcept
  {
    __fpmp2_from_int(__i, &__mp2_hi_, &__mp2_lo_);
  }
  _CCCL_HOST_DEVICE_API void __set_from_int32(uint32_t __i) noexcept
  {
    __fpmp2_from_uint(__i, &__mp2_hi_, &__mp2_lo_);
  }
  _CCCL_HOST_DEVICE_API void __set_from_int64(int64_t __i) noexcept
  {
    __fpmp2_from_ll(__i, &__mp2_hi_, &__mp2_lo_);
  }
  _CCCL_HOST_DEVICE_API void __set_from_int64(uint64_t __i) noexcept
  {
    __fpmp2_from_ull(__i, &__mp2_hi_, &__mp2_lo_);
  }

  // Signedness-overloaded integer getters (mirror of the setters): the width-canonical
  // type from the conversion operator selects the matching signed/unsigned builtin, so
  // the operator body needs no signedness branch. Static + (hi, lo) by value so a single
  // overload set serves both the const and const volatile conversion operators.
  _CCCL_HOST_DEVICE_API static int32_t __to_integer(int32_t, _FpType __hi, _FpType __lo) noexcept
  {
    return __fpmp2_to_int(__hi, __lo);
  }
  _CCCL_HOST_DEVICE_API static uint32_t __to_integer(uint32_t, _FpType __hi, _FpType __lo) noexcept
  {
    return __fpmp2_to_uint(__hi, __lo);
  }
  _CCCL_HOST_DEVICE_API static int64_t __to_integer(int64_t, _FpType __hi, _FpType __lo) noexcept
  {
    return __fpmp2_to_ll(__hi, __lo);
  }
  _CCCL_HOST_DEVICE_API static uint64_t __to_integer(uint64_t, _FpType __hi, _FpType __lo) noexcept
  {
    return __fpmp2_to_ull(__hi, __lo);
  }

  /*
  // Internal storage - two floats (hi, lo) representing double-float precision
  */
  _FpType __mp2_hi_;
  _FpType __mp2_lo_;
}; // class fpmp2

/*********************************************************************
 * Accuracy-explicit arithmetic free functions
 *
 * Allow overriding the arithmetic method for a single operation
 * without changing the type, e.g.:
 *   using ffloat = fp32mp2_low;
 *   ffloat x = sub<fpmp2_accuracy::high>(a, b);  // accurate sub, result stays ffloat
 *
 * This avoids instantiating a second fpmp2 specialization and
 * the associated register pressure on GPU.
 *
 * Two forms are provided per operation:
 *   - Strict form: both/all operands are fpmp2<FpType, met>. Used
 *     when types are already matched.
 *   - Mixed form (one template, symmetric): accepts any combination where
 *     at least one operand is fpmp2 and at least one is a built-in
 *     arithmetic type. The arithmetic side is converted to the fpmp2 side
 *     via its (implicit/explicit) constructor, then dispatched to the
 *     strict form. Examples:
 *       ffloat x = sub<fpmp2_accuracy::high>(a, 1.0f);  // ffloat - float
 *       ffloat y = sub<fpmp2_accuracy::high>(1.0f, a);  // float  - ffloat
 *
 * The predicate "at least one is fpmp2 AND at least one is
 * arithmetic" is the same disjoint-categories trick used by the operator
 * overloads above, so one symmetric template suffices to cover both
 * argument orders without ambiguity against the strict form.
 *********************************************************************/

// Trait: detect any specialization of fpmp2<FpType, met>.
// __fpmp_-prefixed to avoid polluting the global namespace.

template <typename _Tp>
inline constexpr bool __fpmp_is_fpmp2_v = false;
template <typename _FpType, fpmp2_accuracy _TypeAcc>
inline constexpr bool __fpmp_is_fpmp2_v<fpmp2<_FpType, _TypeAcc>> = true;

/*********************************************************************
 * Standard-named math free functions (sqrt / rsqrt / fma / mad). These
 * are plain non-friend free functions: they read the operands through
 * the public hi()/lo() accessors and build the result with the public
 * (hi, lo) constructor, so they need no friendship.
 * (The arithmetic/comparison operators and renormalize remain friends.)
 *********************************************************************/

template <typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> sqrt(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __rhi, __rlo;
  __fpmp2_sqrt(__x.hi(), __x.lo(), &__rhi, &__rlo);
  return fpmp2<_FpType, _TypeAcc>(__rhi, __rlo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> rsqrt(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __rhi, __rlo;
  __fpmp2_rsqrt(__x.hi(), __x.lo(), &__rhi, &__rlo);
  return fpmp2<_FpType, _TypeAcc>(__rhi, __rlo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
fma(const fpmp2<_FpType, _TypeAcc>& __x,
    const fpmp2<_FpType, _TypeAcc>& __y,
    const fpmp2<_FpType, _TypeAcc>& __z) noexcept
{
  _FpType __rhi, __rlo;
  if constexpr (_TypeAcc == fpmp2_accuracy::low)
  {
    __fpmp2_low_fma(__x.hi(), __x.lo(), __y.hi(), __y.lo(), __z.hi(), __z.lo(), &__rhi, &__rlo);
  }
  else if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    __fpmp2_high_fma(__x.hi(), __x.lo(), __y.hi(), __y.lo(), __z.hi(), __z.lo(), &__rhi, &__rlo);
  }
  else
  {
    __fpmp2_fma(__x.hi(), __x.lo(), __y.hi(), __y.lo(), __z.hi(), __z.lo(), &__rhi, &__rlo);
  }
  return fpmp2<_FpType, _TypeAcc>(__rhi, __rlo);
}

_CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
_CCCL_REQUIRES(
  ((__fpmp_is_fpmp2_v<_T1> || __fpmp_is_fpmp2_v<_T2> || __fpmp_is_fpmp2_v<_T3>)
   && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>) ))
[[nodiscard]] _CCCL_HOST_DEVICE_API inline auto fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using mp2 =
    ::cuda::std::conditional_t<__fpmp_is_fpmp2_v<_T1>, _T1, ::cuda::std::conditional_t<__fpmp_is_fpmp2_v<_T2>, _T2, _T3>>;
  return fma(mp2(__x), mp2(__y), mp2(__z));
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
mad(const fpmp2<_FpType, _TypeAcc>& __x,
    const fpmp2<_FpType, _TypeAcc>& __y,
    const fpmp2<_FpType, _TypeAcc>& __z) noexcept
{
  _FpType __rhi, __rlo;
  if constexpr (_TypeAcc == fpmp2_accuracy::low)
  {
    __fpmp2_low_mad(__x.hi(), __x.lo(), __y.hi(), __y.lo(), __z.hi(), __z.lo(), &__rhi, &__rlo);
  }
  else if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    __fpmp2_high_mad(__x.hi(), __x.lo(), __y.hi(), __y.lo(), __z.hi(), __z.lo(), &__rhi, &__rlo);
  }
  else
  {
    __fpmp2_mad(__x.hi(), __x.lo(), __y.hi(), __y.lo(), __z.hi(), __z.lo(), &__rhi, &__rlo);
  }
  return fpmp2<_FpType, _TypeAcc>(__rhi, __rlo);
}

_CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
_CCCL_REQUIRES(
  ((__fpmp_is_fpmp2_v<_T1> || __fpmp_is_fpmp2_v<_T2> || __fpmp_is_fpmp2_v<_T3>)
   && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>) ))
[[nodiscard]] _CCCL_HOST_DEVICE_API inline auto mad(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using mp2 =
    ::cuda::std::conditional_t<__fpmp_is_fpmp2_v<_T1>, _T1, ::cuda::std::conditional_t<__fpmp_is_fpmp2_v<_T2>, _T2, _T3>>;
  return mad(mp2(__x), mp2(__y), mp2(__z));
}

template <fpmp2_accuracy _Acc, typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
add(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __rhi, __rlo;
  if constexpr (_Acc == fpmp2_accuracy::low)
  {
    __fpmp2_low_add(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__rhi, &__rlo);
  }
  else if constexpr (_Acc == fpmp2_accuracy::high)
  {
    __fpmp2_high_add(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__rhi, &__rlo);
  }
  else
  {
    __fpmp2_add(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__rhi, &__rlo);
  }
  return fpmp2<_FpType, _TypeAcc>(__rhi, __rlo);
}

_CCCL_TEMPLATE(fpmp2_accuracy _Acc, typename _T1, typename _T2)
_CCCL_REQUIRES(((__fpmp_is_fpmp2_v<_T1> || __fpmp_is_fpmp2_v<_T2>)
                && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
[[nodiscard]] _CCCL_HOST_DEVICE_API inline auto add(const _T1& __x, const _T2& __y) noexcept
{
  using mp2 = ::cuda::std::conditional_t<__fpmp_is_fpmp2_v<_T1>, _T1, _T2>;
  return add<_Acc>(mp2(__x), mp2(__y));
}

template <fpmp2_accuracy _Acc, typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
sub(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __rhi, __rlo;
  if constexpr (_Acc == fpmp2_accuracy::low)
  {
    __fpmp2_low_sub(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__rhi, &__rlo);
  }
  else if constexpr (_Acc == fpmp2_accuracy::high)
  {
    __fpmp2_high_sub(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__rhi, &__rlo);
  }
  else
  {
    __fpmp2_sub(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__rhi, &__rlo);
  }
  return fpmp2<_FpType, _TypeAcc>(__rhi, __rlo);
}

_CCCL_TEMPLATE(fpmp2_accuracy _Acc, typename _T1, typename _T2)
_CCCL_REQUIRES(((__fpmp_is_fpmp2_v<_T1> || __fpmp_is_fpmp2_v<_T2>)
                && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
[[nodiscard]] _CCCL_HOST_DEVICE_API inline auto sub(const _T1& __x, const _T2& __y) noexcept
{
  using mp2 = ::cuda::std::conditional_t<__fpmp_is_fpmp2_v<_T1>, _T1, _T2>;
  return sub<_Acc>(mp2(__x), mp2(__y));
}

template <fpmp2_accuracy _Acc, typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
mul(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __rhi, __rlo;
  if constexpr (_Acc == fpmp2_accuracy::low)
  {
    __fpmp2_low_mul(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__rhi, &__rlo);
  }
  else
  {
    __fpmp2_mul(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__rhi, &__rlo);
  }
  return fpmp2<_FpType, _TypeAcc>(__rhi, __rlo);
}

_CCCL_TEMPLATE(fpmp2_accuracy _Acc, typename _T1, typename _T2)
_CCCL_REQUIRES(((__fpmp_is_fpmp2_v<_T1> || __fpmp_is_fpmp2_v<_T2>)
                && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
[[nodiscard]] _CCCL_HOST_DEVICE_API inline auto mul(const _T1& __x, const _T2& __y) noexcept
{
  using mp2 = ::cuda::std::conditional_t<__fpmp_is_fpmp2_v<_T1>, _T1, _T2>;
  return mul<_Acc>(mp2(__x), mp2(__y));
}

template <fpmp2_accuracy _Acc, typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
div(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __rhi, __rlo;
  if constexpr (_Acc == fpmp2_accuracy::low)
  {
    __fpmp2_low_div(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__rhi, &__rlo);
  }
  else if constexpr (_Acc == fpmp2_accuracy::high)
  {
    __fpmp2_high_div(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__rhi, &__rlo);
  }
  else
  {
    __fpmp2_div(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__rhi, &__rlo);
  }
  return fpmp2<_FpType, _TypeAcc>(__rhi, __rlo);
}

_CCCL_TEMPLATE(fpmp2_accuracy _Acc, typename _T1, typename _T2)
_CCCL_REQUIRES(((__fpmp_is_fpmp2_v<_T1> || __fpmp_is_fpmp2_v<_T2>)
                && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
[[nodiscard]] _CCCL_HOST_DEVICE_API inline auto div(const _T1& __x, const _T2& __y) noexcept
{
  using mp2 = ::cuda::std::conditional_t<__fpmp_is_fpmp2_v<_T1>, _T1, _T2>;
  return div<_Acc>(mp2(__x), mp2(__y));
}

template <fpmp2_accuracy _Acc, typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
fma(const fpmp2<_FpType, _TypeAcc>& __x,
    const fpmp2<_FpType, _TypeAcc>& __y,
    const fpmp2<_FpType, _TypeAcc>& __z) noexcept
{
  _FpType __rhi, __rlo;
  if constexpr (_Acc == fpmp2_accuracy::low)
  {
    __fpmp2_low_fma(__x.hi(), __x.lo(), __y.hi(), __y.lo(), __z.hi(), __z.lo(), &__rhi, &__rlo);
  }
  else if constexpr (_Acc == fpmp2_accuracy::high)
  {
    __fpmp2_high_fma(__x.hi(), __x.lo(), __y.hi(), __y.lo(), __z.hi(), __z.lo(), &__rhi, &__rlo);
  }
  else
  {
    __fpmp2_fma(__x.hi(), __x.lo(), __y.hi(), __y.lo(), __z.hi(), __z.lo(), &__rhi, &__rlo);
  }
  return fpmp2<_FpType, _TypeAcc>(__rhi, __rlo);
}

_CCCL_TEMPLATE(fpmp2_accuracy _Acc, typename _T1, typename _T2, typename _T3)
_CCCL_REQUIRES(
  ((__fpmp_is_fpmp2_v<_T1> || __fpmp_is_fpmp2_v<_T2> || __fpmp_is_fpmp2_v<_T3>)
   && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>) ))
[[nodiscard]] _CCCL_HOST_DEVICE_API inline auto fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using mp2 =
    ::cuda::std::conditional_t<__fpmp_is_fpmp2_v<_T1>, _T1, ::cuda::std::conditional_t<__fpmp_is_fpmp2_v<_T2>, _T2, _T3>>;
  return fma<_Acc>(mp2(__x), mp2(__y), mp2(__z));
}

template <fpmp2_accuracy _Acc, typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
mad(const fpmp2<_FpType, _TypeAcc>& __x,
    const fpmp2<_FpType, _TypeAcc>& __y,
    const fpmp2<_FpType, _TypeAcc>& __z) noexcept
{
  _FpType __rhi, __rlo;
  if constexpr (_Acc == fpmp2_accuracy::low)
  {
    __fpmp2_low_mad(__x.hi(), __x.lo(), __y.hi(), __y.lo(), __z.hi(), __z.lo(), &__rhi, &__rlo);
  }
  else if constexpr (_Acc == fpmp2_accuracy::high)
  {
    __fpmp2_high_mad(__x.hi(), __x.lo(), __y.hi(), __y.lo(), __z.hi(), __z.lo(), &__rhi, &__rlo);
  }
  else
  {
    __fpmp2_mad(__x.hi(), __x.lo(), __y.hi(), __y.lo(), __z.hi(), __z.lo(), &__rhi, &__rlo);
  }
  return fpmp2<_FpType, _TypeAcc>(__rhi, __rlo);
}

_CCCL_TEMPLATE(fpmp2_accuracy _Acc, typename _T1, typename _T2, typename _T3)
_CCCL_REQUIRES(
  ((__fpmp_is_fpmp2_v<_T1> || __fpmp_is_fpmp2_v<_T2> || __fpmp_is_fpmp2_v<_T3>)
   && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>) ))
[[nodiscard]] _CCCL_HOST_DEVICE_API inline auto mad(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using mp2 =
    ::cuda::std::conditional_t<__fpmp_is_fpmp2_v<_T1>, _T1, ::cuda::std::conditional_t<__fpmp_is_fpmp2_v<_T2>, _T2, _T3>>;
  return mad<_Acc>(mp2(__x), mp2(__y), mp2(__z));
}

#if _CCCL_CUDA_COMPILATION()
/*
 * ============================================================================
 * Warp Shuffle Helpers (CUDA-only, header-only)
 * ============================================================================
 * Overloads of CUDA's modern __shfl_sync family for the fpmp2 pair.
 * Each shuffle operates independently on the (hi, lo) components: the same
 * mask, lane/delta, and width are used for both halves so the two parts of
 * a multi-precision value always travel together to the same destination lane.
 *
 * Mirrors CUDA's API exactly:
 *   __shfl_sync     (mask, var, srcLane,  width = warpSize)
 *   __shfl_xor_sync (mask, var, laneMask, width = warpSize)
 *   __shfl_down_sync(mask, var, delta,    width = warpSize)
 *   __shfl_up_sync  (mask, var, delta,    width = warpSize)
 *
 * Declared in cuda::experimental next to fpmp2, so ADL on an fpmp2 argument
 * finds them at an unqualified call site, exactly as it finds CUDA's built-in
 * scalar overloads for float/double.  The recursive calls are spelled
 * `::__shfl_sync(mask, var.hi(), ...)` to reach CUDA's global-namespace
 * overload rather than recursing into this template.
 *
 * Defined only for CUDA compilation; the warp shuffle primitives have no host
 * counterpart, so host-only translation units never see them.
 *
 * These are thread-cooperation primitives (not math), so they live in the core
 * header and are available via <cuda/fpmp>.
 * ============================================================================
 */

template <typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
__shfl_sync(unsigned mask, const fpmp2<_FpType, _TypeAcc>& var, int srcLane, int width = warpSize) noexcept
{
  return fpmp2<_FpType, _TypeAcc>(
    ::__shfl_sync(mask, var.hi(), srcLane, width), ::__shfl_sync(mask, var.lo(), srcLane, width));
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
__shfl_xor_sync(unsigned mask, const fpmp2<_FpType, _TypeAcc>& var, int laneMask, int width = warpSize) noexcept
{
  return fpmp2<_FpType, _TypeAcc>(
    ::__shfl_xor_sync(mask, var.hi(), laneMask, width), ::__shfl_xor_sync(mask, var.lo(), laneMask, width));
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
__shfl_down_sync(unsigned mask, const fpmp2<_FpType, _TypeAcc>& var, unsigned int delta, int width = warpSize) noexcept
{
  return fpmp2<_FpType, _TypeAcc>(
    ::__shfl_down_sync(mask, var.hi(), delta, width), ::__shfl_down_sync(mask, var.lo(), delta, width));
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
__shfl_up_sync(unsigned mask, const fpmp2<_FpType, _TypeAcc>& var, unsigned int delta, int width = warpSize) noexcept
{
  return fpmp2<_FpType, _TypeAcc>(
    ::__shfl_up_sync(mask, var.hi(), delta, width), ::__shfl_up_sync(mask, var.lo(), delta, width));
}

/*
 * ============================================================================
 * Freestanding Atomic Operations for fpmp2
 * ============================================================================
 * CUDA-style freestanding atomic functions that work with the fpmp2 class.
 * They dispatch to the templated impl (inline mode) or the extern-"C" ABI
 * symbols (library mode), so they work in both modes.
 *
 * fp32mp2 works on every supported architecture. fp64mp2 needs a 128-bit
 * compare-exchange, so it requires sm_90+ and PTX ISA 8.4+; calling it elsewhere
 * fails to link on __fpmp2_dd_atomic_requires_SM_90_and_ptx_isa_840.
 * ============================================================================
 */

// atomicAdd: Atomic addition for fpmp2
// Returns the old value before the addition
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
atomicAdd(fpmp2<_FpType, _TypeAcc>* address, const fpmp2<_FpType, _TypeAcc>& val) noexcept
{
  // The (hi*, lo*) pair comes from the members' own addresses, and the old value is
  // collected into locals and assembled on return. The two components are distinct
  // members rather than an array, so reaching the second one by incrementing a
  // pointer to the first is not valid, however the class happens to be laid out.
  _FpType __old_hi{};
  _FpType __old_lo{};
#  if defined(_CCCL_FPMP_USE_LIB)
  // In library mode, call the library function directly
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    __fp32mp2_atomicAdd(&address->__mp2_hi_, &address->__mp2_lo_, val.hi(), val.lo(), &__old_hi, &__old_lo);
  }
  else if constexpr (__fpmp2_is_fp64_v<_FpType>)
  {
    __fp64mp2_atomicAdd(&address->__mp2_hi_, &address->__mp2_lo_, val.hi(), val.lo(), &__old_hi, &__old_lo);
  }
#  else
  __fpmp2_atomicAdd(&address->__mp2_hi_, &address->__mp2_lo_, val.hi(), val.lo(), &__old_hi, &__old_lo);
#  endif
  return fpmp2<_FpType, _TypeAcc>{__old_hi, __old_lo};
}

// atomicSub: Atomic subtraction for fpmp2
// Returns the old value before the subtraction
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
atomicSub(fpmp2<_FpType, _TypeAcc>* address, const fpmp2<_FpType, _TypeAcc>& val) noexcept
{
  // Same shape as atomicAdd above.
  _FpType __old_hi{};
  _FpType __old_lo{};
#  if defined(_CCCL_FPMP_USE_LIB)
  // In library mode, call the library function directly
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    __fp32mp2_atomicSub(&address->__mp2_hi_, &address->__mp2_lo_, val.hi(), val.lo(), &__old_hi, &__old_lo);
  }
  else if constexpr (__fpmp2_is_fp64_v<_FpType>)
  {
    __fp64mp2_atomicSub(&address->__mp2_hi_, &address->__mp2_lo_, val.hi(), val.lo(), &__old_hi, &__old_lo);
  }
#  else
  __fpmp2_atomicSub(&address->__mp2_hi_, &address->__mp2_lo_, val.hi(), val.lo(), &__old_hi, &__old_lo);
#  endif
  return fpmp2<_FpType, _TypeAcc>{__old_hi, __old_lo};
}

#endif // _CCCL_CUDA_COMPILATION()

/*********************************************************************
 * Aliases for the most common use cases
 *********************************************************************/
using fp32mp2      = fpmp2<float, fpmp2_accuracy::def>;
using fp32mp2_low  = fpmp2<float, fpmp2_accuracy::low>;
using fp32mp2_mid  = fpmp2<float, fpmp2_accuracy::mid>;
using fp32mp2_high = fpmp2<float, fpmp2_accuracy::high>;

using fp64mp2      = fpmp2<double, fpmp2_accuracy::def>;
using fp64mp2_low  = fpmp2<double, fpmp2_accuracy::low>;
using fp64mp2_mid  = fpmp2<double, fpmp2_accuracy::mid>;
using fp64mp2_high = fpmp2<double, fpmp2_accuracy::high>;
} // namespace cuda::experimental

// ============================================================================
// cuda::std overloads for sqrt / fma on the fpmp2 pair.
//
// A qualified cuda::std::sqrt / cuda::std::fma call suppresses ADL, so without
// these overloads it would silently narrow fpmp2 -> double and compute a
// native-double result. These forward to the cuda::experimental implementations
// (which unqualified / ADL calls already resolve to). The exact-type overloads
// cover pure fpmp2 arguments; the constrained fma overload handles mixed
// fpmp2 + built-in arithmetic operands. (mad has no cuda::std counterpart.)
// ============================================================================
_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2<_FpType, _TypeAcc>
sqrt(const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::experimental::sqrt(__x);
}

template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2<_FpType, _TypeAcc>
fma(const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x,
    const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __y,
    const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __z) noexcept
{
  return ::cuda::experimental::fma(__x, __y, __z);
}

_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(
  ((::cuda::experimental::__fpmp_is_fpmp2_v<_T1> || ::cuda::experimental::__fpmp_is_fpmp2_v<_T2>
    || ::cuda::experimental::__fpmp_is_fpmp2_v<_T3>)
   && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>) ))
[[nodiscard]] _CCCL_HOST_DEVICE_API auto fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  return ::cuda::experimental::fma(__x, __y, __z);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_H
