//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPTOOL_CUSTOM_H
#define _CUDA___FP_FPTOOL_CUSTOM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! @file fptool_custom.h
//! @brief fp_custom - a drop-in `double` replacement with configurable precision
//!
//! This header-only library provides an `fp_custom` class template that wraps native
//! double-precision floating-point operations while reducing the exponent and mantissa
//! to a chosen size. It's designed for:
//!
//!   - **Algorithm sensitivity analysis**: Test how algorithms behave with reduced precision
//!   - **Mixed-precision research**: Emulate lower-precision formats (float, bfloat16, etc.)
//!   - **CUDA/CPU compatibility**: Works identically on both host and device code
//!   - **Drop-in replacement**: Use `fp64_custom<>` where you would use `double`
//!
//! ## Quick Start
//!
//! ```cpp
//! #include <cuda/fptool>
//!
//! using namespace cuda::experimental;
//!
//! // Step 1: swap `double` for `fp64_custom<>`; behavior is unchanged.
//! fp64_custom<> a = 1.5, b = 2.5;
//! double native = a + b;
//!
//! // Step 2: ask for a reduced format, here float-like (8 exponent, 23 mantissa bits).
//! fp64_custom<8, 23> c = 1.5, d = 2.5;
//! ```
//!
//! ## Template Parameters
//!
//! | Parameter  | Meaning                                                                   |
//! |------------|---------------------------------------------------------------------------|
//! | `_FpType`  | Base type holding the value; only `double` is supported today             |
//! | `_ExpSize` | Exponent bits to preserve (2-11), or `fp_custom_dynamic_size` for runtime |
//! | `_MantSize`| Mantissa bits to preserve (0-52), or `fp_custom_dynamic_size` for runtime |
//!
//! The `fp64_custom` alias fixes the base type and defaults both sizes to the native
//! FP64 ones, which is the spelling used throughout this documentation.
//!
//! Sizes equal to the native ones (11 and 52) disable the corresponding reduction
//! entirely: the emulation code is discarded at compile time, leaving native `double`
//! arithmetic. Mantissa reduction always uses IEEE 754 round-to-nearest-even.
//!
//! ## Common Precision Configurations
//!
//! | Format | Exponent | Mantissa | Type                 |
//! |--------|----------|----------|----------------------|
//! | FP64   | 11       | 52       | `fp64_custom<>`      |
//! | FP32   | 8        | 23       | `fp64_custom<8, 23>` |
//! | BF16   | 8        | 7        | `fp64_custom<8, 7>`  |
//! | FP16   | 5        | 10       | `fp64_custom<5, 10>` |
//! | TF32   | 8        | 10       | `fp64_custom<8, 10>` |
//! | PO2    | 11       | 0        | `fp64_custom<11, 0>` |
//!
//! Zero mantissa bits leave only the implicit leading 1, so `fp64_custom<11, 0>` rounds
//! every value to the nearest power of two. The exponent, in contrast, starts at two
//! bits: an n-bit exponent field spends its all-ones pattern on infinity and NaN, so it
//! covers 2^n - 2 binades and a single bit would leave none.
//!
//! ## Underflow/Overflow Behavior
//!
//! When exponent bits are reduced (e.g., from 11 to 8 for FP32 emulation), values outside
//! the new dynamic range are clamped:
//!   - **Overflow**: Values too large for reduced exponent → Infinity (±INF)
//!   - **Underflow**: Values too small for reduced exponent → Zero (±0)
//!
//! In both cases the sign is preserved, and the clamped result skips mantissa reduction.
//! NaN and infinity are never clamped.
//!
//! ## How It Works
//!
//! Each arithmetic operation follows this pattern:
//! 1. Apply the precision reduction to input operands
//! 2. Perform the native FP64 operation
//! 3. Apply the precision reduction to the result
//!
//! This models how lower-precision hardware would handle the computation while
//! maintaining full FP64 representation for intermediate storage.
//!
//! ## Runtime Precision Control
//!
//! Passing `fp_custom_dynamic_size` instead of a size takes that field's size from a
//! global variable that can be changed at runtime, without recompiling:
//!
//! ```cpp
//! using Real = fp64_custom<fp_custom_dynamic_size, fp_custom_dynamic_size>;
//!
//! Real a = 1.0, b = 1e-15;
//! double full = a + b;                        // starts at full FP64 precision
//!
//! fp_custom_set_host_mantissa_size(23);       // switch to float-like precision
//! double reduced = a + b;                     // small term is now lost
//! ```
//!
//! The sizes start at the native FP64 values (11 and 52), so a program that never calls
//! a setter behaves exactly like `double`.
//!
//! ### Size Accessors
//!
//! | Function                                   | Target | Description                        |
//! |--------------------------------------------|--------|------------------------------------|
//! | `fp_custom_set_host_mantissa_size(int)`    | CPU    | Set mantissa bits on host (0-52)   |
//! | `fp_custom_set_host_exponent_size(int)`    | CPU    | Set exponent bits on host (2-11)   |
//! | `fp_custom_get_host_mantissa_size()`       | CPU    | Read mantissa bits on host         |
//! | `fp_custom_get_host_exponent_size()`       | CPU    | Read exponent bits on host         |
//! | `fp_custom_set_device_mantissa_size(int)`  | GPU    | Set mantissa bits on device (0-52) |
//! | `fp_custom_set_device_exponent_size(int)`  | GPU    | Set exponent bits on device (2-11) |
//! | `fp_custom_get_device_mantissa_size()`     | GPU    | Read mantissa bits on device       |
//! | `fp_custom_get_device_exponent_size()`     | GPU    | Read exponent bits on device       |
//!
//! Host and device sizes are independent — changing one does not affect the other.
//! The device accessors use `cudaMemcpyToSymbol` / `cudaMemcpyFromSymbol` when called
//! from host code; a `cudaDeviceSynchronize()` before the next kernel launch ensures a
//! new value is visible on the GPU. Called from device code they touch the variable
//! directly, and only thread 0 of block 0 writes it, to avoid races.
//!
//! The sizes live in one program-wide copy each, shared by all translation units. On the
//! device that sharing needs relocatable device code (`-rdc=true`); in whole-program mode
//! each translation unit necessarily gets its own device copy.
//!
//! Under NVRTC only the device accessors exist, since a JIT compilation has no host side.
//! They are then reached from device code, where they touch the variable directly.
//!
//! @note There is a small performance cost compared to fixed sizes because the sizes are
//! read from memory instead of being folded into the code.
//! @note Thread Safety: All operations are thread-safe (no shared mutable state) unless a
//! setter runs concurrently with arithmetic.

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>

#if _CCCL_CUDA_COMPILATION() && _CCCL_HOST_COMPILATION()
// Include the CUDA runtime for host-side functions like cudaMemcpyToSymbol
#  include <cuda_runtime.h>
#endif // _CCCL_CUDA_COMPILATION() && _CCCL_HOST_COMPILATION()

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

// === supported base types and field sizes ===
namespace cuda::experimental
{
//! @brief Size value selecting runtime control of a field
//!
//! Passed as `_ExpSize` or `_MantSize`, it takes that field's size from a global
//! variable instead of the template argument. See the setters and getters below.
inline constexpr uint16_t fp_custom_dynamic_size = static_cast<uint16_t>(-1);

//! @brief Base types fp_custom can hold values in
//!
//! Only double emulation is implemented today; the _FpType axis exists for future
//! extension. _Float64 is accepted as a bit-identical alias for double.
template <typename _Tp>
inline constexpr bool __fp_custom_is_supported_fp_v =
  ::cuda::std::is_same_v<_Tp, double>
// nvcc currently doesn't support _Float64 in device code.
#if __STDCPP_FLOAT64_T__ == 1 && !_CCCL_CUDA_COMPILER(NVCC)
  || ::cuda::std::is_same_v<_Tp, _Float64>
#endif // __STDCPP_FLOAT64_T__ == 1 && !_CCCL_CUDA_COMPILER(NVCC)
  ;

//! @brief Unreduced field sizes of a base type
//!
//! The primary template reports zero sizes so that an unsupported _FpType produces the
//! single static_assert below rather than an incomplete-type error.
template <typename _FpType>
struct __fp_custom_native_sizes
{
  static constexpr uint16_t __exp_size  = 0;
  static constexpr uint16_t __mant_size = 0;
};

template <>
struct __fp_custom_native_sizes<double>
{
  static constexpr uint16_t __exp_size  = 11;
  static constexpr uint16_t __mant_size = 52;
};

#if __STDCPP_FLOAT64_T__ == 1 && !_CCCL_CUDA_COMPILER(NVCC)
template <>
struct __fp_custom_native_sizes<_Float64> : __fp_custom_native_sizes<double>
{};
#endif // __STDCPP_FLOAT64_T__ == 1 && !_CCCL_CUDA_COMPILER(NVCC)

// === runtime sizes ===
// Sizes used by instantiations that pass fp_custom_dynamic_size, initialised to the
// unreduced sizes of the base type so that a program which never calls a setter keeps
// native behavior.
//
// These are variable templates rather than plain variables on purpose: a variable
// template has vague linkage, so all translation units share one copy, while a plain
// `inline _CCCL_DEVICE` variable is rejected by nvcc outside relocatable device code.
//
// They are the only mutable namespace-scope variables in this library; everything else
// at namespace scope is an immutable _CCCL_GLOBAL_CONSTANT or a constexpr trait.
//
// NVRTC compiles device code only, so the host half of the state and of the accessors
// does not exist there; the device half below is what a JIT-compiled kernel uses.
#if !_CCCL_COMPILER(NVRTC)
template <typename _FpType>
int __fp_custom_host_mantissa_size = __fp_custom_native_sizes<_FpType>::__mant_size;

template <typename _FpType>
int __fp_custom_host_exponent_size = __fp_custom_native_sizes<_FpType>::__exp_size;
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()
template <typename _FpType>
_CCCL_DEVICE int __fp_custom_device_mantissa_size = __fp_custom_native_sizes<_FpType>::__mant_size;

template <typename _FpType>
_CCCL_DEVICE int __fp_custom_device_exponent_size = __fp_custom_native_sizes<_FpType>::__exp_size;
#endif // _CCCL_CUDA_COMPILATION()

#if !_CCCL_COMPILER(NVRTC)

//! @brief Set the mantissa size used by host code (0-52)
template <typename _FpType = double>
_CCCL_HOST_API inline void fp_custom_set_host_mantissa_size(int __new_size) noexcept
{
  _CCCL_ASSERT(__new_size >= 0 && __new_size <= __fp_custom_native_sizes<_FpType>::__mant_size,
               "fp_custom mantissa size out of range");
  __fp_custom_host_mantissa_size<_FpType> = __new_size;
}

//! @brief Set the exponent size used by host code (2-11)
template <typename _FpType = double>
_CCCL_HOST_API inline void fp_custom_set_host_exponent_size(int __new_size) noexcept
{
  _CCCL_ASSERT(__new_size >= 2 && __new_size <= __fp_custom_native_sizes<_FpType>::__exp_size,
               "fp_custom exponent size out of range");
  __fp_custom_host_exponent_size<_FpType> = __new_size;
}

//! @brief Read the mantissa size used by host code
template <typename _FpType = double>
[[nodiscard]] _CCCL_HOST_API inline int fp_custom_get_host_mantissa_size() noexcept
{
  return __fp_custom_host_mantissa_size<_FpType>;
}

//! @brief Read the exponent size used by host code
template <typename _FpType = double>
[[nodiscard]] _CCCL_HOST_API inline int fp_custom_get_host_exponent_size() noexcept
{
  return __fp_custom_host_exponent_size<_FpType>;
}

#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()

//! @brief Set the mantissa size used by device code (0-52)
//!
//! From host code the write goes through `cudaMemcpyToSymbol`; from device code only
//! thread 0 of block 0 performs it.
template <typename _FpType = double>
_CCCL_HOST_DEVICE_API inline void fp_custom_set_device_mantissa_size(int __new_size) noexcept
{
  _CCCL_ASSERT(__new_size >= 0 && __new_size <= __fp_custom_native_sizes<_FpType>::__mant_size,
               "fp_custom mantissa size out of range");
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (if (threadIdx.x == 0 && blockIdx.x == 0) { __fp_custom_device_mantissa_size<_FpType> = __new_size; }),
    (cudaMemcpyToSymbol(__fp_custom_device_mantissa_size<_FpType>, &__new_size, sizeof(int));))
}

//! @brief Set the exponent size used by device code (2-11)
template <typename _FpType = double>
_CCCL_HOST_DEVICE_API inline void fp_custom_set_device_exponent_size(int __new_size) noexcept
{
  _CCCL_ASSERT(__new_size >= 2 && __new_size <= __fp_custom_native_sizes<_FpType>::__exp_size,
               "fp_custom exponent size out of range");
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (if (threadIdx.x == 0 && blockIdx.x == 0) { __fp_custom_device_exponent_size<_FpType> = __new_size; }),
    (cudaMemcpyToSymbol(__fp_custom_device_exponent_size<_FpType>, &__new_size, sizeof(int));))
}

//! @brief Read the mantissa size used by device code
template <typename _FpType = double>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline int fp_custom_get_device_mantissa_size() noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (return __fp_custom_device_mantissa_size<_FpType>;),
    (int __size = 0; cudaMemcpyFromSymbol(&__size, __fp_custom_device_mantissa_size<_FpType>, sizeof(int));
     return __size;))
}

//! @brief Read the exponent size used by device code
template <typename _FpType = double>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline int fp_custom_get_device_exponent_size() noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (return __fp_custom_device_exponent_size<_FpType>;),
    (int __size = 0; cudaMemcpyFromSymbol(&__size, __fp_custom_device_exponent_size<_FpType>, sizeof(int));
     return __size;))
}

#endif // _CCCL_CUDA_COMPILATION()

//! @brief Mantissa size in effect for a given _MantSize argument
//!
//! Folds to a constant unless the size is runtime-controlled.
template <typename _FpType, uint16_t _MantSize>
[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API int __fp_custom_mantissa_size() noexcept
{
  if constexpr (_MantSize == fp_custom_dynamic_size)
  {
#if _CCCL_CUDA_COMPILATION()
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return __fp_custom_device_mantissa_size<_FpType>;),
                      (return __fp_custom_host_mantissa_size<_FpType>;))
#else // ^^^ _CCCL_CUDA_COMPILATION() ^^^ / vvv !_CCCL_CUDA_COMPILATION() vvv
    return __fp_custom_host_mantissa_size<_FpType>;
#endif // !_CCCL_CUDA_COMPILATION()
  }
  else
  {
    return static_cast<int>(_MantSize);
  }
}

//! @brief Exponent size in effect for a given _ExpSize argument
template <typename _FpType, uint16_t _ExpSize>
[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API int __fp_custom_exponent_size() noexcept
{
  if constexpr (_ExpSize == fp_custom_dynamic_size)
  {
#if _CCCL_CUDA_COMPILATION()
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return __fp_custom_device_exponent_size<_FpType>;),
                      (return __fp_custom_host_exponent_size<_FpType>;))
#else // ^^^ _CCCL_CUDA_COMPILATION() ^^^ / vvv !_CCCL_CUDA_COMPILATION() vvv
    return __fp_custom_host_exponent_size<_FpType>;
#endif // !_CCCL_CUDA_COMPILATION()
  }
  else
  {
    return static_cast<int>(_ExpSize);
  }
}

// === precision reduction ===
//! @brief Precision reduction applied to operands and results
//!
//! This function modifies the bit representation of a double to simulate
//! reduced precision. It's called before and after each arithmetic operation.
//!
//! The reduction happens in two phases:
//! 1. **Exponent reduction** (if _ExpSize is below the native size, or dynamic):
//!    - Values outside the reduced exponent range become infinity or zero
//!    - Preserves the sign bit
//!    - NaN and infinity pass through unchanged
//!
//! 2. **Mantissa reduction** (if _MantSize is below the native size, or dynamic):
//!    - Excess mantissa bits are removed using IEEE 754 round-to-nearest-even
//!    - NaN and infinity are left untouched
//!
//! With native sizes on both axes the whole body is discarded, so `fp64_custom<>`
//! arithmetic compiles down to plain `double` arithmetic.
//!
//! @param __v  Reference to the bit pattern to modify (modified in place)
//!
//! @note Thread-safe: no shared state is modified
template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize>
_CCCL_TRIVIAL_HOST_DEVICE_API void __fp_custom_reduce(uint64_t& __v) noexcept
{
  constexpr uint16_t __native_exp_size  = __fp_custom_native_sizes<_FpType>::__exp_size;
  constexpr uint16_t __native_mant_size = __fp_custom_native_sizes<_FpType>::__mant_size;

  // === phase 1: exponent range reduction ===
  if constexpr (_ExpSize == fp_custom_dynamic_size || _ExpSize < __native_exp_size)
  {
    const int __exp_size = __fp_custom_exponent_size<_FpType, _ExpSize>();

    /* IEEE 754 double-precision bit layout:
     * [63]    - Sign bit
     * [62:52] - 11-bit exponent (bias 1023)
     * [51:0]  - 52-bit mantissa (implicit leading 1)
     */
    constexpr uint64_t __exp_mask     = 0x7FFULL << 52; // Bits 52-62
    constexpr int64_t __original_bias = 1023; // FP64 exponent bias
    const int64_t __new_bias          = (1LL << (__exp_size - 1)) - 1;
    const int64_t __max_encoded       = (1LL << __exp_size) - 2;

    uint64_t __bits     = __v;
    uint64_t __exp_bits = (__bits & __exp_mask) >> 52;

    /* Infinity and NaN carry an all-ones exponent, which must not be mistaken
     * for a large finite exponent and clamped: that would turn NaN into INF.
     */
    if (__exp_bits == 0x7FF)
    {
      return;
    }

    int64_t __unbiased_exp = static_cast<int64_t>(__exp_bits) - __original_bias;
    int64_t __new_exp_bits = __unbiased_exp + __new_bias;

    /* Check for overflow/underflow in reduced exponent range */
    if (__new_exp_bits > __max_encoded)
    {
      /* Overflow: clamp to FP64 infinity (preserve sign) */
      constexpr uint64_t __sign_mask    = 1ULL << 63;
      constexpr uint64_t __fp64_inf_exp = 0x7FFULL << 52; /* FP64 infinity exponent */
      __v                               = (__bits & __sign_mask) | __fp64_inf_exp;
      return; /* INF doesn't need mantissa reduction */
    }

    if (__new_exp_bits < 1)
    {
      /* Underflow: flush to signed zero */
      constexpr uint64_t __sign_mask = 1ULL << 63;
      __v                            = __bits & __sign_mask;
      return; /* Zero doesn't need mantissa reduction */
    }
    /* Normal range: fall through to mantissa reduction */
  }

  // === phase 2: mantissa precision reduction ===
  if constexpr (_MantSize == fp_custom_dynamic_size || _MantSize < __native_mant_size)
  {
    const int __mant_size = __fp_custom_mantissa_size<_FpType, _MantSize>();

    /* Number of low bits to discard. A runtime size can ask for full precision,
     * and rounding must then be skipped entirely: the masks below would shift
     * by -1. With a fixed size the count is a constant of at least 1, so the
     * guard folds away.
     */
    const int __dropped_mant_size = static_cast<int>(__native_mant_size) - __mant_size;

    // === IEEE 754 round-to-nearest-even (banker's rounding) ===
    /* This is the default rounding mode in IEEE 754 and produces
     * statistically unbiased results for random data.
     *
     * Rules:
     * - If discarded bits > 0.5: round up
     * - If discarded bits < 0.5: round down (truncate)
     * - If discarded bits == 0.5: round to nearest even
     */
    uint64_t __exponent = (__v >> 52) & 0x7FF;
    if (__dropped_mant_size > 0 && __exponent != 0x7FF)
    { /* Skip NaN and Infinity */
      /* __half_mask: bit at position (bits_to_remove - 1), represents 0.5 */
      uint64_t __half_mask = 1ULL << (__dropped_mant_size - 1);
      /* __upper_mask: the two MSBs of the bits being removed */
      uint64_t __upper_mask = __half_mask * 3;
      uint64_t __two_bits   = __v & __upper_mask;

      if (__two_bits & __half_mask)
      {
        /* Discarded value >= 0.5, need to decide between up/down */
        /* If exactly 0.5, round to even; otherwise round up */
        __v += (__two_bits == __half_mask) ? (__half_mask - 1) : __half_mask;
      }
      __v >>= __dropped_mant_size;
      __v <<= __dropped_mant_size;
    }
  }
}

// === main class definition ===
//! @brief Floating-point type with a configurable exponent and mantissa size
//!
//! This class template provides a drop-in replacement for `double` that reduces the
//! precision of every arithmetic operation to the requested field sizes. It stores
//! values using the standard IEEE 754 double-precision format but can simulate lower
//! precisions.
//!
//! ## Features
//! - Implicit conversion from all numeric types
//! - Full operator overloading (+, -, *, /, comparisons)
//! - CUDA host/device compatibility
//! - Zero overhead at native sizes, where the emulation is compiled out
//!
//! ## Memory Layout
//! - Size: 8 bytes (same as double)
//! - Alignment: 8 bytes
//! - Stores raw IEEE 754 bit pattern
//!
//! ## Usage
//! ```cpp
//! using Real = cuda::experimental::fp64_custom<>; // or double for production
//! Real x = 1.5, y = 2.5;
//! Real result = x + y;
//! ```
//!
//! @tparam _FpType Base type holding the value; only `double` is supported today
//! @tparam _ExpSize Exponent bits to preserve, or `fp_custom_dynamic_size`
//! @tparam _MantSize Mantissa bits to preserve, or `fp_custom_dynamic_size`
//!
//! @note The class is trivially copyable and can be used in CUDA kernels
//! @note Instantiations with different sizes are distinct types and do not mix in
//! arithmetic; convert through the base type to combine them.
template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize>
class fp_custom
{
  static_assert(__fp_custom_is_supported_fp_v<_FpType>,
                "cuda::experimental::fp_custom currently supports only _FpType == double (or the bit-identical "
                "_Float64), possible future extension to other base types");
  // An n-bit exponent field reserves the all-ones pattern for infinity and NaN, so it
  // covers 2^n - 2 binades: at least two bits are needed for a single usable one.
  static_assert(!__fp_custom_is_supported_fp_v<_FpType> || _ExpSize == fp_custom_dynamic_size
                  || (_ExpSize >= 2 && _ExpSize <= __fp_custom_native_sizes<_FpType>::__exp_size),
                "cuda::experimental::fp_custom exponent size must be between 2 and the exponent size of the base "
                "type, or fp_custom_dynamic_size");
  // Zero mantissa bits leave only the implicit leading 1, which is a valid request: it
  // rounds every value to the nearest power of two.
  static_assert(!__fp_custom_is_supported_fp_v<_FpType> || _MantSize == fp_custom_dynamic_size
                  || _MantSize <= __fp_custom_native_sizes<_FpType>::__mant_size,
                "cuda::experimental::fp_custom mantissa size must not exceed the mantissa size of the base type, "
                "unless it is fp_custom_dynamic_size");

public:
  // === constructors ===
  //! @brief Default constructor: initializes to zero
  _CCCL_HOST_DEVICE_API constexpr fp_custom() noexcept
      : __bits_{0u}
  {}

  //! @brief Copy constructor, defaulted so the type stays trivially copyable
  //! @note NVCC implicitly makes defaulted special members __host__ __device__
  _CCCL_HIDE_FROM_ABI fp_custom(const fp_custom&) = default;

  // Volatile support: the constructor and assignment operators below cover storage
  // only, i.e. load, store and a bit-preserving round-trip, which is what the legacy
  // pattern of keeping shared-memory scalars in volatile variables needs.
  //
  // A volatile object cannot be an operand of arithmetic or comparison: those take
  // const fp_custom&, and a volatile lvalue never binds to it, not even through the
  // constructor below, because reference-related types are required to bind directly.
  // Copy into a non-volatile local, compute there, store the result back. bit_cast is
  // the exception, deducing volatile fp_custom and performing a volatile load.
  //
  // Each volatile overload is wrapped in a dummy template so that the C++ standard
  // does not consider it a copy constructor or copy assignment operator (a template
  // never is), which preserves trivial copyability.

  //! @brief Copy constructor from volatile
  template <typename _Dummy = void>
  _CCCL_HOST_DEVICE_API fp_custom(const volatile fp_custom& __other) noexcept
      : __bits_{__other.__bits_}
  {}

  //! @brief Construct from double (implicit conversion)
  _CCCL_HOST_DEVICE_API fp_custom(double __d) noexcept
      : __bits_{::cuda::std::bit_cast<uint64_t>(__d)}
  {}

  //! @brief Construct from float (implicit conversion with promotion)
  _CCCL_HOST_DEVICE_API fp_custom(float __f) noexcept
      : __bits_{::cuda::std::bit_cast<uint64_t>(static_cast<double>(__f))}
  {}

  //! @brief Construct from any standard integer type (int / long / long long + unsigned).
  //!  Routes through double, so every width/signedness is handled uniformly and
  //! portably (LP64 and LLP64). Excludes bool / character types, which the
  //! constructor below covers.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_HOST_DEVICE_API fp_custom(_Tp __i) noexcept
      : __bits_{::cuda::std::bit_cast<uint64_t>(static_cast<double>(__i))}
  {}

  //! @brief Construct from bool or a character type
  //!
  //! Excluded from __cccl_is_integer_v, but `1.0 + true` and `1.0 + 'a'` are valid for
  //! double, so mirror that behavior by widening to int and reusing the path above.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND(!::cuda::std::__cccl_is_integer_v<_Tp>))
  _CCCL_HOST_DEVICE_API fp_custom(_Tp __i) noexcept
      : fp_custom(static_cast<int32_t>(__i))
  {}

#if _CCCL_HAS_INT128()
  //! @brief 128-bit integers are deleted: they would silently truncate through double
  _CCCL_HOST_DEVICE_API fp_custom(__int128_t)  = delete;
  _CCCL_HOST_DEVICE_API fp_custom(__uint128_t) = delete;
#endif // _CCCL_HAS_INT128()
#if _CCCL_HAS_FLOAT128()
  //! @brief __float128 is deleted: it would silently lose precision through double,
  //! and makes construction ambiguous with the float / double constructors
  _CCCL_HOST_DEVICE_API fp_custom(__float128) = delete;
#endif // _CCCL_HAS_FLOAT128()

  // === assignment ===
  //! @brief Copy assignment, defaulted so the type stays trivially copyable
  _CCCL_HIDE_FROM_ABI fp_custom& operator=(const fp_custom&) = default;

  //! @brief Assignment to volatile
  //! @note Returns void to avoid the C++20 deprecation of a volatile return type
  template <typename _Dummy = void>
  _CCCL_HOST_DEVICE_API void operator=(const fp_custom& __other) volatile noexcept
  {
    __bits_ = __other.__bits_;
  }

  //! @brief Assignment from volatile
  template <typename _Dummy = void>
  _CCCL_HOST_DEVICE_API fp_custom& operator=(const volatile fp_custom& __other) noexcept
  {
    __bits_ = __other.__bits_;
    return *this;
  }

  //! @brief Assignment from volatile to volatile, e.g. a shared-memory to
  //! shared-memory copy
  template <typename _Dummy = void>
  _CCCL_HOST_DEVICE_API void operator=(const volatile fp_custom& __other) volatile noexcept
  {
    __bits_ = __other.__bits_;
  }

  // === conversions ===
  //! @brief Convert to double (implicit, preserves full precision)
  _CCCL_HOST_DEVICE_API operator double() const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_);
  }

  //! @brief Convert to float (explicit, may lose precision)
  _CCCL_HOST_DEVICE_API explicit operator float() const noexcept
  {
    return static_cast<float>(::cuda::std::bit_cast<double>(__bits_));
  }

  //! @brief Convert to any standard integer type (explicit, truncates toward zero).
  //! Covers int / long / long long + unsigned uniformly; excludes bool / char.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_HOST_DEVICE_API explicit operator _Tp() const noexcept
  {
    return static_cast<_Tp>(::cuda::std::bit_cast<double>(__bits_));
  }

#if _CCCL_HAS_INT128()
  //! @brief See the deleted 128-bit constructors: avoid silent 64-bit truncation
  _CCCL_HOST_DEVICE_API explicit operator __int128_t() const  = delete;
  _CCCL_HOST_DEVICE_API explicit operator __uint128_t() const = delete;
#endif // _CCCL_HAS_INT128()

  // === arithmetic, with precision reduction ===
  //
  // The CUDA intrinsics are called as ::__dadd_rn etc. because this class lives in
  // cuda::experimental, where <cuda/fpemu> declares same-named overloads for its own
  // types: unqualified lookup would stop there and never reach the global scope.

  //! @brief Addition with precision reduction
  //!
  //! Operation flow:
  //! 1. Reduce both operands
  //! 2. Perform native FP64 addition
  //! 3. Reduce the result
  _CCCL_HOST_DEVICE_API fp_custom operator+(const fp_custom& __y) const noexcept
  {
    uint64_t __a = __bits_, __b = __y.__bits_;
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__a);
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__b);
    uint64_t __r{};
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      (__r = ::cuda::std::bit_cast<uint64_t>(
         ::__dadd_rn(::cuda::std::bit_cast<double>(__a), ::cuda::std::bit_cast<double>(__b)));),
      (__r = ::cuda::std::bit_cast<uint64_t>(::cuda::std::bit_cast<double>(__a) + ::cuda::std::bit_cast<double>(__b));))
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__r);
    return fp_custom(::cuda::std::bit_cast<double>(__r));
  }

  //! @brief Subtraction with precision reduction
  _CCCL_HOST_DEVICE_API fp_custom operator-(const fp_custom& __y) const noexcept
  {
    uint64_t __a = __bits_, __b = __y.__bits_;
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__a);
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__b);
    uint64_t __r{};
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      (__r = ::cuda::std::bit_cast<uint64_t>(
         ::__dsub_rn(::cuda::std::bit_cast<double>(__a), ::cuda::std::bit_cast<double>(__b)));),
      (__r = ::cuda::std::bit_cast<uint64_t>(::cuda::std::bit_cast<double>(__a) - ::cuda::std::bit_cast<double>(__b));))
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__r);
    return fp_custom(::cuda::std::bit_cast<double>(__r));
  }

  //! @brief Multiplication with precision reduction
  _CCCL_HOST_DEVICE_API fp_custom operator*(const fp_custom& __y) const noexcept
  {
    uint64_t __a = __bits_, __b = __y.__bits_;
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__a);
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__b);
    uint64_t __r{};
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      (__r = ::cuda::std::bit_cast<uint64_t>(
         ::__dmul_rn(::cuda::std::bit_cast<double>(__a), ::cuda::std::bit_cast<double>(__b)));),
      (__r = ::cuda::std::bit_cast<uint64_t>(::cuda::std::bit_cast<double>(__a) * ::cuda::std::bit_cast<double>(__b));))
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__r);
    return fp_custom(::cuda::std::bit_cast<double>(__r));
  }

  //! @brief Division with precision reduction
  _CCCL_HOST_DEVICE_API fp_custom operator/(const fp_custom& __y) const noexcept
  {
    uint64_t __a = __bits_, __b = __y.__bits_;
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__a);
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__b);
    uint64_t __r{};
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      (__r = ::cuda::std::bit_cast<uint64_t>(
         ::__ddiv_rn(::cuda::std::bit_cast<double>(__a), ::cuda::std::bit_cast<double>(__b)));),
      (__r = ::cuda::std::bit_cast<uint64_t>(::cuda::std::bit_cast<double>(__a) / ::cuda::std::bit_cast<double>(__b));))
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__r);
    return fp_custom(::cuda::std::bit_cast<double>(__r));
  }

  //! @brief Unary negation (sign flip)
  //! @note No precision reduction - just flips the sign bit
  _CCCL_HOST_DEVICE_API fp_custom operator-() const noexcept
  {
    return fp_custom(::cuda::std::bit_cast<double>(__bits_ ^ (1ULL << 63)));
  }

  // === mixed-type arithmetic ===
  //
  // Hidden friends taking one fp_custom and one arithmetic operand, in either order,
  // so that expressions like `x + 2.0` and `3 * x` work. Instantiations with
  // different sizes are deliberately not accepted here.

  //! @brief Mixed-type addition
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend fp_custom operator+(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) + __as_fp_custom(__y);
  }

  //! @brief Mixed-type subtraction
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend fp_custom operator-(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) - __as_fp_custom(__y);
  }

  //! @brief Mixed-type multiplication
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend fp_custom operator*(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) * __as_fp_custom(__y);
  }

  //! @brief Mixed-type division
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend fp_custom operator/(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) / __as_fp_custom(__y);
  }

  // === compound assignment ===
  //! @brief Add and assign
  _CCCL_HOST_DEVICE_API fp_custom& operator+=(const fp_custom& __other) noexcept
  {
    *this = *this + __other;
    return *this;
  }

  //! @brief Subtract and assign
  _CCCL_HOST_DEVICE_API fp_custom& operator-=(const fp_custom& __other) noexcept
  {
    *this = *this - __other;
    return *this;
  }

  //! @brief Multiply and assign
  _CCCL_HOST_DEVICE_API fp_custom& operator*=(const fp_custom& __other) noexcept
  {
    *this = *this * __other;
    return *this;
  }

  //! @brief Divide and assign
  _CCCL_HOST_DEVICE_API fp_custom& operator/=(const fp_custom& __other) noexcept
  {
    *this = *this / __other;
    return *this;
  }

  // === increment and decrement ===
  //! @brief Pre-increment
  _CCCL_HOST_DEVICE_API fp_custom& operator++() noexcept
  {
    *this = *this + fp_custom(1.0);
    return *this;
  }

  //! @brief Pre-decrement
  _CCCL_HOST_DEVICE_API fp_custom& operator--() noexcept
  {
    *this = *this - fp_custom(1.0);
    return *this;
  }

  //! @brief Post-increment
  _CCCL_HOST_DEVICE_API fp_custom operator++(int) noexcept
  {
    fp_custom __temp(*this);
    ++(*this);
    return __temp;
  }

  //! @brief Post-decrement
  _CCCL_HOST_DEVICE_API fp_custom operator--(int) noexcept
  {
    fp_custom __temp(*this);
    --(*this);
    return __temp;
  }

  // === comparison ===
  //! @brief Equality comparison
  _CCCL_HOST_DEVICE_API bool operator==(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_) == ::cuda::std::bit_cast<double>(__y.__bits_);
  }

  //! @brief Inequality comparison
  _CCCL_HOST_DEVICE_API bool operator!=(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_) != ::cuda::std::bit_cast<double>(__y.__bits_);
  }

  //! @brief Less than comparison
  _CCCL_HOST_DEVICE_API bool operator<(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_) < ::cuda::std::bit_cast<double>(__y.__bits_);
  }

  //! @brief Greater than comparison
  _CCCL_HOST_DEVICE_API bool operator>(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_) > ::cuda::std::bit_cast<double>(__y.__bits_);
  }

  //! @brief Less than or equal comparison
  _CCCL_HOST_DEVICE_API bool operator<=(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_) <= ::cuda::std::bit_cast<double>(__y.__bits_);
  }

  //! @brief Greater than or equal comparison
  _CCCL_HOST_DEVICE_API bool operator>=(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_) >= ::cuda::std::bit_cast<double>(__y.__bits_);
  }

  //! @name Mixed-Type Comparisons
  //! @{
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend bool operator==(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) == __as_fp_custom(__y);
  }

  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend bool operator!=(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) != __as_fp_custom(__y);
  }

  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend bool operator<(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) < __as_fp_custom(__y);
  }

  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend bool operator>(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) > __as_fp_custom(__y);
  }

  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend bool operator<=(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) <= __as_fp_custom(__y);
  }

  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend bool operator>=(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) >= __as_fp_custom(__y);
  }
  //! @}

private:
  //! @brief Bring either an fp_custom or an arithmetic operand to fp_custom
  //!
  //! An fp_custom operand is passed through untouched rather than round-tripped
  //! through double, so mixed-type operators cannot perturb the bit pattern.
  _CCCL_TEMPLATE(typename _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp, fp_custom>)
  [[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API static const fp_custom& __as_fp_custom(const _Tp& __x) noexcept
  {
    return __x;
  }

  _CCCL_TEMPLATE(typename _Tp)
  _CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
  [[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API static fp_custom __as_fp_custom(const _Tp& __x) noexcept
  {
    return fp_custom(static_cast<double>(__x));
  }

  //! @brief Raw IEEE 754 bit representation of the value
  //!
  //! Private, with no raw-bits accessor and no raw-bits constructor: the value is
  //! bit-identical to the base type, so `bit_cast` through `double` is the way to
  //! reinterpret a bit pattern as an fp_custom and back.
  uint64_t __bits_;
};

// === math functions ===
//! @brief Square root with precision reduction
//!
//! @param __x Input value
//! @return Square root of x, with the operand and result reduced
//!
//! @note Uses __dsqrt_rn intrinsic on CUDA, ::sqrt on host
template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fp_custom<_FpType, _ExpSize, _MantSize>
sqrt(const fp_custom<_FpType, _ExpSize, _MantSize>& __x) noexcept
{
  uint64_t __a = ::cuda::std::bit_cast<uint64_t>(static_cast<double>(__x));
  __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__a);
  uint64_t __r{};
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (__r = ::cuda::std::bit_cast<uint64_t>(::__dsqrt_rn(::cuda::std::bit_cast<double>(__a)));),
                    (__r = ::cuda::std::bit_cast<uint64_t>(::sqrt(::cuda::std::bit_cast<double>(__a)));))
  __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__r);
  return fp_custom<_FpType, _ExpSize, _MantSize>(::cuda::std::bit_cast<double>(__r));
}

//! @brief Fused multiply-add with precision reduction
//!
//! Computes (x * y) + z with a single rounding operation.
//!
//! @param __x First multiplicand
//! @param __y Second multiplicand
//! @param __z Addend
//! @return (x * y) + z, with all operands and the result reduced
//!
//! @note Uses __fma_rn intrinsic on CUDA, ::fma on host
template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fp_custom<_FpType, _ExpSize, _MantSize>
fma(const fp_custom<_FpType, _ExpSize, _MantSize>& __x,
    const fp_custom<_FpType, _ExpSize, _MantSize>& __y,
    const fp_custom<_FpType, _ExpSize, _MantSize>& __z) noexcept
{
  uint64_t __a = ::cuda::std::bit_cast<uint64_t>(static_cast<double>(__x));
  uint64_t __b = ::cuda::std::bit_cast<uint64_t>(static_cast<double>(__y));
  uint64_t __c = ::cuda::std::bit_cast<uint64_t>(static_cast<double>(__z));
  __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__a);
  __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__b);
  __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__c);
  uint64_t __r{};
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (__r = ::cuda::std::bit_cast<uint64_t>(::__fma_rn(
       ::cuda::std::bit_cast<double>(__a), ::cuda::std::bit_cast<double>(__b), ::cuda::std::bit_cast<double>(__c)));),
    (__r = ::cuda::std::bit_cast<uint64_t>(::fma(
       ::cuda::std::bit_cast<double>(__a), ::cuda::std::bit_cast<double>(__b), ::cuda::std::bit_cast<double>(__c)));))
  __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__r);
  return fp_custom<_FpType, _ExpSize, _MantSize>(::cuda::std::bit_cast<double>(__r));
}

// Trait machinery for the mixed-operand fma below. __is_fp_custom_v detects an
// fp_custom specialization; __has_fp_custom_v is the constraint "at least one operand
// is an fp_custom", which leaves a pure-arithmetic call to the built-in types and, by
// partial ordering, a pure fp_custom call to the more specialized exact-type overload
// above; __fp_custom_pick_t selects the fp_custom type among a set of operands.
//
// The constraint deliberately admits operands of two different fp_custom
// instantiations, which sizes never mix implicitly. Rejecting them here would only
// send the call to ::fma(double, double, double) through the implicit conversion and
// compute at full FP64 precision without a word, so the mix is diagnosed by the
// static_assert in the body instead.
template <class _Tp>
inline constexpr bool __is_fp_custom_v = false;
template <class _FpType, uint16_t _ExpSize, uint16_t _MantSize>
inline constexpr bool __is_fp_custom_v<fp_custom<_FpType, _ExpSize, _MantSize>> = true;

template <class... _Ts>
inline constexpr bool __has_fp_custom_v = (__is_fp_custom_v<_Ts> || ...);

template <class... _Ts>
struct __fp_custom_pick
{
  using type = void;
};
template <class _T0, class... _Ts>
struct __fp_custom_pick<_T0, _Ts...>
{
  using type = ::cuda::std::conditional_t<__is_fp_custom_v<_T0>, _T0, typename __fp_custom_pick<_Ts...>::type>;
};
template <class... _Ts>
using __fp_custom_pick_t = typename __fp_custom_pick<_Ts...>::type;

template <class _Tp, class... _Ts>
inline constexpr bool __fp_custom_same_or_arithmetic_v =
  ((::cuda::std::is_same_v<_Ts, _Tp> || ::cuda::std::is_arithmetic_v<_Ts>) && ...);

//! @brief Fused multiply-add mixing fp_custom with arithmetic operands
//!
//! Without this overload such a call would resolve to `::fma(double, double, double)`
//! through the implicit conversion and quietly compute at full FP64 precision.
_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(__has_fp_custom_v<_T1, _T2, _T3>)
[[nodiscard]] _CCCL_HOST_DEVICE_API inline __fp_custom_pick_t<_T1, _T2, _T3>
fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using _Tp = __fp_custom_pick_t<_T1, _T2, _T3>;
  static_assert(__fp_custom_same_or_arithmetic_v<_Tp, _T1, _T2, _T3>,
                "fma operands mix two different fp_custom instantiations: exponent and mantissa sizes never "
                "convert implicitly, so convert explicitly to the intended type first");
  return fma(_Tp(__x), _Tp(__y), _Tp(__z));
}

// === type aliases ===
//! @brief fp_custom over double, with the native FP64 field sizes as defaults
//!
//! The type to reach for when emulating a format inside FP64: `fp64_custom<>` is a
//! drop-in for `double` with the precision reduction compiled out entirely, and the
//! sizes of a narrower format are given as `fp64_custom<8, 23>`, or left to runtime
//! with `fp_custom_dynamic_size`.
//!
//! @note Being an alias template, it always needs the angle brackets: `fp64_custom<> x;`
template <uint16_t _ExpSize = 11, uint16_t _MantSize = 52>
using fp64_custom = fp_custom<double, _ExpSize, _MantSize>;
} // namespace cuda::experimental

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Overloads of sqrt and fma for fp_custom so the standard spelling cuda::std::sqrt /
// cuda::std::fma selects the reducing implementation. A qualified call suppresses ADL,
// so without these it would silently narrow fp_custom -> double through the implicit
// conversion and compute at full FP64 precision, which is exactly what a precision
// study must not do. These forward to cuda::experimental::sqrt / fma, which unqualified
// and ADL calls already resolve to. The exact-type fma overload wins for pure fp_custom
// calls by partial ordering, while the constrained one handles the mixed case.
template <class _FpType, uint16_t _ExpSize, uint16_t _MantSize>
[[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>
sqrt(const ::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>& __x) noexcept
{
  return ::cuda::experimental::sqrt(__x);
}

template <class _FpType, uint16_t _ExpSize, uint16_t _MantSize>
[[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>
fma(const ::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>& __x,
    const ::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>& __y,
    const ::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>& __z) noexcept
{
  return ::cuda::experimental::fma(__x, __y, __z);
}

_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(::cuda::experimental::__has_fp_custom_v<_T1, _T2, _T3>)
[[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::experimental::__fp_custom_pick_t<_T1, _T2, _T3>
fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  return ::cuda::experimental::fma(__x, __y, __z);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPTOOL_CUSTOM_H
