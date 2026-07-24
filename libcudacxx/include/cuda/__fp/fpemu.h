//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_H
#define _CUDA___FP_FPEMU_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
//! @file fpemu.h
//! @brief Main header file for the FPEMU floating point scalar emulation library
//!
//! This is the main header file that provides access to the complete FPEMU library.
//! It includes all the necessary headers for:
//!
//! - Core definitions, macros and enumerations (fpemu_common.h)
//! - Class templates (fpemu, fpemu_unpacked)
//! - Public API functions (operators, builtins, conversions)
//! - Implementation files for specific scalar operations:
//!   - Comparison operations (fpemu_impl_cmp.h)
//!   - Type conversions (fpemu_impl_cvt.h)
//!   - Fused multiply-add (fpemu_impl_fma.h)
//!   - Addition (fpemu_impl_add.h)
//!   - Subtraction (fpemu_impl_sub.h)
//!   - Multiplication (fpemu_impl_mul.h)
//!   - Division (fpemu_impl_div.h)
//!   - Square root (fpemu_impl_sqrt.h)
//!   - Other operations (fpemu_impl_others.h)
//!
//! The library provides IEEE-754 compliant emulated scalar floating point operations
//! with configurable rounding modes and computation methods.
//!
//! Accuracy levels (template parameter 'fpemu_accuracy'):
//!   - fpemu_accuracy::high — correctly rounded, full IEEE-754 range including
//!                        infinities, NaNs, and subnormals
//!   - fpemu_accuracy::mid  — up to 1-2 least significant mantissa bits of error,
//!                        limited INF, NaN and subnormal support
//!   - fpemu_accuracy::low  — up to half of the mantissa bits may be lost,
//!                        limited INF, NaN and subnormal support
//!   - fpemu_accuracy::def  — default selector; equals high (IEEE-correct)
//!
//! The API supports both host and device code through appropriate decorators and
//! can utilize different computational backends based on template parameters.

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cstdint>

// Public API surface (fpemu_accuracy selector + CCCL_FPEMU_LIB / CCCL_FPEMU_INLINE
// compile-mode knobs) lives in fpemu_common.h; all library-internal machinery
// (vocabulary types, decorator/ABI/declaration macros, helper functions) lives in
// fpemu_impl.h. The class below stores raw __fpbits64 bits, so it needs both.
#include <cuda/__fp/fpemu_common.h>
#include <cuda/__fp/fpemu_impl.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// The public accuracy selector fpemu_accuracy is defined in
// <cuda/__fp/fpemu_common.h> (the public API header); the internal vocabulary
// types (__fpbits64 / __fpbits64_unpacked) and helpers come from
// <cuda/__fp/fpemu_impl.h>. Both are included above so the class can store raw
// bits while keeping every FP header self-contained.

// Forward declaration of unpacked floating-point class
template <typename _FpType, fpemu_accuracy _Met>
class fpemu_unpacked;

// Underlying element types accepted by the emulated classes. Only double is
// implemented, but C++23's _Float64 (the type behind std::float64_t) is a
// *distinct* type from double even though it is bit-identical, so accept it too
// where the implementation provides it. The standard feature-test macro
// __STDCPP_FLOAT64_T__ both guards the _Float64 token and guarantees the type is
// available (so no compiler version table is needed); where _Float64 is merely an
// alias for double (pre-C++23 GCC/clang) the double term below already covers it.
template <typename _Tp>
inline constexpr bool __fpemu_is_supported_fp_v =
  ::cuda::std::is_same_v<_Tp, double>
#if defined(__STDCPP_FLOAT64_T__) && (__STDCPP_FLOAT64_T__ == 1)
  || ::cuda::std::is_same_v<_Tp, _Float64>
#endif // __STDCPP_FLOAT64_T__
  ;

//! @brief Primary emulated double-precision floating-point class template
//!
//! The fpemu class template represents a double-precision (64-bit)
//! floating-point number, emulated according to IEEE-754 semantics but with
//! configurable accuracy level.
//!
//! @tparam met Accuracy level (fpemu_accuracy::high, mid, low; def == high)
//!              - high: Correctly rounded with full IEEE-754 range
//!              - mid: 1-2 LSB error with normal range
//!              - low: Low accuracy with normal range
//!
//! This class provides:
//!   - Storage of the value as __fpbits64 (raw IEEE-754 format)
//!   - Construction from and conversion to standard C++ types (int, float, double)
//!   - Arithmetic operators and mathematical functions
//!   - Fine-grained control over rounding and accuracy level
//!   - Portable host/device compatibility (CUDA/HIP/etc)
//!
//! Usage:
//!   fpemu<double, fpemu_accuracy::high> x{1.5};
//!   fpemu<double> y = x + 2.0;
//!   double z = static_cast<double>(y);
template <typename _FpType = double, fpemu_accuracy _Met = fpemu_accuracy::def>
class fpemu
{
public:
  // Only double emulation is implemented today; the _FpType axis exists for future
  // extension. _Float64 is accepted as a bit-identical alias for double (see
  // __fpemu_is_supported_fp_v).
  static_assert(__fpemu_is_supported_fp_v<_FpType>,
                "cuda::experimental::fpemu currently supports only _FpType == double (or the bit-identical _Float64), "
                "possible future extension to other types emulation");

private:
  // Internal representation of the floating-point value (__fpbits64 is defined in
  // fpemu_common.h). Private: fpemu<double> is trivially copyable and bit-identical
  // to its 64-bit IEEE-754 representation, so use bit_cast to reinterpret it; no
  // raw-bits accessor is provided.
  __fpbits64 __bits_;

public:
  /*
  // Constructors and assignment operators
  */
  // Basic constructors
  _CCCL_API constexpr fpemu() noexcept
      : __bits_{0u}
  {}
  /*
  // Defaulted copy constructor (trivially copyable)
  // Note: NVCC implicitly makes defaulted special members __host__ __device__
  */
  fpemu(const fpemu& __other) = default;

  /*
  // Copy constructor from volatile fpemu
  // Template so it is NOT a copy constructor per the C++ standard.
  // The volatile overloads are wrapped in dummy templates
  // so that the C++ standard does not consider them copy constructors/assignment
  // operators (a template is never a copy constructor or copy assignment operator),
  // preserving trivial copyability while retaining volatile access support.
  */
  template <typename _Dummy = void>
  _CCCL_API fpemu(const volatile fpemu& __other) noexcept
      : __bits_{__other.__bits_}
  {}

  // Defaulted copy assignment operator (trivially copyable)
  fpemu& operator=(const fpemu& __other) = default;

  /*
  // Assignment operator to volatile fpemu
  // Template so it is NOT a copy assignment operator per the C++ standard
  // Returns void to avoid C++20 -Wvolatile (deprecated volatile return)
  */
  template <typename _Dummy = void>
  _CCCL_API void operator=(const fpemu& __other) volatile noexcept
  {
    __bits_ = __other.__bits_;
  }

  /*
  // Assignment operator from volatile fpemu
  // Template so it is NOT a copy assignment operator per the C++ standard
  */
  template <typename _Dummy = void>
  _CCCL_API fpemu& operator=(const volatile fpemu& __other) noexcept
  {
    __bits_ = __other.__bits_;
    return *this;
  }

  /*
  // Conversion operators
  */
  // ==== Conversions from other types to fpemu:
  // Implicit conversions from floating-point types
  _CCCL_API fpemu(float __f) noexcept;
  _CCCL_API fpemu(double __d) noexcept;
  // Construction from any standard integer type (int / long / long long + unsigned).
  // The value is canonicalized to the accuracy-correct 32- or 64-bit builtin: the
  // target width comes from __num_bits_v and the signedness-correct fixed-width type
  // from __make_nbit_int_t, so the static_cast selects the matching overloaded setter
  // (signed vs unsigned) below. All widths are implicit, mirroring the implicit
  // float/double ctors and the IEEE-754 `long -> double` conversion (64-bit values may
  // lose precision). bool / character types are excluded by __cccl_is_integer_v.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_API fpemu(_Tp __i) noexcept
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
  // and `1.0 + 'a'` are valid for double, so mirror that behavior: widen the value to
  // int32 and reuse the int32 constructor path (no dedicated char/bool handling).
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND(!::cuda::std::__cccl_is_integer_v<_Tp>))
  _CCCL_API fpemu(_Tp __i) noexcept
      : fpemu(static_cast<int32_t>(__i))
  {}
#if _CCCL_HAS_INT128()
  // 128-bit integers would silently truncate to 64 bits, so they are deleted until
  // real 128-bit support is added (tracking issue: extended-precision fp <-> __int128).
  _CCCL_API fpemu(__int128_t)  = delete;
  _CCCL_API fpemu(__uint128_t) = delete;
#endif // _CCCL_HAS_INT128()
#if _CCCL_HAS_FLOAT128()
  // __float128 -> double would silently lose precision (and today makes construction
  // ambiguous with the float/double ctors), so it is deleted for parity with the
  // 128-bit integer ctors until real extended-precision support exists.
  _CCCL_API fpemu(__float128) = delete;
#endif // _CCCL_HAS_FLOAT128()
  // Converting constructor from another accuracy (same packed representation, so a
  // pure reinterpretation). Explicit: an accuracy change must be opted into via
  // direct-init / static_cast, mirroring fpmp2 and the IEEE-754 narrowing ctors.
  template <fpemu_accuracy _Acc2>
  _CCCL_API explicit fpemu(const fpemu<double, _Acc2>& __src) noexcept;
  // Converting constructor from the unpacked representation (packs to the 64-bit form).
  template <fpemu_accuracy _Acc2>
  _CCCL_API explicit fpemu(const fpemu_unpacked<double, _Acc2>& __src) noexcept;

  // ==== Conversion from fpemu to other types:
  // Implicit conversion to double
  _CCCL_API operator double() const noexcept;
  // Explicit conversions to other types
  _CCCL_API explicit operator float() const noexcept;
  // Explicit conversion to any standard integer type (int / long / long long + unsigned).
  // The target width comes from __num_bits_v and the signedness-correct fixed-width type
  // from __make_nbit_int_t, selecting the matching overloaded __to_integer helper below;
  // excludes bool / character types.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_API explicit operator _Tp() const noexcept
  {
    using _Up =
      ::cuda::std::__make_nbit_int_t<(::cuda::std::__num_bits_v<_Tp> <= 32) ? 32 : 64, ::cuda::std::is_signed_v<_Tp>>;
    return static_cast<_Tp>(__to_integer(_Up{}));
  }
#if _CCCL_HAS_INT128()
  // See the deleted 128-bit constructors above: avoid silent 64-bit truncation.
  _CCCL_API explicit operator __int128_t() const  = delete;
  _CCCL_API explicit operator __uint128_t() const = delete;
#endif // _CCCL_HAS_INT128()

private:
  // Accuracy-correct integer <-> value helpers (defined out-of-line where the fpemu
  // builtins are visible). Kept non-template so the definitions stay out-of-line.
  _CCCL_API void __set_from_int32(int32_t __i) noexcept;
  _CCCL_API void __set_from_int32(uint32_t __i) noexcept;
  _CCCL_API void __set_from_int64(int64_t __i) noexcept;
  _CCCL_API void __set_from_int64(uint64_t __i) noexcept;
  _CCCL_API int32_t __to_integer(int32_t) const noexcept;
  _CCCL_API uint32_t __to_integer(uint32_t) const noexcept;
  _CCCL_API int64_t __to_integer(int64_t) const noexcept;
  _CCCL_API uint64_t __to_integer(uint64_t) const noexcept;

public:
  /*
  // Arithmetic operations:
  */
  // === mul ===
  // (*)
  template <fpemu_accuracy _Acc>
  _CCCL_API friend fpemu<double, _Acc>
  operator*(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept;
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu> || ::cuda::std::is_same_v<_T2, fpemu>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend fpemu operator*(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu(__x) * fpemu(__y);
  }

  // === div ===
  // (/)
  template <fpemu_accuracy _Acc>
  _CCCL_API friend fpemu<double, _Acc>
  operator/(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept;
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu> || ::cuda::std::is_same_v<_T2, fpemu>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend fpemu operator/(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu(__x) / fpemu(__y);
  }

  // === add ===
  // (+)
  template <fpemu_accuracy _Acc>
  _CCCL_API friend fpemu<double, _Acc>
  operator+(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept;
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu> || ::cuda::std::is_same_v<_T2, fpemu>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend fpemu operator+(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu(__x) + fpemu(__y);
  }

  // === sub ===
  // (-)
  template <fpemu_accuracy _Acc>
  _CCCL_API friend fpemu<double, _Acc>
  operator-(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept;
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu> || ::cuda::std::is_same_v<_T2, fpemu>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend fpemu operator-(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu(__x) - fpemu(__y);
  }

  // Prefix increment/decrement
  _CCCL_API fpemu& operator++() noexcept
  {
    this = this + fpemu(1.0);
    return *this;
  }
  _CCCL_API fpemu& operator--() noexcept
  {
    this = this - fpemu(1.0);
    return *this;
  }
  // Postfix increment/decrement
  _CCCL_API fpemu operator++(int) noexcept
  {
    fpemu __temp(*this);
    this = this + fpemu(1.0);
    return __temp;
  }
  _CCCL_API fpemu operator--(int) noexcept
  {
    fpemu __temp(*this);
    this = this - fpemu(1.0);
    return __temp;
  }
  // Compound assignment operators
  _CCCL_API fpemu& operator+=(const fpemu& __other) noexcept
  {
    *this = *this + __other;
    return *this;
  }
  _CCCL_API fpemu& operator-=(const fpemu& __other) noexcept
  {
    *this = *this - __other;
    return *this;
  }
  _CCCL_API fpemu& operator*=(const fpemu& __other) noexcept
  {
    *this = *this * __other;
    return *this;
  }
  _CCCL_API fpemu& operator/=(const fpemu& __other) noexcept
  {
    *this = *this / __other;
    return *this;
  }
  // Unary negation operator (implementation in fpemu_impl_others.h)
  _CCCL_API fpemu operator-() const noexcept;

  /*
  // Comparison operators:
  */
  // equality (==)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu> || ::cuda::std::is_same_v<_T2, fpemu>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend bool operator==(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu(__x) == fpemu(__y);
  }
  // inequality (!=)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu> || ::cuda::std::is_same_v<_T2, fpemu>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend bool operator!=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu(__x) != fpemu(__y);
  }
  // less than (<)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu> || ::cuda::std::is_same_v<_T2, fpemu>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend bool operator<(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu(__x) < fpemu(__y);
  }
  // greater than (>)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu> || ::cuda::std::is_same_v<_T2, fpemu>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend bool operator>(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu(__x) > fpemu(__y);
  }
  // less than or equal to (<=)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu> || ::cuda::std::is_same_v<_T2, fpemu>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend bool operator<=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu(__x) <= fpemu(__y);
  }
  // greater than or equal to (>=)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu> || ::cuda::std::is_same_v<_T2, fpemu>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend bool operator>=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu(__x) >= fpemu(__y);
  }
}; // class fpemu

//! @brief Unpacked emulated double-precision floating-point class template
//!
//! The fpemu_unpacked class template represents a double-precision (64-bit)
//! floating-point number in a decomposed (sign / exponent / mantissa) form,
//! emulated according to IEEE-754 semantics but with configurable accuracy level.
//! It trades the compact packed layout of fpemu for direct field access, which the
//! emulation builtins use to avoid repeated pack/unpack work in chained operations.
//!
//! @tparam met Accuracy level (fpemu_accuracy::high, mid, low; def == high)
//!              - high: Correctly rounded with full IEEE-754 range
//!              - mid: 1-2 LSB error with normal range
//!              - low: Low accuracy with normal range
//!
//! This class provides:
//!   - Storage of the value as __fpbits64_unpacked (sign, exponent, mantissa)
//!   - Construction from and conversion to standard C++ types (int, float, double)
//!   - Arithmetic operators and mathematical functions
//!   - Fine-grained control over rounding and accuracy level
//!   - Portable host/device compatibility (CUDA/HIP/etc)
//!
//! Usage:
//!   fpemu_unpacked<double, fpemu_accuracy::high> x{1.5};
//!   fpemu_unpacked<double> y = x + 2.0;
//!   double z = static_cast<double>(y);
template <typename _FpType = double, fpemu_accuracy _Met = fpemu_accuracy::def>
class fpemu_unpacked
{
public:
  // Only double emulation is implemented today; the _FpType axis exists for future
  // extension. _Float64 is accepted as a bit-identical alias for double (see
  // __fpemu_is_supported_fp_v).
  static_assert(__fpemu_is_supported_fp_v<_FpType>,
                "cuda::experimental::fpemu_unpacked currently supports only _FpType == double (or the bit-identical "
                "_Float64)");

private:
  // Internal representation of the unpacked floating-point value (__fpbits64_unpacked
  // is defined in fpemu_common.h). Private: fpemu_unpacked<double> is trivially
  // copyable and bit-identical to its __fpbits64_unpacked representation, so use
  // bit_cast to reinterpret it; no raw-bits accessor is provided.
  __fpbits64_unpacked __bits_;

public:
  /*
  // Constructors and assignment operators
  */
  // Basic constructors
  _CCCL_API constexpr fpemu_unpacked() noexcept
      : __bits_{0u, 0, 0}
  {}
  /*
  // Defaulted copy constructor (trivially copyable)
  // Note: NVCC implicitly makes defaulted special members __host__ __device__
  */
  fpemu_unpacked(const fpemu_unpacked& __other) = default;

  /*
  // Copy constructor from volatile fpemu_unpacked
  // Template so it is NOT a copy constructor per the C++ standard.
  // The volatile overloads are wrapped in dummy templates
  // so that the C++ standard does not consider them copy constructors/assignment
  // operators (a template is never a copy constructor or copy assignment operator),
  // preserving trivial copyability while retaining volatile access support.
  */
  template <typename _Dummy = void>
  _CCCL_API fpemu_unpacked(const volatile fpemu_unpacked& __other) noexcept
  {
    __bits_.sign     = __other.__bits_.sign;
    __bits_.exponent = __other.__bits_.exponent;
    __bits_.mantissa = __other.__bits_.mantissa;
  }

  // Defaulted copy assignment operator (trivially copyable)
  fpemu_unpacked& operator=(const fpemu_unpacked& __other) = default;

  /*
  // Assignment operator to volatile fpemu_unpacked
  // Template so it is NOT a copy assignment operator per the C++ standard
  // Returns void to avoid C++20 -Wvolatile (deprecated volatile return)
  */
  template <typename _Dummy = void>
  _CCCL_API void operator=(const fpemu_unpacked& __other) volatile noexcept
  {
    __bits_.sign     = __other.__bits_.sign;
    __bits_.exponent = __other.__bits_.exponent;
    __bits_.mantissa = __other.__bits_.mantissa;
  }

  /*
  // Assignment operator from volatile fpemu_unpacked
  // Template so it is NOT a copy assignment operator per the C++ standard
  */
  template <typename _Dummy = void>
  _CCCL_API fpemu_unpacked& operator=(const volatile fpemu_unpacked& __other) noexcept
  {
    __bits_.sign     = __other.__bits_.sign;
    __bits_.exponent = __other.__bits_.exponent;
    __bits_.mantissa = __other.__bits_.mantissa;
    return *this;
  }
  /*
  // Conversion operators
  */
  // ==== Conversions from other types to fpemu_unpacked:
#if defined __CUDACC__
  // Implicit conversions from floating-point types
  _CCCL_API fpemu_unpacked(float f) noexcept;
  _CCCL_API fpemu_unpacked(double d) noexcept;
#  define _CCCL_FPEMU_UNP_NARROW_EXPLICIT
#else
  // Explicit conversions from floating-point types (to avoid ambiguity with packed type)
  _CCCL_API explicit fpemu_unpacked(float __f) noexcept;
  _CCCL_API explicit fpemu_unpacked(double __d) noexcept;
#  define _CCCL_FPEMU_UNP_NARROW_EXPLICIT explicit
#endif
  // Construction from any standard integer type (int / long / long long + unsigned).
  // The value is canonicalized to the accuracy-correct 32- or 64-bit builtin: the target
  // width comes from __num_bits_v and the signedness-correct fixed-width type from
  // __make_nbit_int_t, so the static_cast selects the matching overloaded setter (signed
  // vs unsigned) below. Explicitness follows the surrounding float/double ctors (implicit
  // on device, explicit on host) to avoid ambiguity with the packed type; 64-bit values
  // may lose precision. bool / character types are excluded by __cccl_is_integer_v.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_API _CCCL_FPEMU_UNP_NARROW_EXPLICIT fpemu_unpacked(_Tp __i) noexcept
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
  // and `1.0 + 'a'` are valid for double, so mirror that behavior: widen the value to
  // int32 and reuse the int32 constructor path (no dedicated char/bool handling).
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND(!::cuda::std::__cccl_is_integer_v<_Tp>))
  _CCCL_API _CCCL_FPEMU_UNP_NARROW_EXPLICIT fpemu_unpacked(_Tp __i) noexcept
      : fpemu_unpacked(static_cast<int32_t>(__i))
  {}
#if _CCCL_HAS_INT128()
  // 128-bit integers would silently truncate to 64 bits, so they are deleted until
  // real 128-bit support is added (tracking issue: extended-precision fp <-> __int128).
  // Mirror the integer ctor's explicitness so copy-init overload sets are unchanged.
  _CCCL_API _CCCL_FPEMU_UNP_NARROW_EXPLICIT fpemu_unpacked(__int128_t)  = delete;
  _CCCL_API _CCCL_FPEMU_UNP_NARROW_EXPLICIT fpemu_unpacked(__uint128_t) = delete;
#endif // _CCCL_HAS_INT128()
#if _CCCL_HAS_FLOAT128()
  // __float128 -> double would silently lose precision (and today makes construction
  // ambiguous with the float/double ctors), so it is deleted for parity with the
  // 128-bit integer ctors until real extended-precision support exists.
  // Mirror the integer ctor's explicitness so copy-init overload sets are unchanged.
  _CCCL_API _CCCL_FPEMU_UNP_NARROW_EXPLICIT fpemu_unpacked(__float128) = delete;
#endif // _CCCL_HAS_FLOAT128()
#undef _CCCL_FPEMU_UNP_NARROW_EXPLICIT
  // Converting constructor from another accuracy (same unpacked representation, so a
  // pure reinterpretation). Explicit for the same reason as the packed class.
  template <fpemu_accuracy _Acc2>
  _CCCL_API explicit fpemu_unpacked(const fpemu_unpacked<double, _Acc2>& __src) noexcept;
  // Converting constructor from the packed representation (unpacks the 64-bit form).
  template <fpemu_accuracy _Acc2>
  _CCCL_API explicit fpemu_unpacked(const fpemu<double, _Acc2>& __src) noexcept;

  // ==== Conversion from fpemu_unpacked to other types:
  // Implicit conversion to double
  _CCCL_API operator double() const noexcept;
  // Explicit conversions to other types
  _CCCL_API explicit operator float() const noexcept;
  // Explicit conversion to any standard integer type (int / long / long long + unsigned).
  // The target width comes from __num_bits_v and the signedness-correct fixed-width type
  // from __make_nbit_int_t, selecting the matching overloaded __to_integer helper below;
  // excludes bool / character types.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_API explicit operator _Tp() const noexcept
  {
    using _Up =
      ::cuda::std::__make_nbit_int_t<(::cuda::std::__num_bits_v<_Tp> <= 32) ? 32 : 64, ::cuda::std::is_signed_v<_Tp>>;
    return static_cast<_Tp>(__to_integer(_Up{}));
  }
#if _CCCL_HAS_INT128()
  // See the deleted 128-bit constructors above: avoid silent 64-bit truncation.
  _CCCL_API explicit operator __int128_t() const  = delete;
  _CCCL_API explicit operator __uint128_t() const = delete;
#endif // _CCCL_HAS_INT128()

private:
  // Accuracy-correct integer <-> value helpers (defined out-of-line where the fpemu
  // builtins are visible). Kept non-template so the definitions stay out-of-line.
  _CCCL_API void __set_from_int32(int32_t __i) noexcept;
  _CCCL_API void __set_from_int32(uint32_t __i) noexcept;
  _CCCL_API void __set_from_int64(int64_t __i) noexcept;
  _CCCL_API void __set_from_int64(uint64_t __i) noexcept;
  _CCCL_API int32_t __to_integer(int32_t) const noexcept;
  _CCCL_API uint32_t __to_integer(uint32_t) const noexcept;
  _CCCL_API int64_t __to_integer(int64_t) const noexcept;
  _CCCL_API uint64_t __to_integer(uint64_t) const noexcept;

public:
  /*
  // Arithmetic operations:
  */
  // === mul ===
  // (*)
  template <fpemu_accuracy _Acc>
  _CCCL_API friend fpemu_unpacked<double, _Acc>
  operator*(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept;
  // (/)
  template <fpemu_accuracy _Acc>
  _CCCL_API friend fpemu_unpacked<double, _Acc>
  operator/(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept;
  // (+)
  template <fpemu_accuracy _Acc>
  _CCCL_API friend fpemu_unpacked<double, _Acc>
  operator+(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept;
  // (-)
  template <fpemu_accuracy _Acc>
  _CCCL_API friend fpemu_unpacked<double, _Acc>
  operator-(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept;

  // == mul ==
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu_unpacked> || ::cuda::std::is_same_v<_T2, fpemu_unpacked>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend fpemu_unpacked operator*(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu_unpacked(__x) * fpemu_unpacked(__y);
  }

  // === div ===
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu_unpacked> || ::cuda::std::is_same_v<_T2, fpemu_unpacked>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend fpemu_unpacked operator/(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu_unpacked(__x) / fpemu_unpacked(__y);
  }

  // === add ===
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu_unpacked> || ::cuda::std::is_same_v<_T2, fpemu_unpacked>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend fpemu_unpacked operator+(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu_unpacked(__x) + fpemu_unpacked(__y);
  }

  // === sub ===
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu_unpacked> || ::cuda::std::is_same_v<_T2, fpemu_unpacked>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend fpemu_unpacked operator-(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu_unpacked(__x) - fpemu_unpacked(__y);
  }

  // Prefix increment/decrement
  _CCCL_API fpemu_unpacked& operator++() noexcept
  {
    this = this + fpemu_unpacked(1.0);
    return *this;
  }
  _CCCL_API fpemu_unpacked& operator--() noexcept
  {
    this = this - fpemu_unpacked(1.0);
    return *this;
  }
  // Postfix increment/decrement
  _CCCL_API fpemu_unpacked operator++(int) noexcept
  {
    fpemu_unpacked __temp(*this);
    this = this + fpemu_unpacked(1.0);
    return __temp;
  }
  _CCCL_API fpemu_unpacked operator--(int) noexcept
  {
    fpemu_unpacked __temp(*this);
    this = this - fpemu_unpacked(1.0);
    return __temp;
  }
  // Compound assignment operators
  _CCCL_API fpemu_unpacked& operator+=(const fpemu_unpacked& __other) noexcept
  {
    *this = *this + __other;
    return *this;
  }
  _CCCL_API fpemu_unpacked& operator-=(const fpemu_unpacked& __other) noexcept
  {
    *this = *this - __other;
    return *this;
  }
  _CCCL_API fpemu_unpacked& operator*=(const fpemu_unpacked& __other) noexcept
  {
    *this = *this * __other;
    return *this;
  }
  _CCCL_API fpemu_unpacked& operator/=(const fpemu_unpacked& __other) noexcept
  {
    *this = *this / __other;
    return *this;
  }
  // Unary negation operator (implementation in fpemu_impl_others.h)
  _CCCL_API fpemu_unpacked operator-() const noexcept;

  /*
  // Comparison operators:
  */
  // equality (==)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu_unpacked> || ::cuda::std::is_same_v<_T2, fpemu_unpacked>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend bool operator==(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu_unpacked(__x) == fpemu_unpacked(__y);
  }
  // inequality (!=)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu_unpacked> || ::cuda::std::is_same_v<_T2, fpemu_unpacked>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend bool operator!=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu_unpacked(__x) != fpemu_unpacked(__y);
  }
  // less than (<)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu_unpacked> || ::cuda::std::is_same_v<_T2, fpemu_unpacked>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend bool operator<(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu_unpacked(__x) < fpemu_unpacked(__y);
  }
  // greater than (>)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu_unpacked> || ::cuda::std::is_same_v<_T2, fpemu_unpacked>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend bool operator>(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu_unpacked(__x) > fpemu_unpacked(__y);
  }
  // less than or equal to (<=)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu_unpacked> || ::cuda::std::is_same_v<_T2, fpemu_unpacked>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend bool operator<=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu_unpacked(__x) <= fpemu_unpacked(__y);
  }
  // greater than or equal to (>=)
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpemu_unpacked> || ::cuda::std::is_same_v<_T2, fpemu_unpacked>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_API friend bool operator>=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpemu_unpacked(__x) >= fpemu_unpacked(__y);
  }

}; // class fpemu_unpacked

/*
// Aliases for the emulated floating-point types
*/
using fp64emu      = fpemu<double, fpemu_accuracy::def>;
using fp64emu_low  = fpemu<double, fpemu_accuracy::low>;
using fp64emu_mid  = fpemu<double, fpemu_accuracy::mid>;
using fp64emu_high = fpemu<double, fpemu_accuracy::high>;

using fp64emu_unpacked      = fpemu_unpacked<double, fpemu_accuracy::def>;
using fp64emu_unpacked_low  = fpemu_unpacked<double, fpemu_accuracy::low>;
using fp64emu_unpacked_mid  = fpemu_unpacked<double, fpemu_accuracy::mid>;
using fp64emu_unpacked_high = fpemu_unpacked<double, fpemu_accuracy::high>;

// Trait machinery for the mixed-operand free-function builtins (fma, __dadd_rn, dot,
// cmul, ...) shared by the packed fpemu and unpacked fpemu_unpacked classes.
// __is_fpemu_v detects an fpemu / fpemu_unpacked specialization; __fpemu_pick_t selects
// the fpemu-family type among a set of operands; __fpemu_mixed_v is the constraint "at
// least one fpemu-family operand AND at least one arithmetic operand" (so pure
// fpemu-only calls bind to the exact-match cores, and pure-arithmetic calls are left to
// the built-in types).
template <class _Tp>
inline constexpr bool __is_fpemu_v = false;
template <class _FpType, fpemu_accuracy _Acc>
inline constexpr bool __is_fpemu_v<fpemu<_FpType, _Acc>> = true;
template <class _FpType, fpemu_accuracy _Acc>
inline constexpr bool __is_fpemu_v<fpemu_unpacked<_FpType, _Acc>> = true;

template <class... _Ts>
inline constexpr bool __fpemu_mixed_v = (__is_fpemu_v<_Ts> || ...) && (::cuda::std::is_arithmetic_v<_Ts> || ...);

template <class... _Ts>
struct __fpemu_pick
{
  using type = void;
};
template <class _T0, class... _Ts>
struct __fpemu_pick<_T0, _Ts...>
{
  using type = ::cuda::std::conditional_t<__is_fpemu_v<_T0>, _T0, typename __fpemu_pick<_Ts...>::type>;
};
template <class... _Ts>
using __fpemu_pick_t = typename __fpemu_pick<_Ts...>::type;

// Define this macro so that the API sections in _impl.hpp files are activated.
// The _impl.hpp files are structured with implementation code under their own
// include guard, and API code (operators, class methods) under this guard.
// This ensures API code is only compiled after class definitions are complete.
#define _CCCL_FPEMU_API_CLASSES_DEFINED
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#include <cuda/__fp/fpemu_impl_add.h>
#include <cuda/__fp/fpemu_impl_cmp.h>
#include <cuda/__fp/fpemu_impl_cvt.h>
#include <cuda/__fp/fpemu_impl_div.h>
#include <cuda/__fp/fpemu_impl_fma.h>
#include <cuda/__fp/fpemu_impl_mul.h>
#include <cuda/__fp/fpemu_impl_others.h>
#include <cuda/__fp/fpemu_impl_sqrt.h>
#include <cuda/__fp/fpemu_impl_sub.h>

#endif // _CUDA___FP_FPEMU_H
