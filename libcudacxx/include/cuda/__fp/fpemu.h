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
/**
 * @file fpemu.h
 * @brief Main header file for the FPEMU floating point scalar emulation library
 *
 * This is the main header file that provides access to the complete FPEMU library.
 * It includes all the necessary headers for:
 *
 * - Core definitions, macros and enumerations (fpemu_common.h)
 * - Class templates (fpemu, fpemu_unpacked)
 * - Public API functions (operators, builtins, conversions)
 * - Implementation files for specific scalar operations:
 *   - Comparison operations (fpemu_impl_cmp.h)
 *   - Type conversions (fpemu_impl_cvt.h)
 *   - Fused multiply-add (fpemu_impl_fma.h)
 *   - Addition (fpemu_impl_add.h)
 *   - Subtraction (fpemu_impl_sub.h)
 *   - Multiplication (fpemu_impl_mul.h)
 *   - Division (fpemu_impl_div.h)
 *   - Square root (fpemu_impl_sqrt.h)
 *   - Other operations (fpemu_impl_others.h)
 *
 * The library provides IEEE-754 compliant emulated scalar floating point operations
 * with configurable rounding modes and computation methods.
 *
 * Accuracy levels (template parameter 'fpemu_accuracy'):
 *   - fpemu_accuracy::high — correctly rounded, full IEEE-754 range including
 *                        infinities, NaNs, and subnormals
 *   - fpemu_accuracy::mid  — up to 1-2 least significant mantissa bits of error,
 *                        limited INF, NaN and subnormal support
 *   - fpemu_accuracy::low  — up to half of the mantissa bits may be lost,
 *                        limited INF, NaN and subnormal support
 *   - fpemu_accuracy::def  — default selector; equals high (IEEE-correct)
 *
 * The API supports both host and device code through appropriate decorators and
 * can utilize different computational backends based on template parameters.
 */
 
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/cstdint>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>

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

    /**
    * @brief Tag type for constructing an fpemu directly from raw __fpbits64 bits.
    *
    * @internal Library-internal. This disambiguates the raw-bits constructor
    * `fpemu(__fpbits64_construct_tag, const __fpbits64&)` from the value-converting
    * constructors (fpemu(double), fpemu(integer), ...): since __fpbits64 is just
    * uint64_t, a plain bits constructor would collide with the integer-value
    * constructors. The builtin forwarders (fpemu_impl_*.h) use it to wrap a raw
    * __fp64emu_* result back into an fpemu without a conversion. Not public.
    *
    * Usage (internal):
    *   return fpemu<double, _Acc>(__fpbits64_construct, __fp64emu_from_double(x));
    */
    struct __fpbits64_construct_tag { explicit __fpbits64_construct_tag() = default; };

    // Global constant instance of __fpbits64_construct_tag for convenient usage
    // (host/device accessible)
    _CCCL_GLOBAL_CONSTANT __fpbits64_construct_tag __fpbits64_construct{};

    // Forward declaration of unpacked floating-point class
    template <typename _FpType, fpemu_accuracy _Met> class fpemu_unpacked;

    /**
    * @brief Primary emulated double-precision floating-point class template
    *
    * The fpemu class template represents a double-precision (64-bit)
    * floating-point number, emulated according to IEEE-754 semantics but with 
    * configurable accuracy level.
    *
    * @tparam met Accuracy level (fpemu_accuracy::high, mid, low; def == high)
    *              - high: Correctly rounded with full IEEE-754 range
    *              - mid: 1-2 LSB error with normal range
    *              - low: Low accuracy with normal range
    *
    * This class provides:
    *   - Storage of the value as __fpbits64 (raw IEEE-754 format)
    *   - Construction from and conversion to standard C++ types (int, float, double)
    *   - Arithmetic operators and mathematical functions
    *   - Fine-grained control over rounding and accuracy level
    *   - Portable host/device compatibility (CUDA/HIP/etc)
    * 
    * Usage:
    *   fpemu<double, fpemu_accuracy::high> x{1.5};
    *   fpemu<double> y = x + 2.0;
    *   double z = static_cast<double>(y);
    */
    template <typename _FpType = double, fpemu_accuracy _Met = fpemu_accuracy::def> 
    class fpemu 
    {
     public:

        // Only double emulation is implemented today; the _FpType axis exists for future extension.
        static_assert(::cuda::std::is_same_v<_FpType, double>, "cuda::experimental::fpemu currently supports only _FpType == double, possible future extension to other types emulation");

        // Internal representation of the floating-point value
        // __fpbits64 is defined in fpemu_common.h
        __fpbits64 bits;
        
        /*
        // Constructors and assignment operators
        */
        // Basic constructors
        _CCCL_API inline fpemu() noexcept : bits{0u} {}
        _CCCL_API inline fpemu(__fpbits64_construct_tag, const __fpbits64& __f) noexcept : bits(__f) {}
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
        template<typename _Dummy = void>
        _CCCL_API inline fpemu(const volatile fpemu& __other) noexcept : bits(__other.bits) {}

        // Defaulted copy assignment operator (trivially copyable)
        fpemu& operator=(const fpemu& __other) = default;

        /*
        // Assignment operator to volatile fpemu
        // Template so it is NOT a copy assignment operator per the C++ standard
        // Returns void to avoid C++20 -Wvolatile (deprecated volatile return)
        */
        template<typename _Dummy = void>
        _CCCL_API inline void operator=(const fpemu& __other) volatile noexcept { bits = __other.bits; }

        /*
        // Assignment operator from volatile fpemu
        // Template so it is NOT a copy assignment operator per the C++ standard
        */
        template<typename _Dummy = void>
        _CCCL_API inline fpemu& operator=(const volatile fpemu& __other) noexcept { bits = __other.bits; return *this; }

        /*
        // Conversion operators
        */
        // ==== Conversions from other types to fpemu:
        // Implicit conversions from floating-point types
        _CCCL_API inline fpemu(float __f) noexcept ;
        _CCCL_API inline fpemu(double __d) noexcept ;
        // Construction from any standard integer type (int / long / long long + unsigned).
        // 32-bit and narrower are lossless in double and stay implicit; 64-bit may lose
        // precision and are explicit (as the prior fixed-width API required). Dispatch is
        // by width and signedness to the accuracy-correct integer builtins (via the private
        // out-of-line helpers below), so every integer type is handled portably.
        // bool / character types are excluded by __cccl_is_integer_v.
        _CCCL_TEMPLATE(class _Tp)
        _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND(sizeof(_Tp) <= sizeof(int32_t)))
        _CCCL_API inline fpemu(_Tp __i) noexcept
        {
            if constexpr (::cuda::std::__cccl_is_signed_integer_v<_Tp>) { __set_from_int (static_cast<int32_t>(__i)); }
            else                                                        { __set_from_uint(static_cast<uint32_t>(__i)); }
        }
        _CCCL_TEMPLATE(class _Tp)
        _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND(sizeof(_Tp) > sizeof(int32_t)))
        _CCCL_API explicit inline fpemu(_Tp __i) noexcept
        {
            if constexpr (::cuda::std::__cccl_is_signed_integer_v<_Tp>) { __set_from_ll (static_cast<int64_t>(__i)); }
            else                                                        { __set_from_ull(static_cast<uint64_t>(__i)); }
        }
        // Type conversion to fpemu with other accuracy and range
        template<fpemu_accuracy _Acc = _Met> _CCCL_API inline operator fpemu<double, _Acc>() const noexcept ;
        // Type conversion from fpemu to fpemu_unpacked (explicit to avoid overload ambiguity)
        template<fpemu_accuracy _Acc = _Met> _CCCL_API explicit inline operator fpemu_unpacked<double, _Acc>() const noexcept ;

        // ==== Conversion from fpemu to other types:
        // Implicit conversion to double
        _CCCL_API inline operator double() const noexcept ;
        // Explicit conversions to other types
        _CCCL_API explicit inline operator float()    const noexcept ;
        // Explicit conversion to any standard integer type (int / long / long long + unsigned).
        // Dispatches by width and signedness to the accuracy-correct integer builtins (via the
        // private out-of-line helpers below); excludes bool / character types.
        _CCCL_TEMPLATE(class _Tp)
        _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
        _CCCL_API inline explicit operator _Tp() const noexcept
        {
            if constexpr (::cuda::std::__cccl_is_signed_integer_v<_Tp>)
            {
                if constexpr (sizeof(_Tp) <= sizeof(int32_t)) { return static_cast<_Tp>(__to_int()); }
                else                                          { return static_cast<_Tp>(__to_ll());  }
            }
            else
            {
                if constexpr (sizeof(_Tp) <= sizeof(uint32_t)) { return static_cast<_Tp>(__to_uint()); }
                else                                           { return static_cast<_Tp>(__to_ull());  }
            }
        }

     private:
        // Accuracy-correct integer <-> value helpers (defined out-of-line where the fpemu
        // builtins are visible). Kept non-template so the definitions stay out-of-line.
        _CCCL_API inline void     __set_from_int (int32_t  __i) noexcept ;
        _CCCL_API inline void     __set_from_uint(uint32_t __i) noexcept ;
        _CCCL_API inline void     __set_from_ll  (int64_t  __i) noexcept ;
        _CCCL_API inline void     __set_from_ull (uint64_t __i) noexcept ;
        _CCCL_API inline int32_t  __to_int () const noexcept ;
        _CCCL_API inline uint32_t __to_uint() const noexcept ;
        _CCCL_API inline int64_t  __to_ll  () const noexcept ;
        _CCCL_API inline uint64_t __to_ull () const noexcept ;
     public:

        /*
        //  CUDA builtins functions for conversions
        */
        // double to float
        template<fpemu_accuracy _Acc> _CCCL_API friend inline float  __double2float (fpemu<double, _Acc> __x) noexcept ;
        // double to integer
        template<fpemu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_rn (fpemu<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_rz (fpemu<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_ru (fpemu<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_rd (fpemu<double, _Acc> __x) noexcept ;
        // double to unsigned integer
        template<fpemu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_rn (fpemu<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_rz (fpemu<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_ru (fpemu<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_rd (fpemu<double, _Acc> __x) noexcept ;
        // double to signed integer
        template<fpemu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_rn (fpemu<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_rz (fpemu<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_ru (fpemu<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_rd (fpemu<double, _Acc> __x) noexcept ;
        // double to unsigned integer
        template<fpemu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_rn (fpemu<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_rz (fpemu<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_ru (fpemu<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_rd (fpemu<double, _Acc> __x) noexcept ;
        // other types to double
        template<fpemu_accuracy _Acc> _CCCL_API friend inline fpemu<double, _Acc> __int2double   (int32_t __x)  noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline fpemu<double, _Acc> __uint2double  (uint32_t __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline fpemu<double, _Acc> __ll2double    (int64_t __x)  noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline fpemu<double, _Acc> __ull2double   (uint64_t __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline fpemu<double, _Acc> __float2double (float __x)    noexcept ;
    
        /*
        // Arithmetic operations:
        */
        // === mul ===
        // (*)
        template<fpemu_accuracy _Acc> _CCCL_API friend fpemu<double, _Acc> operator*(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept ;
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu operator*(const _T1& __x, const _T2& __y) noexcept { return fpemu(__x) * fpemu(__y); }
        // dmul_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __dmul_rn(const _T1& __x, const _T2& __y) noexcept { return __dmul_rn(fpemu(__x), fpemu(__y)); }
        // dmul_rz
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __dmul_rz(const _T1& __x, const _T2& __y) noexcept { return __dmul_rz(fpemu(__x), fpemu(__y)); }
        // dmul_ru
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __dmul_ru(const _T1& __x, const _T2& __y) noexcept { return __dmul_ru(fpemu(__x), fpemu(__y)); }
        // dmul_rd
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __dmul_rd(const _T1& __x, const _T2& __y) noexcept { return __dmul_rd(fpemu(__x), fpemu(__y)); }
        
        // === div ===
        // (/)
        template<fpemu_accuracy _Acc> _CCCL_API friend fpemu<double, _Acc> operator/(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept ;
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu operator/(const _T1& __x, const _T2& __y) noexcept { return fpemu(__x) / fpemu(__y); }
        // ddiv_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __ddiv_rn(const _T1& __x, const _T2& __y) noexcept { return __ddiv_rn(fpemu(__x), fpemu(__y)); }
        // ddiv_rz
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __ddiv_rz(const _T1& __x, const _T2& __y) noexcept { return __ddiv_rz(fpemu(__x), fpemu(__y)); }
        // ddiv_ru
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __ddiv_ru(const _T1& __x, const _T2& __y) noexcept { return __ddiv_ru(fpemu(__x), fpemu(__y)); }
        // ddiv_rd
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __ddiv_rd(const _T1& __x, const _T2& __y) noexcept { return __ddiv_rd(fpemu(__x), fpemu(__y)); }

        // === add ===
        // (+)
        template<fpemu_accuracy _Acc> _CCCL_API friend fpemu<double, _Acc> operator+(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept ;
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu operator+(const _T1& __x, const _T2& __y) noexcept { return fpemu(__x) + fpemu(__y); }
        // dadd_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __dadd_rn(const _T1& __x, const _T2& __y) noexcept {  return __dadd_rn(fpemu(__x), fpemu(__y)); }
        // dadd_rz
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __dadd_rz(const _T1& __x, const _T2& __y) noexcept {  return __dadd_rz(fpemu(__x), fpemu(__y)); }
        // dadd_ru
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __dadd_ru(const _T1& __x, const _T2& __y) noexcept { return __dadd_ru(fpemu(__x), fpemu(__y)); }
        // dadd_rd
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __dadd_rd(const _T1& __x, const _T2& __y) noexcept { return __dadd_rd(fpemu(__x), fpemu(__y)); }

        // === sub ===
        // (-)
        template<fpemu_accuracy _Acc> _CCCL_API friend fpemu<double, _Acc> operator-(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept ;
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu operator-(const _T1& __x, const _T2& __y) noexcept { return fpemu(__x) - fpemu(__y); }
        // dsub_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __dsub_rn(const _T1& __x, const _T2& __y) noexcept { return __dsub_rn(fpemu(__x), fpemu(__y)); }
        // dsub_rz
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __dsub_rz(const _T1& __x, const _T2& __y) noexcept { return __dsub_rz(fpemu(__x), fpemu(__y)); }
        // dsub_ru
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __dsub_ru(const _T1& __x, const _T2& __y) noexcept { return __dsub_ru(fpemu(__x), fpemu(__y)); }
        // dsub_rd
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu __dsub_rd(const _T1& __x, const _T2& __y) noexcept { return __dsub_rd(fpemu(__x), fpemu(__y)); }

        // === sqrt ===
        // sqrt
        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fpemu sqrt(const _T1& __x) noexcept { return sqrt(fpemu(__x)); }        
        // dsqrt_rn
        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fpemu __dsqrt_rn(const _T1& __x) noexcept { return __dsqrt_rn(fpemu(__x)); }

        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fpemu __dsqrt_rz(const _T1& __x) noexcept { return __dsqrt_rz(fpemu(__x)); }
        // dsqrt_ru
        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fpemu __dsqrt_ru(const _T1& __x) noexcept { return __dsqrt_ru(fpemu(__x)); }
        // dsqrt_rd
        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fpemu __dsqrt_rd(const _T1& __x) noexcept { return __dsqrt_rd(fpemu(__x)); }

        // === fma ===
        // fma
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu> || ::cuda::std::is_same_v<_T3,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fpemu fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return fma(fpemu(__x), fpemu(__y), fpemu(__z)); }
        // dfma_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu> || ::cuda::std::is_same_v<_T3,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fpemu __fma_rn(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __fma_rn(fpemu(__x), fpemu(__y), fpemu(__z)); }
        // dfma_rz
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu> || ::cuda::std::is_same_v<_T3,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fpemu __fma_rz(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __fma_rz(fpemu(__x), fpemu(__y), fpemu(__z)); }
        // dfma_ru
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu> || ::cuda::std::is_same_v<_T3,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fpemu __fma_ru(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __fma_ru(fpemu(__x), fpemu(__y), fpemu(__z)); }
        // dfma_rd
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu> || ::cuda::std::is_same_v<_T3,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fpemu __fma_rd(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __fma_rd(fpemu(__x), fpemu(__y), fpemu(__z)); }

        // === mad ===
        // mad
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu> || ::cuda::std::is_same_v<_T3,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fpemu mad(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return mad(fpemu(__x), fpemu(__y), fpemu(__z)); }
        // dmad_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu> || ::cuda::std::is_same_v<_T3,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fpemu __mad_rn(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __mad_rn(fpemu(__x), fpemu(__y), fpemu(__z)); }

        // === dot ===
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3, typename _T4)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu> || ::cuda::std::is_same_v<_T3,fpemu> || ::cuda::std::is_same_v<_T4,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3> || ::cuda::std::is_arithmetic_v<_T4>)))
            _CCCL_API friend  fpemu dot(const _T1& __x1, const _T2& __y1, const _T3& __x2, const _T4& __y2) noexcept { return dot(fpemu(__x1), fpemu(__y1), fpemu(__x2), fpemu(__y2)); }

         // === cmul ===
         _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3, typename _T4)
         _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu> || ::cuda::std::is_same_v<_T3,fpemu> || ::cuda::std::is_same_v<_T4,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3> || ::cuda::std::is_arithmetic_v<_T4>)))
             _CCCL_API friend void cmul(const _T1& __x_re, const _T2& __x_im, const _T3& __y_re, const _T4& __y_im, fpemu& __r_re, fpemu& __r_im) noexcept { cmul(fpemu(__x_re), fpemu(__x_im), fpemu(__y_re), fpemu(__y_im), __r_re, __r_im); }

        // Prefix increment/decrement
        _CCCL_API fpemu& operator++() noexcept { this = this + fpemu(1.0); return *this; }
        _CCCL_API fpemu& operator--() noexcept { this = this - fpemu(1.0); return *this; }
        // Postfix increment/decrement
        _CCCL_API fpemu  operator++(int) noexcept { fpemu __temp(*this); this = this + fpemu(1.0); return __temp; }
        _CCCL_API fpemu  operator--(int) noexcept { fpemu __temp(*this); this = this - fpemu(1.0); return __temp; }
        // Compound assignment operators
        _CCCL_API fpemu& operator+=(const fpemu& __other) noexcept { *this = *this + __other; return *this; }
        _CCCL_API fpemu& operator-=(const fpemu& __other) noexcept { *this = *this - __other; return *this; }
        _CCCL_API fpemu& operator*=(const fpemu& __other) noexcept { *this = *this * __other; return *this; }
        _CCCL_API fpemu& operator/=(const fpemu& __other) noexcept { *this = *this / __other; return *this; }
        // Unary negation operator (implementation in fpemu_impl_others.h)
        _CCCL_API fpemu  operator-() const noexcept ;

        /*
        // Comparison operators:
        */       
        // equality (==)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator==(const _T1& __x, const _T2& __y) noexcept { return fpemu(__x) == fpemu(__y); }
        // inequality (!=)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator!=(const _T1& __x, const _T2& __y) noexcept { return fpemu(__x) != fpemu(__y); }
        // less than (<)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator<(const _T1& __x, const _T2& __y) noexcept { return fpemu(__x) < fpemu(__y); }
        // greater than (>)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator>(const _T1& __x, const _T2& __y) noexcept { return fpemu(__x) > fpemu(__y); }
        // less than or equal to (<=)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator<=(const _T1& __x, const _T2& __y) noexcept { return fpemu(__x) <= fpemu(__y); }
        // greater than or equal to (>=)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu> || ::cuda::std::is_same_v<_T2,fpemu>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator>=(const _T1& __x, const _T2& __y) noexcept { return fpemu(__x) >= fpemu(__y); }
    }; // class fpemu 


    template <typename _FpType = double, fpemu_accuracy _Met = fpemu_accuracy::def> 
    class fpemu_unpacked 
    {
     public:

        // Only double emulation is implemented today; the _FpType axis exists for future extension.
        static_assert(::cuda::std::is_same_v<_FpType, double>, "cuda::experimental::fpemu_unpacked currently supports only _FpType == double");

        // Internal representation of the unpacked floating-point value
        // __fpbits64_unpacked is defined in fpemu_common.h
        __fpbits64_unpacked bits;
        
        /*
        // Constructors and assignment operators
        */
        // Basic constructors
        _CCCL_API inline fpemu_unpacked() noexcept : bits{0u, 0, 0} {}
        _CCCL_API inline fpemu_unpacked(__fpbits64_construct_tag, const __fpbits64_unpacked& __f) noexcept : bits(__f) {}
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
        template<typename _Dummy = void>
        _CCCL_API inline fpemu_unpacked(const volatile fpemu_unpacked& __other) noexcept
        { 
            bits.sign = __other.bits.sign; 
            bits.exponent = __other.bits.exponent; 
            bits.mantissa = __other.bits.mantissa; 
        }

        // Defaulted copy assignment operator (trivially copyable)
        fpemu_unpacked& operator=(const fpemu_unpacked& __other) = default;

        /*
        // Assignment operator to volatile fpemu_unpacked
        // Template so it is NOT a copy assignment operator per the C++ standard
        // Returns void to avoid C++20 -Wvolatile (deprecated volatile return)
        */
        template<typename _Dummy = void>
        _CCCL_API inline void operator=(const fpemu_unpacked& __other) volatile noexcept
        { 
            bits.sign = __other.bits.sign; 
            bits.exponent = __other.bits.exponent; 
            bits.mantissa = __other.bits.mantissa; 
        }

        /*
        // Assignment operator from volatile fpemu_unpacked
        // Template so it is NOT a copy assignment operator per the C++ standard
        */
        template<typename _Dummy = void>
        _CCCL_API inline fpemu_unpacked& operator=(const volatile fpemu_unpacked& __other) noexcept
        { 
            bits.sign = __other.bits.sign; 
            bits.exponent = __other.bits.exponent; 
            bits.mantissa = __other.bits.mantissa; 
            return *this; 
        }
        /*
        // Conversion operators
        */
        // ==== Conversions from other types to fpemu_unpacked:
#if defined __CUDACC__
        // Implicit conversions from floating-point types 
        _CCCL_API  inline fpemu_unpacked(float f) noexcept ;
        _CCCL_API  inline fpemu_unpacked(double d) noexcept ;        
#  define _CCCL_FPEMU_UNP_NARROW_EXPLICIT
#else
        // Explicit conversions from floating-point types (to avoid ambiguity with packed type)
        _CCCL_API explicit inline fpemu_unpacked(float __f) noexcept ;
        _CCCL_API explicit inline fpemu_unpacked(double __d) noexcept ;        
#  define _CCCL_FPEMU_UNP_NARROW_EXPLICIT explicit
#endif
        // Construction from any standard integer type (int / long / long long + unsigned).
        // 32-bit and narrower are lossless in double; 64-bit values may lose precision. The
        // narrow-integer explicitness follows the surrounding float ctors (implicit on device,
        // explicit on host) to avoid ambiguity with the packed type; 64-bit is always explicit.
        // Dispatch is by width and signedness to the accuracy-correct integer builtins (via the
        // private out-of-line helpers below); bool / character types are excluded.
        _CCCL_TEMPLATE(class _Tp)
        _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND(sizeof(_Tp) <= sizeof(int32_t)))
        _CCCL_API _CCCL_FPEMU_UNP_NARROW_EXPLICIT inline fpemu_unpacked(_Tp __i) noexcept
        {
            if constexpr (::cuda::std::__cccl_is_signed_integer_v<_Tp>) { __set_from_int (static_cast<int32_t>(__i)); }
            else                                                        { __set_from_uint(static_cast<uint32_t>(__i)); }
        }
        _CCCL_TEMPLATE(class _Tp)
        _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND(sizeof(_Tp) > sizeof(int32_t)))
        _CCCL_API explicit inline fpemu_unpacked(_Tp __i) noexcept
        {
            if constexpr (::cuda::std::__cccl_is_signed_integer_v<_Tp>) { __set_from_ll (static_cast<int64_t>(__i)); }
            else                                                        { __set_from_ull(static_cast<uint64_t>(__i)); }
        }
#undef _CCCL_FPEMU_UNP_NARROW_EXPLICIT
        // Type conversion to fpemu_unpacked with other accuracy and range
        template<fpemu_accuracy _Acc = _Met> _CCCL_API inline operator fpemu_unpacked<double, _Acc>() const noexcept ;
        // Type conversion from fpemu_unpacked to fpemu (explicit to avoid overload ambiguity)
        template<fpemu_accuracy _Acc = _Met> _CCCL_API explicit inline operator fpemu<double, _Acc>() const noexcept ;

        // ==== Conversion from fpemu_unpacked to other types:
        // Implicit conversion to double
        _CCCL_API inline operator double() const noexcept ;
        // Explicit conversions to other types
        _CCCL_API explicit inline operator float()    const noexcept ;
        // Explicit conversion to any standard integer type (int / long / long long + unsigned).
        // Dispatches by width and signedness to the accuracy-correct integer builtins (via the
        // private out-of-line helpers below); excludes bool / character types.
        _CCCL_TEMPLATE(class _Tp)
        _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
        _CCCL_API inline explicit operator _Tp() const noexcept
        {
            if constexpr (::cuda::std::__cccl_is_signed_integer_v<_Tp>)
            {
                if constexpr (sizeof(_Tp) <= sizeof(int32_t)) { return static_cast<_Tp>(__to_int()); }
                else                                          { return static_cast<_Tp>(__to_ll());  }
            }
            else
            {
                if constexpr (sizeof(_Tp) <= sizeof(uint32_t)) { return static_cast<_Tp>(__to_uint()); }
                else                                           { return static_cast<_Tp>(__to_ull());  }
            }
        }

     private:
        // Accuracy-correct integer <-> value helpers (defined out-of-line where the fpemu
        // builtins are visible). Kept non-template so the definitions stay out-of-line.
        _CCCL_API inline void     __set_from_int (int32_t  __i) noexcept ;
        _CCCL_API inline void     __set_from_uint(uint32_t __i) noexcept ;
        _CCCL_API inline void     __set_from_ll  (int64_t  __i) noexcept ;
        _CCCL_API inline void     __set_from_ull (uint64_t __i) noexcept ;
        _CCCL_API inline int32_t  __to_int () const noexcept ;
        _CCCL_API inline uint32_t __to_uint() const noexcept ;
        _CCCL_API inline int64_t  __to_ll  () const noexcept ;
        _CCCL_API inline uint64_t __to_ull () const noexcept ;
     public:

        /*
        //  CUDA builtins functions for conversions
        */
        template<fpemu_accuracy _Acc> _CCCL_API friend inline float    __double2float  (fpemu_unpacked<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline int32_t  __double2int_rz (fpemu_unpacked<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_rz(fpemu_unpacked<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline int64_t  __double2ll_rz  (fpemu_unpacked<double, _Acc> __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_rz (fpemu_unpacked<double, _Acc> __x) noexcept ;

        template<fpemu_accuracy _Acc> _CCCL_API friend inline fpemu_unpacked<double, _Acc> __float2double (float __x)    noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline fpemu_unpacked<double, _Acc> __int2double   (int32_t __x)  noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline fpemu_unpacked<double, _Acc> __uint2double  (uint32_t __x) noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline fpemu_unpacked<double, _Acc> __ll2double    (int64_t __x)  noexcept ;
        template<fpemu_accuracy _Acc> _CCCL_API friend inline fpemu_unpacked<double, _Acc> __ull2double   (uint64_t __x) noexcept ;

        /*
        // Arithmetic operations:
        */
        // === mul ===
        // (*)
        template<fpemu_accuracy _Acc> _CCCL_API friend fpemu_unpacked<double, _Acc> operator*(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept ;
        // (/)
        template<fpemu_accuracy _Acc> _CCCL_API friend fpemu_unpacked<double, _Acc> operator/(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept ;
        // (+)
        template<fpemu_accuracy _Acc> _CCCL_API friend fpemu_unpacked<double, _Acc> operator+(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept ;
        // (-)
        template<fpemu_accuracy _Acc> _CCCL_API friend fpemu_unpacked<double, _Acc> operator-(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept ;
        

        // == mul ==
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu_unpacked operator*(const _T1& __x, const _T2& __y) noexcept { return fpemu_unpacked(__x) * fpemu_unpacked(__y); }
        // dmul_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu_unpacked __dmul_rn(const _T1& __x, const _T2& __y) noexcept { return __dmul_rn(fpemu_unpacked(__x), fpemu_unpacked(__y)); }

        // === div ===
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu_unpacked operator/(const _T1& __x, const _T2& __y) noexcept { return fpemu_unpacked(__x) / fpemu_unpacked(__y); }
        // ddiv_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu_unpacked __ddiv_rn(const _T1& __x, const _T2& __y) noexcept { return __ddiv_rn(fpemu_unpacked(__x), fpemu_unpacked(__y)); }

        // === add ===
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu_unpacked operator+(const _T1& __x, const _T2& __y) noexcept { return fpemu_unpacked(__x) + fpemu_unpacked(__y); }
        // dadd_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu_unpacked __dadd_rn(const _T1& __x, const _T2& __y) noexcept {  return __dadd_rn(fpemu_unpacked(__x), fpemu_unpacked(__y)); }

        // === sub ===
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu_unpacked operator-(const _T1& __x, const _T2& __y) noexcept { return fpemu_unpacked(__x) - fpemu_unpacked(__y); }
        // dsub_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fpemu_unpacked __dsub_rn(const _T1& __x, const _T2& __y) noexcept { return __dsub_rn(fpemu_unpacked(__x), fpemu_unpacked(__y)); }

        // === sqrt ===
        // sqrt
        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fpemu_unpacked sqrt(const _T1& __x) noexcept { return sqrt(fpemu_unpacked(__x)); }        
        // dsqrt_rn
        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fpemu_unpacked __dsqrt_rn(const _T1& __x) noexcept { return __dsqrt_rn(fpemu_unpacked(__x)); }

        // === fma ===
        // fma
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked> || ::cuda::std::is_same_v<_T3,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fpemu_unpacked fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return fma(fpemu_unpacked(__x), fpemu_unpacked(__y), fpemu_unpacked(__z)); }
        // dfma_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked> || ::cuda::std::is_same_v<_T3,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fpemu_unpacked __fma_rn(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __fma_rn(fpemu_unpacked(__x), fpemu_unpacked(__y), fpemu_unpacked(__z)); }

        // === mad ===
        // mad
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked> || ::cuda::std::is_same_v<_T3,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fpemu_unpacked mad(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return mad(fpemu_unpacked(__x), fpemu_unpacked(__y), fpemu_unpacked(__z)); }
        // dmad_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked> || ::cuda::std::is_same_v<_T3,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fpemu_unpacked __mad_rn(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __mad_rn(fpemu_unpacked(__x), fpemu_unpacked(__y), fpemu_unpacked(__z)); }

        // === dot ===
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3, typename _T4)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked> || ::cuda::std::is_same_v<_T3,fpemu_unpacked> || ::cuda::std::is_same_v<_T4,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3> || ::cuda::std::is_arithmetic_v<_T4>)))
            _CCCL_API friend  fpemu_unpacked dot(const _T1& __x1, const _T2& __y1, const _T3& __x2, const _T4& __y2) noexcept { return dot(fpemu_unpacked(__x1), fpemu_unpacked(__y1), fpemu_unpacked(__x2), fpemu_unpacked(__y2)); }

         // === cmul ===
         _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3, typename _T4)
         _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked> || ::cuda::std::is_same_v<_T3,fpemu_unpacked> || ::cuda::std::is_same_v<_T4,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3> || ::cuda::std::is_arithmetic_v<_T4>)))
             _CCCL_API friend void cmul(const _T1& __x_re, const _T2& __x_im, const _T3& __y_re, const _T4& __y_im, fpemu_unpacked& __r_re, fpemu_unpacked& __r_im) noexcept { cmul(fpemu_unpacked(__x_re), fpemu_unpacked(__x_im), fpemu_unpacked(__y_re), fpemu_unpacked(__y_im), __r_re, __r_im); }

        // Prefix increment/decrement
        _CCCL_API fpemu_unpacked& operator++() noexcept { this = this + fpemu_unpacked(1.0); return *this; }
        _CCCL_API fpemu_unpacked& operator--() noexcept { this = this - fpemu_unpacked(1.0); return *this; }
        // Postfix increment/decrement
        _CCCL_API fpemu_unpacked  operator++(int) noexcept { fpemu_unpacked __temp(*this); this = this + fpemu_unpacked(1.0); return __temp; }
        _CCCL_API fpemu_unpacked  operator--(int) noexcept { fpemu_unpacked __temp(*this); this = this - fpemu_unpacked(1.0); return __temp; }
        // Compound assignment operators
        _CCCL_API fpemu_unpacked& operator+=(const fpemu_unpacked& __other) noexcept { *this = *this + __other; return *this; }
        _CCCL_API fpemu_unpacked& operator-=(const fpemu_unpacked& __other) noexcept { *this = *this - __other; return *this; }
        _CCCL_API fpemu_unpacked& operator*=(const fpemu_unpacked& __other) noexcept { *this = *this * __other; return *this; }
        _CCCL_API fpemu_unpacked& operator/=(const fpemu_unpacked& __other) noexcept { *this = *this / __other; return *this; }
        // Unary negation operator (implementation in fpemu_impl_others.h)
        _CCCL_API fpemu_unpacked  operator-() const noexcept ;

        /*
        // Comparison operators:
        */       
        // equality (==)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator==(const _T1& __x, const _T2& __y) noexcept { return fpemu_unpacked(__x) == fpemu_unpacked(__y); }
        // inequality (!=)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator!=(const _T1& __x, const _T2& __y) noexcept { return fpemu_unpacked(__x) != fpemu_unpacked(__y); }
        // less than (<)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator<(const _T1& __x, const _T2& __y) noexcept { return fpemu_unpacked(__x) < fpemu_unpacked(__y); }
        // greater than (>)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator>(const _T1& __x, const _T2& __y) noexcept { return fpemu_unpacked(__x) > fpemu_unpacked(__y); }
        // less than or equal to (<=)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator<=(const _T1& __x, const _T2& __y) noexcept { return fpemu_unpacked(__x) <= fpemu_unpacked(__y); }
        // greater than or equal to (>=)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fpemu_unpacked> || ::cuda::std::is_same_v<_T2,fpemu_unpacked>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator>=(const _T1& __x, const _T2& __y) noexcept { return fpemu_unpacked(__x) >= fpemu_unpacked(__y); }

        // C++20-style bit_cast for unpacked floating-point types
        template<typename _To, fpemu_accuracy _Acc> 
        _CCCL_API friend inline _To bit_cast(const fpemu_unpacked<double, _Acc>& __from) noexcept ;

    }; // class fpemu_unpacked 

    /*
    // Aliases for the emulated floating-point types
    */
    using fp64emu               = fpemu<double, fpemu_accuracy::def>;
    using fp64emu_low           = fpemu<double, fpemu_accuracy::low>;
    using fp64emu_mid           = fpemu<double, fpemu_accuracy::mid>;
    using fp64emu_high          = fpemu<double, fpemu_accuracy::high>;

    using fp64emu_unpacked      = fpemu_unpacked<double, fpemu_accuracy::def>;
    using fp64emu_unpacked_low  = fpemu_unpacked<double, fpemu_accuracy::low>;
    using fp64emu_unpacked_mid  = fpemu_unpacked<double, fpemu_accuracy::mid>;
    using fp64emu_unpacked_high = fpemu_unpacked<double, fpemu_accuracy::high>;

// Define this macro so that the API sections in _impl.hpp files are activated.
// The _impl.hpp files are structured with implementation code under their own
// include guard, and API code (operators, class methods) under this guard.
// This ensures API code is only compiled after class definitions are complete.
#define _CCCL_FPEMU_API_CLASSES_DEFINED

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#include <cuda/__fp/fpemu_impl_cmp.h>
#include <cuda/__fp/fpemu_impl_cvt.h>
#include <cuda/__fp/fpemu_impl_fma.h>
#include <cuda/__fp/fpemu_impl_add.h>
#include <cuda/__fp/fpemu_impl_sub.h>
#include <cuda/__fp/fpemu_impl_mul.h>
#include <cuda/__fp/fpemu_impl_div.h>
#include <cuda/__fp/fpemu_impl_sqrt.h>
#include <cuda/__fp/fpemu_impl_others.h>

#endif // _CUDA___FP_FPEMU_H
