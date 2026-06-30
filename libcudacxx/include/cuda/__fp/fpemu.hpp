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
 * @file fpemu.hpp
 * @brief Main header file for the FPEMU floating point scalar emulation library
 *
 * This is the main header file that provides access to the complete FPEMU library.
 * It includes all the necessary headers for:
 *
 * - Core definitions, macros and enumerations (fpemu_common.hpp)
 * - Class templates (fp64emu_t, fp64emu_unpacked_t)
 * - Public API functions (operators, builtins, conversions)
 * - Implementation files for specific scalar operations:
 *   - Comparison operations (fpemu_impl_cmp.hpp)
 *   - Type conversions (fpemu_impl_cvt.hpp)
 *   - Fused multiply-add (fpemu_impl_fma.hpp)
 *   - Addition (fpemu_impl_add.hpp)
 *   - Subtraction (fpemu_impl_sub.hpp)
 *   - Multiplication (fpemu_impl_mul.hpp)
 *   - Division (fpemu_impl_div.hpp)
 *   - Square root (fpemu_impl_sqrt.hpp)
 *   - Other operations (fpemu_impl_others.hpp)
 *
 * The library provides IEEE-754 compliant emulated scalar floating point operations
 * with configurable rounding modes and computation methods.
 *
 * Accuracy levels (template parameter 'fp64emu_accuracy'):
 *   - fp64emu_accuracy::high — correctly rounded, full IEEE-754 range including
 *                        infinities, NaNs, and subnormals
 *   - fp64emu_accuracy::mid  — up to 1-2 least significant mantissa bits of error,
 *                        limited INF, NaN and subnormal support
 *   - fp64emu_accuracy::low  — up to half of the mantissa bits may be lost,
 *                        limited INF, NaN and subnormal support
 *   - fp64emu_accuracy::def  — default selector; equals high (IEEE-correct)
 *
 * The API supports both host and device code through appropriate decorators and
 * can utilize different computational backends based on template parameters.
 */
 
#include <cstdint>
#include <iostream>
#include <string>

#include <cuda/__fp/fpemu_common.hpp>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

    /**
    * @brief Tag type for explicit construction of fpbits64_t values
    *
    * This struct serves as a tag type to disambiguate constructors that take
    * raw bit values. It prevents implicit conversions from raw integers to
    * floating-point values and ensures that bit-level construction is explicit.
    *
    * Usage:
    *   fpbits64_t value = fpbits64_construct_t{}, raw_bits;
    *   // or by the constexpr instance:
    *   fpbits64_t value = fpbits64_construct, raw_bits;
    */
    struct fpbits64_construct_t { explicit fpbits64_construct_t() = default; };

    // Constexpr instance of fpbits64_construct_t for convenient usage
    inline constexpr fpbits64_construct_t fpbits64_construct{};

#if __FPEMU_UNPACKED__ == 1
    // Forward declaration of unpacked floating-point class
    template <fp64emu_accuracy met> class fp64emu_unpacked_t;
#endif

    /**
    * @brief Primary emulated double-precision floating-point class template
    *
    * The fp64emu_t class template represents a double-precision (64-bit)
    * floating-point number, emulated according to IEEE-754 semantics but with 
    * configurable accuracy level.
    *
    * @tparam met Accuracy level (fp64emu_accuracy::high, mid, low; def == high)
    *              - high: Correctly rounded with full IEEE-754 range
    *              - mid: 1-2 LSB error with normal range
    *              - low: Low accuracy with normal range
    *
    * This class provides:
    *   - Storage of the value as fpbits64_t (raw IEEE-754 format)
    *   - Construction from and conversion to standard C++ types (int, float, double)
    *   - Arithmetic operators and mathematical functions
    *   - Fine-grained control over rounding and accuracy level
    *   - Portable host/device compatibility (CUDA/HIP/etc)
    * 
    * Usage:
    *   fp64emu_t<fp64emu_accuracy::high> x{1.5};
    *   fp64emu_t<> y = x + 2.0;
    *   double z = static_cast<double>(y);
    */
    template <fp64emu_accuracy met = fp64emu_accuracy::def> 
    class fp64emu_t 
    {
     public:

        // Internal representation of the floating-point value
        // fpbits64_t is defined in fpemu_common.hpp
        fpbits64_t bits;
        
        /*
        // Constructors and assignment operators
        */
        // Basic constructors
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t() : bits{0u} {}
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t(fpbits64_construct_t, const fpbits64_t& f) : bits(f) {}
        /*
        // Defaulted copy constructor (trivially copyable)
        // Note: NVCC implicitly makes defaulted special members __host__ __device__
        */
        fp64emu_t(const fp64emu_t& other) = default;

        /*
        // Copy constructor from volatile fp64emu_t
        // Template so it is NOT a copy constructor per the C++ standard.
        // The volatile overloads are wrapped in dummy templates
        // so that the C++ standard does not consider them copy constructors/assignment
        // operators (a template is never a copy constructor or copy assignment operator),
        // preserving trivial copyability while retaining volatile access support.
        */
        template<typename Dummy = void>
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t(const volatile fp64emu_t& other) : bits(other.bits) {}

        // Defaulted copy assignment operator (trivially copyable)
        fp64emu_t& operator=(const fp64emu_t& other) = default;

        /*
        // Assignment operator to volatile fp64emu_t
        // Template so it is NOT a copy assignment operator per the C++ standard
        // Returns void to avoid C++20 -Wvolatile (deprecated volatile return)
        */
        template<typename Dummy = void>
        __FPEMU_HOST_DEVICE_DECL__ inline void operator=(const fp64emu_t& other) volatile { bits = other.bits; }

        /*
        // Assignment operator from volatile fp64emu_t
        // Template so it is NOT a copy assignment operator per the C++ standard
        */
        template<typename Dummy = void>
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t& operator=(const volatile fp64emu_t& other) { bits = other.bits; return *this; }

        /*
        // Conversion operators
        */
        // ==== Conversions from other types to fp64emu_t:
        // Implicit conversions from floating-point types
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t(float f);
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t(double d);
        // Implicit conversions from integer types
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t(int32_t i);
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_t(uint32_t i);

        // Explicit conversions from 64-bit integers 
        // required due to ambiguity with other constructors
        __FPEMU_HOST_DEVICE_DECL__ explicit inline fp64emu_t(int64_t i);
        __FPEMU_HOST_DEVICE_DECL__ explicit inline fp64emu_t(uint64_t i);
        // Explicit conversion from long long int types when their range is wider than int64_t
        __FPEMU_HOST_DEVICE_DECL__ explicit inline fp64emu_t(long long unsigned int i) { *this = fp64emu_t((uint64_t)i); }
        __FPEMU_HOST_DEVICE_DECL__ explicit inline fp64emu_t(long long  int i)         { *this = fp64emu_t((int64_t)i);  }
        // Type conversion to fp64emu_t with other accuracy and range
        template<fp64emu_accuracy m = met> __FPEMU_HOST_DEVICE_DECL__ inline operator fp64emu_t<m>() const;
#if __FPEMU_UNPACKED__ == 1
        // Type conversion from fp64emu_t to fp64emu_unpacked_t (explicit to avoid overload ambiguity)
        template<fp64emu_accuracy m = met> __FPEMU_HOST_DEVICE_DECL__ explicit inline operator fp64emu_unpacked_t<m>() const;
#endif

        // ==== Conversion from fp64emu_t to other types:
        // Implicit conversion to double
        __FPEMU_HOST_DEVICE_DECL__ inline operator double() const;
        // Explicit conversions to other types
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator float()    const;
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator int32_t()  const;
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator uint32_t() const;
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator int64_t()  const;
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator uint64_t() const;
        // Explicit conversion to long long int types when their range is wider than int64_t
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator long long unsigned int() const { return (uint64_t)(*this); }
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator long long int() const          { return (int64_t)(*this); }

        /*
        //  CUDA builtins functions for conversions
        */
        // double to float
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline float  __double2float (fp64emu_t<m> x);
        // double to integer
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline int32_t __double2int_rn (fp64emu_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline int32_t __double2int_rz (fp64emu_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline int32_t __double2int_ru (fp64emu_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline int32_t __double2int_rd (fp64emu_t<m> x);
        // double to unsigned integer
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline uint32_t __double2uint_rn (fp64emu_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline uint32_t __double2uint_rz (fp64emu_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline uint32_t __double2uint_ru (fp64emu_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline uint32_t __double2uint_rd (fp64emu_t<m> x);
        // double to signed integer
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline int64_t __double2ll_rn (fp64emu_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline int64_t __double2ll_rz (fp64emu_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline int64_t __double2ll_ru (fp64emu_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline int64_t __double2ll_rd (fp64emu_t<m> x);
        // double to unsigned integer
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline uint64_t __double2ull_rn (fp64emu_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline uint64_t __double2ull_rz (fp64emu_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline uint64_t __double2ull_ru (fp64emu_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline uint64_t __double2ull_rd (fp64emu_t<m> x);
        // other types to double
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline fp64emu_t<m> __int2double   (int32_t x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline fp64emu_t<m> __uint2double  (uint32_t x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline fp64emu_t<m> __ll2double    (int64_t x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline fp64emu_t<m> __ull2double   (uint64_t x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline fp64emu_t<m> __float2double (float x);
    
        /*
        // Arithmetic operations:
        */
        // === mul ===
        // (*)
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend fp64emu_t<m> operator*(const fp64emu_t<m>& x, const fp64emu_t<m>& y);
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type> 
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t operator*(const T1& x, const T2& y) { return fp64emu_t(x) * fp64emu_t(y); }
        // dmul_rn
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dmul_rn(const T1& x, const T2& y) { return __dmul_rn(fp64emu_t(x), fp64emu_t(y)); }
        // dmul_rz
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dmul_rz(const T1& x, const T2& y) { return __dmul_rz(fp64emu_t(x), fp64emu_t(y)); }
        // dmul_ru
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dmul_ru(const T1& x, const T2& y) { return __dmul_ru(fp64emu_t(x), fp64emu_t(y)); }
        // dmul_rd
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dmul_rd(const T1& x, const T2& y) { return __dmul_rd(fp64emu_t(x), fp64emu_t(y)); }
        
        // === div ===
        // (/)
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend fp64emu_t<m> operator/(const fp64emu_t<m>& x, const fp64emu_t<m>& y);
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type> 
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t operator/(const T1& x, const T2& y) { return fp64emu_t(x) / fp64emu_t(y); }
        // ddiv_rn
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __ddiv_rn(const T1& x, const T2& y) { return __ddiv_rn(fp64emu_t(x), fp64emu_t(y)); }
        // ddiv_rz
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __ddiv_rz(const T1& x, const T2& y) { return __ddiv_rz(fp64emu_t(x), fp64emu_t(y)); }
        // ddiv_ru
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __ddiv_ru(const T1& x, const T2& y) { return __ddiv_ru(fp64emu_t(x), fp64emu_t(y)); }
        // ddiv_rd
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __ddiv_rd(const T1& x, const T2& y) { return __ddiv_rd(fp64emu_t(x), fp64emu_t(y)); }

        // === add ===
        // (+)
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend fp64emu_t<m> operator+(const fp64emu_t<m>& x, const fp64emu_t<m>& y);
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type> 
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t operator+(const T1& x, const T2& y) { return fp64emu_t(x) + fp64emu_t(y); }
        // dadd_rn
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dadd_rn(const T1& x, const T2& y) {  return __dadd_rn(fp64emu_t(x), fp64emu_t(y)); }
        // dadd_rz
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dadd_rz(const T1& x, const T2& y) {  return __dadd_rz(fp64emu_t(x), fp64emu_t(y)); }
        // dadd_ru
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dadd_ru(const T1& x, const T2& y) { return __dadd_ru(fp64emu_t(x), fp64emu_t(y)); }
        // dadd_rd
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dadd_rd(const T1& x, const T2& y) { return __dadd_rd(fp64emu_t(x), fp64emu_t(y)); }

        // === sub ===
        // (-)
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend fp64emu_t<m> operator-(const fp64emu_t<m>& x, const fp64emu_t<m>& y);
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type> 
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t operator-(const T1& x, const T2& y) { return fp64emu_t(x) - fp64emu_t(y); }
        // dsub_rn
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dsub_rn(const T1& x, const T2& y) { return __dsub_rn(fp64emu_t(x), fp64emu_t(y)); }
        // dsub_rz
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dsub_rz(const T1& x, const T2& y) { return __dsub_rz(fp64emu_t(x), fp64emu_t(y)); }
        // dsub_ru
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dsub_ru(const T1& x, const T2& y) { return __dsub_ru(fp64emu_t(x), fp64emu_t(y)); }
        // dsub_rd
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dsub_rd(const T1& x, const T2& y) { return __dsub_rd(fp64emu_t(x), fp64emu_t(y)); }

        // === sqrt ===
        // sqrt
        template<typename T1, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value) && (std::is_arithmetic<T1>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t sqrt(const T1& x) { return sqrt(fp64emu_t(x)); }        
        // dsqrt_rn
        template<typename T1, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value) && (std::is_arithmetic<T1>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dsqrt_rn(const T1& x) { return __dsqrt_rn(fp64emu_t(x)); }

        template<typename T1, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value) && (std::is_arithmetic<T1>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dsqrt_rz(const T1& x) { return __dsqrt_rz(fp64emu_t(x)); }
        // dsqrt_ru
        template<typename T1, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value) && (std::is_arithmetic<T1>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dsqrt_ru(const T1& x) { return __dsqrt_ru(fp64emu_t(x)); }
        // dsqrt_rd
        template<typename T1, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value) && (std::is_arithmetic<T1>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __dsqrt_rd(const T1& x) { return __dsqrt_rd(fp64emu_t(x)); }

        // === fma ===
        // fma
        template<typename T1, typename T2, typename T3, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value || std::is_same<T3,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t fma(const T1& x, const T2& y, const T3& z) { return fma(fp64emu_t(x), fp64emu_t(y), fp64emu_t(z)); }
        // dfma_rn
        template<typename T1, typename T2, typename T3, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value || std::is_same<T3,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __fma_rn(const T1& x, const T2& y, const T3& z) { return __fma_rn(fp64emu_t(x), fp64emu_t(y), fp64emu_t(z)); }
        // dfma_rz
        template<typename T1, typename T2, typename T3, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value || std::is_same<T3,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __fma_rz(const T1& x, const T2& y, const T3& z) { return __fma_rz(fp64emu_t(x), fp64emu_t(y), fp64emu_t(z)); }
        // dfma_ru
        template<typename T1, typename T2, typename T3, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value || std::is_same<T3,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __fma_ru(const T1& x, const T2& y, const T3& z) { return __fma_ru(fp64emu_t(x), fp64emu_t(y), fp64emu_t(z)); }
        // dfma_rd
        template<typename T1, typename T2, typename T3, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value || std::is_same<T3,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __fma_rd(const T1& x, const T2& y, const T3& z) { return __fma_rd(fp64emu_t(x), fp64emu_t(y), fp64emu_t(z)); }

        // === mad ===
        // mad
        template<typename T1, typename T2, typename T3, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value || std::is_same<T3,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t mad(const T1& x, const T2& y, const T3& z) { return mad(fp64emu_t(x), fp64emu_t(y), fp64emu_t(z)); }
        // dmad_rn
        template<typename T1, typename T2, typename T3, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value || std::is_same<T3,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t __mad_rn(const T1& x, const T2& y, const T3& z) { return __mad_rn(fp64emu_t(x), fp64emu_t(y), fp64emu_t(z)); }

        // === dot ===
        template<typename T1, typename T2, typename T3, typename T4, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value || std::is_same<T3,fp64emu_t>::value || std::is_same<T4,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value || std::is_arithmetic<T4>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_t dot(const T1& x1, const T2& y1, const T3& x2, const T4& y2) { return dot(fp64emu_t(x1), fp64emu_t(y1), fp64emu_t(x2), fp64emu_t(y2)); }

         // === cmul ===
         template<typename T1, typename T2, typename T3, typename T4, typename = typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value || std::is_same<T3,fp64emu_t>::value || std::is_same<T4,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value || std::is_arithmetic<T4>::value))>::type>
             __FPEMU_HOST_DEVICE_DECL__ friend void cmul(const T1& x_re, const T2& x_im, const T3& y_re, const T4& y_im, fp64emu_t& r_re, fp64emu_t& r_im) { cmul(fp64emu_t(x_re), fp64emu_t(x_im), fp64emu_t(y_re), fp64emu_t(y_im), r_re, r_im); }

        // Prefix increment/decrement
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_t& operator++() { this = this + fp64emu_t(1.0); return *this; }
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_t& operator--() { this = this - fp64emu_t(1.0); return *this; }
        // Postfix increment/decrement
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_t  operator++(int) { fp64emu_t temp(*this); this = this + fp64emu_t(1.0); return temp; }
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_t  operator--(int) { fp64emu_t temp(*this); this = this - fp64emu_t(1.0); return temp; }
        // Compound assignment operators
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_t& operator+=(const fp64emu_t& other) { *this = *this + other; return *this; }
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_t& operator-=(const fp64emu_t& other) { *this = *this - other; return *this; }
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_t& operator*=(const fp64emu_t& other) { *this = *this * other; return *this; }
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_t& operator/=(const fp64emu_t& other) { *this = *this / other; return *this; }
        // Unary negation operator (implementation in fpemu_impl_others.hpp)
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_t  operator-() const;

        /*
        // Comparison operators:
        */       
        // equality (==)
        template<typename T1, typename T2>
            __FPEMU_HOST_DEVICE_DECL__ friend typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value)), bool>::type
            operator==(const T1& x, const T2& y) { return fp64emu_t(x) == fp64emu_t(y); }
        // inequality (!=)
        template<typename T1, typename T2>
            __FPEMU_HOST_DEVICE_DECL__ friend typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value)), bool>::type
            operator!=(const T1& x, const T2& y) { return fp64emu_t(x) != fp64emu_t(y); }
        // less than (<)
        template<typename T1, typename T2>
            __FPEMU_HOST_DEVICE_DECL__ friend typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value)), bool>::type
            operator<(const T1& x, const T2& y) { return fp64emu_t(x) < fp64emu_t(y); }
        // greater than (>)
        template<typename T1, typename T2>
            __FPEMU_HOST_DEVICE_DECL__ friend typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value)), bool>::type
            operator>(const T1& x, const T2& y) { return fp64emu_t(x) > fp64emu_t(y); }
        // less than or equal to (<=)
        template<typename T1, typename T2>
            __FPEMU_HOST_DEVICE_DECL__ friend typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value)), bool>::type
            operator<=(const T1& x, const T2& y) { return fp64emu_t(x) <= fp64emu_t(y); }
        // greater than or equal to (>=)
        template<typename T1, typename T2>
            __FPEMU_HOST_DEVICE_DECL__ friend typename std::enable_if<((std::is_same<T1,fp64emu_t>::value || std::is_same<T2,fp64emu_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value)), bool>::type
            operator>=(const T1& x, const T2& y) { return fp64emu_t(x) >= fp64emu_t(y); }
        // Friend function for stream insertion
        __FPEMU_HOST_DEVICE_DECL__ friend ::std::ostream& operator<<(::std::ostream& os, const fp64emu_t& ef) { os << reinterpret_cast<const double&>(ef.bits); return os; }
    }; // class fp64emu_t 

#if __FPEMU_UNPACKED__ == 1

    template <fp64emu_accuracy met = fp64emu_accuracy::def> 
    class fp64emu_unpacked_t 
    {
     public:

        // Internal representation of the unpacked floating-point value
        // fpbits64_unpacked_t is defined in fpemu_common.hpp
        fpbits64_unpacked_t bits;
        
        /*
        // Constructors and assignment operators
        */
        // Basic constructors
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t() : bits{0u, 0, 0} {}
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t(fpbits64_construct_t, const fpbits64_unpacked_t& f) : bits(f) {}
        /*
        // Defaulted copy constructor (trivially copyable)
        // Note: NVCC implicitly makes defaulted special members __host__ __device__
        */
        fp64emu_unpacked_t(const fp64emu_unpacked_t& other) = default;

        /*
        // Copy constructor from volatile fp64emu_unpacked_t
        // Template so it is NOT a copy constructor per the C++ standard.
        // The volatile overloads are wrapped in dummy templates
        // so that the C++ standard does not consider them copy constructors/assignment
        // operators (a template is never a copy constructor or copy assignment operator),
        // preserving trivial copyability while retaining volatile access support.
        */
        template<typename Dummy = void>
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t(const volatile fp64emu_unpacked_t& other)
        { 
            bits.sign = other.bits.sign; 
            bits.exponent = other.bits.exponent; 
            bits.mantissa = other.bits.mantissa; 
        }

        // Defaulted copy assignment operator (trivially copyable)
        fp64emu_unpacked_t& operator=(const fp64emu_unpacked_t& other) = default;

        /*
        // Assignment operator to volatile fp64emu_unpacked_t
        // Template so it is NOT a copy assignment operator per the C++ standard
        // Returns void to avoid C++20 -Wvolatile (deprecated volatile return)
        */
        template<typename Dummy = void>
        __FPEMU_HOST_DEVICE_DECL__ inline void operator=(const fp64emu_unpacked_t& other) volatile
        { 
            bits.sign = other.bits.sign; 
            bits.exponent = other.bits.exponent; 
            bits.mantissa = other.bits.mantissa; 
        }

        /*
        // Assignment operator from volatile fp64emu_unpacked_t
        // Template so it is NOT a copy assignment operator per the C++ standard
        */
        template<typename Dummy = void>
        __FPEMU_HOST_DEVICE_DECL__ inline fp64emu_unpacked_t& operator=(const volatile fp64emu_unpacked_t& other)
        { 
            bits.sign = other.bits.sign; 
            bits.exponent = other.bits.exponent; 
            bits.mantissa = other.bits.mantissa; 
            return *this; 
        }
        /*
        // Conversion operators
        */
        // ==== Conversions from other types to fp64emu_unpacked_t:
#if defined __CUDACC__
        // Implicit conversions from floating-point types 
        __FPEMU_HOST_DEVICE_DECL__  inline fp64emu_unpacked_t(float f);
        __FPEMU_HOST_DEVICE_DECL__  inline fp64emu_unpacked_t(double d);        
        // Explicit conversions from integer types
        __FPEMU_HOST_DEVICE_DECL__  inline fp64emu_unpacked_t(int32_t i);
        __FPEMU_HOST_DEVICE_DECL__  inline fp64emu_unpacked_t(uint32_t i);
#else
        // Explicit conversions from floating-point types (to avoid ambiguity with packed type)
        __FPEMU_HOST_DEVICE_DECL__ explicit inline fp64emu_unpacked_t(float f);
        __FPEMU_HOST_DEVICE_DECL__ explicit inline fp64emu_unpacked_t(double d);        
        // Explicit conversions from integer types (to avoid ambiguity with packed type)
        __FPEMU_HOST_DEVICE_DECL__ explicit inline fp64emu_unpacked_t(int32_t i);
        __FPEMU_HOST_DEVICE_DECL__ explicit inline fp64emu_unpacked_t(uint32_t i);
#endif
        // Explicit conversions from 64-bit integers 
        // required due to ambiguity with other constructors
        __FPEMU_HOST_DEVICE_DECL__ explicit inline fp64emu_unpacked_t(int64_t i);
        __FPEMU_HOST_DEVICE_DECL__ explicit inline fp64emu_unpacked_t(uint64_t i);

        // Explicit conversion from long long int types when their range is wider than int64_t
        __FPEMU_HOST_DEVICE_DECL__ explicit inline fp64emu_unpacked_t(long long unsigned int i) { *this = fp64emu_unpacked_t((uint64_t)i); }
        __FPEMU_HOST_DEVICE_DECL__ explicit inline fp64emu_unpacked_t(long long  int i)         { *this = fp64emu_unpacked_t((int64_t)i);  }
        // Type conversion to fp64emu_unpacked_t with other accuracy and range
        template<fp64emu_accuracy m = met> __FPEMU_HOST_DEVICE_DECL__ inline operator fp64emu_unpacked_t<m>() const;
        // Type conversion from fp64emu_unpacked_t to fp64emu_t (explicit to avoid overload ambiguity)
        template<fp64emu_accuracy m = met> __FPEMU_HOST_DEVICE_DECL__ explicit inline operator fp64emu_t<m>() const;

        // ==== Conversion from fp64emu_unpacked_t to other types:
        // Implicit conversion to double
        __FPEMU_HOST_DEVICE_DECL__ inline operator double() const;
        // Explicit conversions to other types
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator float()    const;
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator int32_t()  const;
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator uint32_t() const;
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator int64_t()  const;
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator uint64_t() const;
        // Explicit conversion to long long int types when their range is wider than int64_t
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator long long unsigned int() const { return (uint64_t)(*this); }
        __FPEMU_HOST_DEVICE_DECL__ explicit inline operator long long int() const          { return (int64_t)(*this); }

        /*
        //  CUDA builtins functions for conversions
        */
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline float __double2float(fp64emu_unpacked_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline int32_t __double2int_rz(fp64emu_unpacked_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline uint32_t __double2uint_rz(fp64emu_unpacked_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline int64_t __double2ll_rz(fp64emu_unpacked_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline uint64_t __double2ull_rz(fp64emu_unpacked_t<m> x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline fp64emu_unpacked_t<m> __float2double (float x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline fp64emu_unpacked_t<m> __int2double   (int32_t x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline fp64emu_unpacked_t<m> __uint2double  (uint32_t x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline fp64emu_unpacked_t<m> __ll2double    (int64_t x);
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend inline fp64emu_unpacked_t<m> __ull2double   (uint64_t x);

        /*
        // Arithmetic operations:
        */
        // === mul ===
        // (*)
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend fp64emu_unpacked_t<m> operator*(const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y);
        // (/)
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend fp64emu_unpacked_t<m> operator/(const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y);
        // (+)
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend fp64emu_unpacked_t<m> operator+(const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y);
        // (-)
        template<fp64emu_accuracy m> __FPEMU_HOST_DEVICE_DECL__ friend fp64emu_unpacked_t<m> operator-(const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y);
        

        // == mul ==
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type> 
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t operator*(const T1& x, const T2& y) { return fp64emu_unpacked_t(x) * fp64emu_unpacked_t(y); }
        // dmul_rn
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t __dmul_rn(const T1& x, const T2& y) { return __dmul_rn(fp64emu_unpacked_t(x), fp64emu_unpacked_t(y)); }

        // === div ===
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type> 
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t operator/(const T1& x, const T2& y) { return fp64emu_unpacked_t(x) / fp64emu_unpacked_t(y); }
        // ddiv_rn
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t __ddiv_rn(const T1& x, const T2& y) { return __ddiv_rn(fp64emu_unpacked_t(x), fp64emu_unpacked_t(y)); }

        // === add ===
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type> 
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t operator+(const T1& x, const T2& y) { return fp64emu_unpacked_t(x) + fp64emu_unpacked_t(y); }
        // dadd_rn
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t __dadd_rn(const T1& x, const T2& y) {  return __dadd_rn(fp64emu_unpacked_t(x), fp64emu_unpacked_t(y)); }

        // === sub ===
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type> 
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t operator-(const T1& x, const T2& y) { return fp64emu_unpacked_t(x) - fp64emu_unpacked_t(y); }
        // dsub_rn
        template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t __dsub_rn(const T1& x, const T2& y) { return __dsub_rn(fp64emu_unpacked_t(x), fp64emu_unpacked_t(y)); }

        // === sqrt ===
        // sqrt
        template<typename T1, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t sqrt(const T1& x) { return sqrt(fp64emu_unpacked_t(x)); }        
        // dsqrt_rn
        template<typename T1, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t __dsqrt_rn(const T1& x) { return __dsqrt_rn(fp64emu_unpacked_t(x)); }

        // === fma ===
        // fma
        template<typename T1, typename T2, typename T3, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value || std::is_same<T3,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t fma(const T1& x, const T2& y, const T3& z) { return fma(fp64emu_unpacked_t(x), fp64emu_unpacked_t(y), fp64emu_unpacked_t(z)); }
        // dfma_rn
        template<typename T1, typename T2, typename T3, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value || std::is_same<T3,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t __fma_rn(const T1& x, const T2& y, const T3& z) { return __fma_rn(fp64emu_unpacked_t(x), fp64emu_unpacked_t(y), fp64emu_unpacked_t(z)); }

        // === mad ===
        // mad
        template<typename T1, typename T2, typename T3, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value || std::is_same<T3,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t mad(const T1& x, const T2& y, const T3& z) { return mad(fp64emu_unpacked_t(x), fp64emu_unpacked_t(y), fp64emu_unpacked_t(z)); }
        // dmad_rn
        template<typename T1, typename T2, typename T3, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value || std::is_same<T3,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t __mad_rn(const T1& x, const T2& y, const T3& z) { return __mad_rn(fp64emu_unpacked_t(x), fp64emu_unpacked_t(y), fp64emu_unpacked_t(z)); }

        // === dot ===
        template<typename T1, typename T2, typename T3, typename T4, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value || std::is_same<T3,fp64emu_unpacked_t>::value || std::is_same<T4,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value || std::is_arithmetic<T4>::value))>::type>
            __FPEMU_HOST_DEVICE_DECL__ friend  fp64emu_unpacked_t dot(const T1& x1, const T2& y1, const T3& x2, const T4& y2) { return dot(fp64emu_unpacked_t(x1), fp64emu_unpacked_t(y1), fp64emu_unpacked_t(x2), fp64emu_unpacked_t(y2)); }

         // === cmul ===
         template<typename T1, typename T2, typename T3, typename T4, typename = typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value || std::is_same<T3,fp64emu_unpacked_t>::value || std::is_same<T4,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value || std::is_arithmetic<T4>::value))>::type>
             __FPEMU_HOST_DEVICE_DECL__ friend void cmul(const T1& x_re, const T2& x_im, const T3& y_re, const T4& y_im, fp64emu_unpacked_t& r_re, fp64emu_unpacked_t& r_im) { cmul(fp64emu_unpacked_t(x_re), fp64emu_unpacked_t(x_im), fp64emu_unpacked_t(y_re), fp64emu_unpacked_t(y_im), r_re, r_im); }

        // Prefix increment/decrement
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_unpacked_t& operator++() { this = this + fp64emu_unpacked_t(1.0); return *this; }
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_unpacked_t& operator--() { this = this - fp64emu_unpacked_t(1.0); return *this; }
        // Postfix increment/decrement
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_unpacked_t  operator++(int) { fp64emu_unpacked_t temp(*this); this = this + fp64emu_unpacked_t(1.0); return temp; }
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_unpacked_t  operator--(int) { fp64emu_unpacked_t temp(*this); this = this - fp64emu_unpacked_t(1.0); return temp; }
        // Compound assignment operators
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_unpacked_t& operator+=(const fp64emu_unpacked_t& other) { *this = *this + other; return *this; }
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_unpacked_t& operator-=(const fp64emu_unpacked_t& other) { *this = *this - other; return *this; }
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_unpacked_t& operator*=(const fp64emu_unpacked_t& other) { *this = *this * other; return *this; }
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_unpacked_t& operator/=(const fp64emu_unpacked_t& other) { *this = *this / other; return *this; }
        // Unary negation operator (implementation in fpemu_impl_others.hpp)
        __FPEMU_HOST_DEVICE_DECL__ fp64emu_unpacked_t  operator-() const;

        /*
        // Comparison operators:
        */       
        // equality (==)
        template<typename T1, typename T2>
            __FPEMU_HOST_DEVICE_DECL__ friend typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value)), bool>::type
            operator==(const T1& x, const T2& y) { return fp64emu_unpacked_t(x) == fp64emu_unpacked_t(y); }
        // inequality (!=)
        template<typename T1, typename T2>
            __FPEMU_HOST_DEVICE_DECL__ friend typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value)), bool>::type
            operator!=(const T1& x, const T2& y) { return fp64emu_unpacked_t(x) != fp64emu_unpacked_t(y); }
        // less than (<)
        template<typename T1, typename T2>
            __FPEMU_HOST_DEVICE_DECL__ friend typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value)), bool>::type
            operator<(const T1& x, const T2& y) { return fp64emu_unpacked_t(x) < fp64emu_unpacked_t(y); }
        // greater than (>)
        template<typename T1, typename T2>
            __FPEMU_HOST_DEVICE_DECL__ friend typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value)), bool>::type
            operator>(const T1& x, const T2& y) { return fp64emu_unpacked_t(x) > fp64emu_unpacked_t(y); }
        // less than or equal to (<=)
        template<typename T1, typename T2>
            __FPEMU_HOST_DEVICE_DECL__ friend typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value)), bool>::type
            operator<=(const T1& x, const T2& y) { return fp64emu_unpacked_t(x) <= fp64emu_unpacked_t(y); }
        // greater than or equal to (>=)
        template<typename T1, typename T2>
            __FPEMU_HOST_DEVICE_DECL__ friend typename std::enable_if<((std::is_same<T1,fp64emu_unpacked_t>::value || std::is_same<T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value)), bool>::type
            operator>=(const T1& x, const T2& y) { return fp64emu_unpacked_t(x) >= fp64emu_unpacked_t(y); }

        // Friend function for stream insertion (defined after impl includes)
        template<fp64emu_accuracy m>
        __FPEMU_HOST_DEVICE_DECL__ friend ::std::ostream& operator<<(::std::ostream& os, const fp64emu_unpacked_t<m>& ef);

        // C++20-style bit_cast for unpacked floating-point types
        template<typename To, fp64emu_accuracy m> 
        __FPEMU_HOST_DEVICE_DECL__ friend inline To bit_cast(const fp64emu_unpacked_t<m>& from);

    }; // class fp64emu_unpacked_t 
#endif // __FPEMU_UNPACKED__ == 1

    /*
    // Aliases for the emulated floating-point types
    */
    using fp64emu      = fp64emu_t<fp64emu_accuracy::def>;
    using fp64emu_low  = fp64emu_t<fp64emu_accuracy::low>;
    using fp64emu_mid  = fp64emu_t<fp64emu_accuracy::mid>;
    using fp64emu_high = fp64emu_t<fp64emu_accuracy::high>;
#if __FPEMU_UNPACKED__ == 1
    using fp64emu_unpacked      = fp64emu_unpacked_t<fp64emu_accuracy::def>;
    using fp64emu_unpacked_low  = fp64emu_unpacked_t<fp64emu_accuracy::low>;
    using fp64emu_unpacked_mid  = fp64emu_unpacked_t<fp64emu_accuracy::mid>;
    using fp64emu_unpacked_high = fp64emu_unpacked_t<fp64emu_accuracy::high>;
#endif

// Define this macro so that the API sections in _impl.hpp files are activated.
// The _impl.hpp files are structured with implementation code under their own
// include guard, and API code (operators, class methods) under this guard.
// This ensures API code is only compiled after class definitions are complete.
#define __FPEMU_API_CLASSES_DEFINED__

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#include <cuda/__fp/fpemu_impl_cmp.hpp>
#include <cuda/__fp/fpemu_impl_cvt.hpp>
#include <cuda/__fp/fpemu_impl_fma.hpp>
#include <cuda/__fp/fpemu_impl_add.hpp>
#include <cuda/__fp/fpemu_impl_sub.hpp>
#include <cuda/__fp/fpemu_impl_mul.hpp>
#include <cuda/__fp/fpemu_impl_div.hpp>
#include <cuda/__fp/fpemu_impl_sqrt.hpp>
#include <cuda/__fp/fpemu_impl_others.hpp>

#endif // _CUDA___FP_FPEMU_H
