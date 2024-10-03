// -*- C++ -*-
//===---------------------------- test_macros.h ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_MACROS_HPP
#define SUPPORT_TEST_MACROS_HPP

// Attempt to get STL specific macros like _LIBCUDACXX_VERSION using the most
// minimal header possible. If we're testing libc++, we should use `<__config>`.
// If <__config> isn't available, fall back to <ciso646>.
#ifdef __has_include
#  if __has_include("<__config>")
#    include <__config>
#    define TEST_IMP_INCLUDED_HEADER
#  endif
#endif
#ifndef TEST_IMP_INCLUDED_HEADER
#  ifndef __CUDACC_RTC__
#    include <ciso646>
#  endif // __CUDACC_RTC__
#endif

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wvariadic-macros"
#endif

#define TEST_CONCAT1(X, Y) X##Y
#define TEST_CONCAT(X, Y)  TEST_CONCAT1(X, Y)

#ifdef _CCCL_HAS_FEATURE
#  define TEST_HAS_FEATURE(X) _CCCL_HAS_FEATURE(X)
#else
#  define TEST_HAS_FEATURE(X) 0
#endif

#ifndef __has_include
#  define __has_include(...) 0
#endif

#ifdef __has_extension
#  define TEST_HAS_EXTENSION(X) __has_extension(X)
#else
#  define TEST_HAS_EXTENSION(X) 0
#endif

#ifdef __has_builtin
#  define TEST_HAS_BUILTIN(X) __has_builtin(X)
#else
#  define TEST_HAS_BUILTIN(X) 0
#endif
#ifdef __is_identifier
// '__is_identifier' returns '0' if '__x' is a reserved identifier provided by
// the compiler and '1' otherwise.
#  define TEST_HAS_BUILTIN_IDENTIFIER(X) !__is_identifier(X)
#else
#  define TEST_HAS_BUILTIN_IDENTIFIER(X) 0
#endif

#if defined(__INTEL_COMPILER)
#  define TEST_COMPILER_ICC
#elif defined(__NVCOMPILER)
#  define TEST_COMPILER_NVHPC
#elif defined(__clang__)
#  define TEST_COMPILER_CLANG
#  if defined(__apple_build_version__)
#    define TEST_COMPILER_APPLE_CLANG
#  endif
#elif defined(__GNUC__)
#  define TEST_COMPILER_GCC
#elif defined(_MSC_VER)
#  define TEST_COMPILER_MSVC
#elif defined(__CUDACC_RTC__)
#  define TEST_COMPILER_NVRTC
#elif defined(__EDG__)
#  define TEST_COMPILER_EDG
#endif

#if defined(_CCCL_CUDA_COMPILER_NVCC)
#  define TEST_COMPILER_NVCC
#  define TEST_COMPILER_EDG
#elif defined(_CCCL_CUDA_COMPILER_NVHPC)
#  define TEST_COMPILER_NVHPC_CUDA
#elif defined(_CCCL_CUDA_COMPILER_CLANG)
#  define TEST_COMPILER_CLANG_CUDA
#endif // no cuda compiler

#if defined(__apple_build_version__)
#  define TEST_APPLE_CLANG_VER (__clang_major__ * 100) + __clang_minor__
#elif defined(__clang_major__)
#  define TEST_CLANG_VER (__clang_major__ * 100) + __clang_minor__
#elif defined(__GNUC__)
#  define TEST_GCC_VER     (__GNUC__ * 100 + __GNUC_MINOR__)
#  define TEST_GCC_VER_NEW (TEST_GCC_VER * 10 + __GNUC_PATCHLEVEL__)
#endif

/* Make a nice name for the standard version */
#ifndef TEST_STD_VER
#  if defined(TEST_COMPILER_MSVC)
#    if !defined(_MSVC_LANG)
#      define TEST_STD_VER 2003
#    elif _MSVC_LANG <= 201103L
#      define TEST_STD_VER 2011
#    elif _MSVC_LANG <= 201402L
#      define TEST_STD_VER 2014
#    elif _MSVC_LANG <= 201703L
#      define TEST_STD_VER 2017
#    elif _MSVC_LANG <= 202002L
#      define TEST_STD_VER 2020
#    else
#      define TEST_STD_VER 2099 // Greater than current standard.
// This is deliberately different than _CCCL_STD_VER to discourage matching them up.
#    endif
#  else
#    if __cplusplus <= 199711L
#      define TEST_STD_VER 2003
#    elif __cplusplus <= 201103L
#      define TEST_STD_VER 2011
#    elif __cplusplus <= 201402L
#      define TEST_STD_VER 2014
#    elif __cplusplus <= 201703L
#      define TEST_STD_VER 2017
#    elif __cplusplus <= 202002L
#      define TEST_STD_VER 2020
#    else
#      define TEST_STD_VER 2099 // Greater than current standard.
// This is deliberately different than _CCCL_STD_VER to discourage matching them up.
#    endif
#  endif
#endif // TEST_STD_VER

// Attempt to deduce the GLIBC version
#if (defined(__has_include) && __has_include(<features.h>)) || \
    defined(__linux__)
#  include <features.h>
#  if defined(__GLIBC_PREREQ)
#    define TEST_HAS_GLIBC
#    define TEST_GLIBC_PREREQ(major, minor) __GLIBC_PREREQ(major, minor)
#  endif
#endif

#if TEST_HAS_BUILTIN(__builtin_is_constant_evaluated) || (defined(_CCCL_COMPILER_GCC) && _GNUC_VER >= 900) \
  || (defined(_CCCL_COMPILER_MSVC) && _MSC_VER > 1924 && !defined(_CCCL_CUDACC_BELOW_11_3))
#  define TEST_IS_CONSTANT_EVALUATED() _CUDA_VSTD::__libcpp_is_constant_evaluated()
#else
#  define TEST_IS_CONSTANT_EVALUATED() false
#endif

#if TEST_STD_VER >= 2023
#  define TEST_IS_CONSTANT_EVALUATED_CXX23() TEST_IS_CONSTANT_EVALUATED()
#else // ^^^ C++23 ^^^ / vvv C++20 vvv
#  define TEST_IS_CONSTANT_EVALUATED_CXX23() false
#endif // ^^^ TEST_STD_VER <= 2020

#define TEST_ALIGNOF(...)       alignof(__VA_ARGS__)
#define TEST_ALIGNAS(...)       alignas(__VA_ARGS__)
#define TEST_CONSTEXPR          constexpr
#define TEST_NOEXCEPT           noexcept
#define TEST_NOEXCEPT_FALSE     noexcept(false)
#define TEST_NOEXCEPT_COND(...) noexcept(__VA_ARGS__)
#if TEST_STD_VER >= 2014
#  define TEST_CONSTEXPR_CXX14 constexpr
#else
#  define TEST_CONSTEXPR_CXX14
#endif
#if TEST_STD_VER >= 2017
#  define TEST_CONSTEXPR_CXX17 constexpr
#else
#  define TEST_CONSTEXPR_CXX17
#endif
#if TEST_STD_VER >= 2020
#  define TEST_CONSTEXPR_CXX20 constexpr
#else
#  define TEST_CONSTEXPR_CXX20
#endif
#if TEST_STD_VER >= 2023
#  define TEST_CONSTEXPR_CXX23 constexpr
#else
#  define TEST_CONSTEXPR_CXX23
#endif
#if TEST_STD_VER > 2014
#  define TEST_THROW_SPEC(...)
#else
#  define TEST_THROW_SPEC(...) throw(__VA_ARGS__)
#endif

#if defined(TEST_COMPILER_NVRTC)
#  define NEW_IS_NOEXCEPT_NVRTC noexcept
#else
#  define NEW_IS_NOEXCEPT_NVRTC
#endif // TEST_COMPILER_NVRTC

// Sniff out to see if the underling C library has C11 features
// Note that at this time (July 2018), MacOS X and iOS do NOT.
// This is cribbed from __config; but lives here as well because we can't assume libc++
#if defined(__FreeBSD__)
//  Specifically, FreeBSD does NOT have timespec_get, even though they have all
//  the rest of C11 - this is PR#38495
#  define TEST_HAS_C11_FEATURES
#elif defined(__Fuchsia__) || defined(__wasi__)
#  define TEST_HAS_C11_FEATURES
#  define TEST_HAS_TIMESPEC_GET
#elif defined(__linux__)
// This block preserves the old behavior used by include/__config:
// _LIBCUDACXX_GLIBC_PREREQ would be defined to 0 if __GLIBC_PREREQ was not
// available. The configuration here may be too vague though, as Bionic, uClibc,
// newlib, etc may all support these features but need to be configured.
#  if defined(TEST_GLIBC_PREREQ)
#    if TEST_GLIBC_PREREQ(2, 17)
#      define TEST_HAS_TIMESPEC_GET
#      define TEST_HAS_C11_FEATURES
#    endif
#  elif defined(_LIBCUDACXX_HAS_MUSL_LIBC)
#    define TEST_HAS_C11_FEATURES
#    define TEST_HAS_TIMESPEC_GET
#  endif
#endif

/* Features that were introduced in C++14 */
#if TEST_STD_VER >= 2014
#  define TEST_HAS_EXTENDED_CONSTEXPR
#  define TEST_HAS_VARIABLE_TEMPLATES
#endif

/* Features that were introduced in C++17 */
#if TEST_STD_VER >= 2017
#endif

/* Features that were introduced after C++17 */
#if TEST_STD_VER > 2017
#endif

#define TEST_ALIGNAS_TYPE(...) TEST_ALIGNAS(TEST_ALIGNOF(__VA_ARGS__))

#if !TEST_HAS_FEATURE(cxx_rtti) && !defined(__cpp_rtti) && !defined(__GXX_RTTI)
#  define TEST_HAS_NO_RTTI
#endif

#ifndef TEST_HAS_NO_EXCEPTIONS
#  if (defined(_CCCL_COMPILER_MSVC) && _HAS_EXCEPTIONS == 0) \
    || (!defined(_CCCL_COMPILER_MSVC) && !__EXCEPTIONS) // Catches all non msvc based compilers
#    define TEST_HAS_NO_EXCEPTIONS
#  endif
#endif // !TEST_HAS_NO_EXCEPTIONS

#if defined(TEST_COMPILER_NVCC) || defined(TEST_COMPILER_NVRTC)
#  define TEST_HAS_NO_EXCEPTIONS
#endif

#if TEST_HAS_FEATURE(address_sanitizer) || TEST_HAS_FEATURE(memory_sanitizer) || TEST_HAS_FEATURE(thread_sanitizer)
#  define TEST_HAS_SANITIZERS
#endif

#if defined(_LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION) \
  || (!(TEST_STD_VER > 2014 || (defined(__cpp_aligned_new) && __cpp_aligned_new >= 201606L)))
#  define TEST_HAS_NO_ALIGNED_ALLOCATION
#endif

#if defined(_LIBCUDACXX_SAFE_STATIC)
#  define TEST_SAFE_STATIC _LIBCUDACXX_SAFE_STATIC
#else
#  define TEST_SAFE_STATIC
#endif

#if defined(_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR)
#  define TEST_HAS_NO_SPACESHIP_OPERATOR
#endif

#define ASSERT_NOEXCEPT(...) static_assert(noexcept(__VA_ARGS__), "Operation must be noexcept")

#define ASSERT_NOT_NOEXCEPT(...) static_assert(!noexcept(__VA_ARGS__), "Operation must NOT be noexcept")

#if TEST_STD_VER > 2011
#  define STATIC_ASSERT_CXX14(Pred) static_assert(Pred, "")
#else
#  define STATIC_ASSERT_CXX14(Pred) assert(Pred)
#endif

/* Macros for testing libc++ specific behavior and extensions */
#if defined(_LIBCUDACXX_VERSION)
#  define LIBCPP_ASSERT(...)              assert(__VA_ARGS__)
#  define LIBCPP_STATIC_ASSERT(...)       static_assert(__VA_ARGS__)
#  define LIBCPP_ASSERT_NOEXCEPT(...)     ASSERT_NOEXCEPT(__VA_ARGS__)
#  define LIBCPP_ASSERT_NOT_NOEXCEPT(...) ASSERT_NOT_NOEXCEPT(__VA_ARGS__)
#  define LIBCPP_ONLY(...)                __VA_ARGS__
#else
#  define LIBCPP_ASSERT(...)              ((void) 0)
#  define LIBCPP_STATIC_ASSERT(...)       ((void) 0)
#  define LIBCPP_ASSERT_NOEXCEPT(...)     ((void) 0)
#  define LIBCPP_ASSERT_NOT_NOEXCEPT(...) ((void) 0)
#  define LIBCPP_ONLY(...)                ((void) 0)
#endif

#ifndef __has_cpp_attribute
#  define __has_cpp_attribute(__x) 0
#endif // !__has_cpp_attribute

#if __has_cpp_attribute(nodiscard)
#  define TEST_NODISCARD [[nodiscard]]
#else
#  define TEST_NODISCARD
#endif

#if defined(TEST_COMPILER_MSVC)
#  define TEST_NORETURN __declspec(noreturn)
#elif __has_cpp_attribute(noreturn)
#  define TEST_NORETURN [[noreturn]]
#else
#  define TEST_NORETURN __attribute__((noreturn))
#endif

#define TEST_IGNORE_NODISCARD (void)

namespace test_macros_detail
{
template <class T, class U>
struct is_same
{
  enum
  {
    value = 0
  };
};
template <class T>
struct is_same<T, T>
{
  enum
  {
    value = 1
  };
};
} // namespace test_macros_detail

#define ASSERT_SAME_TYPE(...) \
  static_assert((test_macros_detail::is_same<__VA_ARGS__>::value), "Types differ unexpectedly")

#ifndef TEST_HAS_NO_EXCEPTIONS
#  define TEST_THROW(...) throw __VA_ARGS__
#else
#  define TEST_THROW(...) assert(#__VA_ARGS__)
#endif

#ifndef TEST_HAS_NO_INT128_T
#  ifdef _LIBCUDACXX_HAS_NO_INT128
#    define TEST_HAS_NO_INT128_T
#  endif
#endif

#if defined(_LIBCUDACXX_HAS_NO_LOCALIZATION)
#  define TEST_HAS_NO_LOCALIZATION
#endif

#if defined(_LIBCUDACXX_NO_HAS_CHAR8_T)
#  define TEST_HAS_NO_CHAR8_T
#endif

#if defined(_LIBCUDACXX_HAS_NO_UNICODE_CHARS)
#  define TEST_HAS_NO_UNICODE_CHARS
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(TEST_COMPILER_NVRTC)
template <class Tp>
__host__ __device__ inline void DoNotOptimize(Tp const& value)
{
  asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
__host__ __device__ inline void DoNotOptimize(Tp& value)
{
#  if defined(__clang__)
  asm volatile("" : "+r,m"(value) : : "memory");
#  else
  asm volatile("" : "+m,r"(value) : : "memory");
#  endif
}
#else
#  include <intrin.h>
template <class Tp>
inline void DoNotOptimize(Tp const& value)
{
  const volatile void* volatile unused = __builtin_addressof(value);
  static_cast<void>(unused);
  _ReadWriteBarrier();
}
#endif

#if defined(__GNUC__)
#  define TEST_ALWAYS_INLINE __attribute__((always_inline))
#  define TEST_NOINLINE      __attribute__((noinline))
#elif defined(_MSC_VER)
#  define TEST_ALWAYS_INLINE __forceinline
#  define TEST_NOINLINE      __declspec(noinline)
#else
#  define TEST_ALWAYS_INLINE
#  define TEST_NOINLINE
#endif

// NVCC can't handle static member variables, so with a little care
// a function returning a reference will result in the same thing
#ifdef __CUDA_ARCH__
#  define _STATIC_MEMBER_IMPL(type) __shared__ type v;
#else
#  define _STATIC_MEMBER_IMPL(type) static type v;
#endif

#if defined(__CUDA_ARCH__)
#  define STATIC_TEST_GLOBAL_VAR static __device__
#else
#  define STATIC_TEST_GLOBAL_VAR
#endif

#if defined(__CUDA_ARCH__)
#  define TEST_ACCESSIBLE __device__
#else
#  define TEST_ACCESSIBLE
#endif

#define TEST_EXEC_CHECK_DISABLE _CCCL_EXEC_CHECK_DISABLE

#define STATIC_MEMBER_VAR(name, type)     \
  __host__ __device__ static type& name() \
  {                                       \
    _STATIC_MEMBER_IMPL(type);            \
    return v;                             \
  }

template <class... T>
__host__ __device__ constexpr bool unused(T&&...)
{
  return true;
}

// Define a helper macro to properly suppress warnings
#define _TEST_TOSTRING2(x) #x
#define _TEST_TOSTRING(x)  _TEST_TOSTRING2(x)
#if defined(TEST_COMPILER_CLANG_CUDA)
#  define TEST_NV_DIAG_SUPPRESS(WARNING)
#elif defined(__NVCC_DIAG_PRAGMA_SUPPORT__)
#  if defined(TEST_COMPILER_MSVC)
#    define TEST_NV_DIAG_SUPPRESS(WARNING) __pragma(_TEST_TOSTRING(nv_diag_suppress WARNING))
#  else // ^^^ MSVC ^^^ / vvv not MSVC vvv
#    define TEST_NV_DIAG_SUPPRESS(WARNING) _Pragma(_TEST_TOSTRING(nv_diag_suppress WARNING))
#  endif // not MSVC
#else // ^^^ __NVCC_DIAG_PRAGMA_SUPPORT__ ^^^ / vvv !__NVCC_DIAG_PRAGMA_SUPPORT__ vvv
#  if defined(TEST_COMPILER_MSVC)
#    define TEST_NV_DIAG_SUPPRESS(WARNING) __pragma(_TEST_TOSTRING(diag_suppress WARNING))
#  else // ^^^ MSVC ^^^ / vvv not MSVC vvv
#    define TEST_NV_DIAG_SUPPRESS(WARNING) _Pragma(_TEST_TOSTRING(diag_suppress WARNING))
#  endif // not MSVC
#endif

#define TEST_CONSTEXPR_GLOBAL _CCCL_CONSTEXPR_GLOBAL

// Some convenience macros for checking nvcc versions
#if defined(__CUDACC__) && _CCCL_CUDACC_VER < 1103000
#  define TEST_COMPILER_CUDACC_BELOW_11_3
#endif // defined(__CUDACC__) && _CCCL_CUDACC_VER < 1103000
#if defined(__CUDACC__) && _CCCL_CUDACC_VER < 1203000
#  define TEST_COMPILER_CUDACC_BELOW_12_3
#endif // defined(__CUDACC__) && _CCCL_CUDACC_VER < 1203000

#if defined(TEST_COMPILER_MSVC)
#  if _MSC_VER < 1920
#    define TEST_COMPILER_MSVC_2017
#  elif _MSC_VER < 1930
#    define TEST_COMPILER_MSVC_2019
#  else
#    define TEST_COMPILER_MSVC_2022
#  endif
#endif // defined(TEST_COMPILER_MSVC)

#if defined(TEST_COMPILER_NVHPC) || defined(TEST_COMPILER_ICC)
#  define TEST_COMPILER_BROKEN_SMF_NOEXCEPT
#endif // TEST_COMPILER_NVHPC || TEST_COMPILER_ICC

#if (defined(TEST_WINDOWS_DLL) && !defined(_MSC_VER)) || defined(__MVS__)
// Normally, a replaced e.g. 'operator new' ends up used if the user code
// does a call to e.g. 'operator new[]'; it's enough to replace the base
// versions and have it override all of them.
//
// When the fallback operators are located within the libc++ library and we
// can't override the calls within it (see above), this fallback mechanism
// doesn't work either.
//
// On Windows, when using the MSVC vcruntime, the operator new/delete fallbacks
// are linked separately from the libc++ library, linked statically into
// the end user executable, and these fallbacks work even in DLL configurations.
// In MinGW configurations when built as a DLL, and on zOS, these fallbacks
// don't work though.
#  define ASSERT_WITH_OPERATOR_NEW_FALLBACKS(...) ((void) (__VA_ARGS__))
#else
#  define ASSERT_WITH_OPERATOR_NEW_FALLBACKS(...) assert(__VA_ARGS__)
#endif

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

#endif // SUPPORT_TEST_MACROS_HPP
