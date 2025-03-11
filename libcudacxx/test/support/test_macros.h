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
#  if __has_include(<cuda/__cccl_config>)
#    include <cuda/__cccl_config>
#    include <cuda/std/__internal/features.h>
#  elif __has_include("<__config>")
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

#ifdef _CCCL_HAS_FEATURE
#  define TEST_HAS_FEATURE(X) _CCCL_HAS_FEATURE(X)
#else
#  define TEST_HAS_FEATURE(X) 0
#endif

#ifndef __has_include
#  define __has_include(...) 0
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

#if defined(__NVCOMPILER)
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

#if _CCCL_CUDA_COMPILER(NVCC)
#  define TEST_COMPILER_NVCC
#  define TEST_COMPILER_EDG
#elif _CCCL_CUDA_COMPILER(NVHPC)
#  define TEST_COMPILER_NVHPC_CUDA
#elif _CCCL_CUDA_COMPILER(CLANG)
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

#if TEST_HAS_BUILTIN(__builtin_is_constant_evaluated) || _CCCL_COMPILER(GCC, >=, 9) \
  || (_CCCL_COMPILER(MSVC) && _MSC_VER > 1924)
#  define TEST_IS_CONSTANT_EVALUATED() cuda::std::is_constant_evaluated()
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

// Sniff out to see if the underling C library has C11 features
// Note that at this time (July 2018), MacOS X and iOS do NOT.
// This is cribbed from __config; but lives here as well because we can't assume libc++
#if defined(__linux__)
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

#define TEST_ALIGNAS_TYPE(...) TEST_ALIGNAS(TEST_ALIGNOF(__VA_ARGS__))

#if !TEST_HAS_FEATURE(cxx_rtti) && !defined(__cpp_rtti) && !defined(__GXX_RTTI)
#  define TEST_HAS_NO_RTTI
#endif

#ifndef TEST_HAS_NO_EXCEPTIONS
#  if (_CCCL_COMPILER(MSVC) && _HAS_EXCEPTIONS == 0) || (!_CCCL_COMPILER(MSVC) && !__EXCEPTIONS) // Catches all non
                                                                                                 // msvc based
                                                                                                 // compilers
#    define TEST_HAS_NO_EXCEPTIONS
#  endif
#endif // !TEST_HAS_NO_EXCEPTIONS

#if defined(TEST_COMPILER_NVCC) || defined(TEST_COMPILER_NVRTC)
#  define TEST_HAS_NO_EXCEPTIONS
#endif

#if TEST_HAS_FEATURE(address_sanitizer) || TEST_HAS_FEATURE(memory_sanitizer) || TEST_HAS_FEATURE(thread_sanitizer)
#  define TEST_HAS_SANITIZERS
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
#  define TEST_NV_DIAG_SUPPRESS(WARNING) _CCCL_PRAGMA(nv_diag_suppress WARNING)
#else // ^^^ __NVCC_DIAG_PRAGMA_SUPPORT__ ^^^ / vvv !__NVCC_DIAG_PRAGMA_SUPPORT__ vvv
#  define TEST_NV_DIAG_SUPPRESS(WARNING) _CCCL_PRAGMA(diag_suppress WARNING)
#endif

#if defined(TEST_COMPILER_MSVC)
#  if _MSC_VER < 1920
#    error "MSVC version not supported"
#  elif _MSC_VER < 1930
#    define TEST_COMPILER_MSVC_2019
#  else
#    define TEST_COMPILER_MSVC_2022
#  endif
#endif // defined(TEST_COMPILER_MSVC)

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

#endif // SUPPORT_TEST_MACROS_HPP
