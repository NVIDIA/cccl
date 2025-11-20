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

#include <cuda/std/detail/__config>

// Use the CCCL compiler detection
#define TEST_COMPILER(...)       _CCCL_COMPILER(__VA_ARGS__)
#define TEST_CUDA_COMPILER(...)  _CCCL_CUDA_COMPILER(__VA_ARGS__)
#define TEST_HAS_CUDA_COMPILER() _CCCL_HAS_CUDA_COMPILER()

// Use the CCCL diagnostic suppression
#define TEST_DIAG_SUPPRESS_CLANG(...) _CCCL_DIAG_SUPPRESS_CLANG(__VA_ARGS__)
#define TEST_DIAG_SUPPRESS_GCC(...)   _CCCL_DIAG_SUPPRESS_GCC(__VA_ARGS__)
#define TEST_DIAG_SUPPRESS_NVHPC(...) _CCCL_DIAG_SUPPRESS_NVHPC(__VA_ARGS__)
#define TEST_DIAG_SUPPRESS_MSVC(...)  _CCCL_DIAG_SUPPRESS_MSVC(__VA_ARGS__)
#define TEST_NV_DIAG_SUPPRESS(...)    _CCCL_BEGIN_NV_DIAG_SUPPRESS(__VA_ARGS__)

// Use the CCCL C++ dialect detection
#define TEST_STD_VER _CCCL_STD_VER

// Use the CCCL constexpr macros
#define TEST_CONSTEXPR_CXX20 _CCCL_CONSTEXPR_CXX20
#define TEST_CONSTEXPR_CXX23 _CCCL_CONSTEXPR_CXX23

// Use the CCCL global variable hack
#define TEST_GLOBAL_VARIABLE static _CCCL_GLOBAL_VARIABLE

// Use the CCCL exceptions detection
#define TEST_HAS_EXCEPTIONS() _CCCL_HAS_EXCEPTIONS()

#if TEST_HAS_EXCEPTIONS()
#  define TEST_THROW(...) throw __VA_ARGS__
#else
#  define TEST_THROW(...) assert(#__VA_ARGS__)
#endif

// Use the CCCL spaceship detection
#define TEST_HAS_SPACESHIP() _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
#  define TEST_IS_CONSTANT_EVALUATED() _CCCL_BUILTIN_IS_CONSTANT_EVALUATED()
#else
#  define TEST_IS_CONSTANT_EVALUATED() false
#endif

#if TEST_STD_VER >= 2023
#  define TEST_IS_CONSTANT_EVALUATED_CXX23() TEST_IS_CONSTANT_EVALUATED()
#else // ^^^ C++23 ^^^ / vvv C++20 vvv
#  define TEST_IS_CONSTANT_EVALUATED_CXX23() false
#endif // ^^^ TEST_STD_VER <= 2020

// Attempt to deduce the GLIBC version
#if _CCCL_HAS_INCLUDE(<features.h>) || defined(__linux__)
#  include <features.h>
#  if defined(__GLIBC_PREREQ)
#    define TEST_HAS_GLIBC
#    define TEST_GLIBC_PREREQ(major, minor) __GLIBC_PREREQ(major, minor)
#  endif
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
#  endif
#endif

#if !_CCCL_HAS_FEATURE(cxx_rtti) && !defined(__cpp_rtti) && !defined(__GXX_RTTI)
#  define TEST_HAS_NO_RTTI
#endif

#if _CCCL_HAS_FEATURE(address_sanitizer) || _CCCL_HAS_FEATURE(memory_sanitizer) || _CCCL_HAS_FEATURE(thread_sanitizer)
#  define TEST_HAS_SANITIZERS
#endif

#define TEST_IGNORE_NODISCARD (void)

#if TEST_COMPILER(NVRTC, >=, 13)
#  define TEST_NVRTC_VIRTUAL_DEFAULT_DTOR_ANNOTATION __host__ __device__
#else
#  define TEST_NVRTC_VIRTUAL_DEFAULT_DTOR_ANNOTATION
#endif

#if TEST_COMPILER(MSVC)
#  include <intrin.h>
template <class Tp>
inline void DoNotOptimize(Tp const& value)
{
  [[maybe_unused]] const volatile void* volatile unused = __builtin_addressof(value);
  _ReadWriteBarrier();
}
#else // ^^^ TEST_COMPILER(MSVC) ^^^ / vvv !TEST_COMPILER(MSVC) vvv
template <class Tp>
__host__ __device__ inline void DoNotOptimize(Tp const& value)
{
  asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
__host__ __device__ inline void DoNotOptimize(Tp& value)
{
#  if TEST_COMPILER(CLANG)
  asm volatile("" : "+r,m"(value) : : "memory");
#  else
  asm volatile("" : "+m,r"(value) : : "memory");
#  endif
}
#endif // !TEST_COMPILER(MSVC)

// NVCC can't handle static member variables, so with a little care
// a function returning a reference will result in the same thing
#ifdef __CUDA_ARCH__
#  define _STATIC_MEMBER_IMPL(type) __shared__ type v;
#else
#  define _STATIC_MEMBER_IMPL(type) static type v;
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

#endif // SUPPORT_TEST_MACROS_HPP
