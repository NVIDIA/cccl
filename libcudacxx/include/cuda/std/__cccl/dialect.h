//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_DIALECT_H
#define __CCCL_DIALECT_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/builtin.h>
#include <cuda/std/__cccl/host_std_lib.h>

///////////////////////////////////////////////////////////////////////////////
// Determine the C++ standard dialect
///////////////////////////////////////////////////////////////////////////////
#if _CCCL_COMPILER(MSVC)
#  if _MSVC_LANG <= 201103L
#    define _CCCL_STD_VER 2011
#  elif _MSVC_LANG <= 201402L
#    define _CCCL_STD_VER 2014
#  elif _MSVC_LANG <= 201703L
#    define _CCCL_STD_VER 2017
#  elif _MSVC_LANG <= 202002L
#    define _CCCL_STD_VER 2020
#  else
#    define _CCCL_STD_VER 2023 // current year, or date of c++2b ratification
#  endif
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#  if __cplusplus <= 199711L
#    define _CCCL_STD_VER 2003
#  elif __cplusplus <= 201103L
#    define _CCCL_STD_VER 2011
#  elif __cplusplus <= 201402L
#    define _CCCL_STD_VER 2014
#  elif __cplusplus <= 201703L
#    define _CCCL_STD_VER 2017
#  elif __cplusplus <= 202002L
#    define _CCCL_STD_VER 2020
#  elif __cplusplus <= 202302L
#    define _CCCL_STD_VER 2023
#  else
#    define _CCCL_STD_VER 2024 // current year, or date of c++2c ratification
#  endif
#endif // !_CCCL_COMPILER(MSVC)

///////////////////////////////////////////////////////////////////////////////
// Conditionally enable constexpr per standard dialect
///////////////////////////////////////////////////////////////////////////////

#if _CCCL_STD_VER >= 2020
#  define _CCCL_CONSTEXPR_CXX20 constexpr
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
#  define _CCCL_CONSTEXPR_CXX20
#endif // _CCCL_STD_VER <= 2017

#if _CCCL_STD_VER >= 2023
#  define _CCCL_CONSTEXPR_CXX23 constexpr
#else // ^^^ C++23 ^^^ / vvv C++20 vvv
#  define _CCCL_CONSTEXPR_CXX23
#endif // _CCCL_STD_VER <= 2020

///////////////////////////////////////////////////////////////////////////////
// Detect whether we can use some language features based on standard dialect
///////////////////////////////////////////////////////////////////////////////

// concepts are only available from C++20 onwards
#if _CCCL_STD_VER <= 2017 || __cpp_concepts < 201907L
#  define _CCCL_HAS_CONCEPTS() 0
#else // ^^^ no concepts ^^^ / vvv has concepts vvv
#  define _CCCL_HAS_CONCEPTS() 1
#endif // ^^^ has concepts ^^^

// Three way comparison is only available from C++20 onwards
#if _CCCL_STD_VER <= 2017 || __cpp_impl_three_way_comparison < 201907L
#  define _CCCL_NO_THREE_WAY_COMPARISON
#endif // _CCCL_STD_VER <= 2017 || __cpp_impl_three_way_comparison < 201907L

// Some compilers turn on pack indexing in pre-C++26 code. We want to use it if it is
// available.
#if defined(__cpp_pack_indexing) && !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(CLANG, <, 20)
#  define _CCCL_HAS_PACK_INDEXING() 1
#else // ^^^ has pack indexing ^^^ / vvv no pack indexing vvv
#  define _CCCL_HAS_PACK_INDEXING() 0
#endif // no pack indexing

#if _CCCL_STD_VER <= 2017 || __cpp_consteval < 201811L
#  define _CCCL_NO_CONSTEVAL
#  define _CCCL_CONSTEVAL constexpr
#else
#  define _CCCL_CONSTEVAL consteval
#endif

///////////////////////////////////////////////////////////////////////////////
// Conditionally use certain language features depending on availability
///////////////////////////////////////////////////////////////////////////////

// We need to treat host and device separately
#if _CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC)
#  define _CCCL_GLOBAL_CONSTANT _CCCL_DEVICE constexpr
#else // ^^^ _CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC) ^^^ /
      // vvv !_CCCL_DEVICE_COMPILATION() || _CCCL_CUDA_COMPILER(NVHPC) vvv
#  define _CCCL_GLOBAL_CONSTANT inline constexpr
#endif // !_CCCL_DEVICE_COMPILATION() || _CCCL_CUDA_COMPILER(NVHPC)

#if _CCCL_STD_VER >= 2020 && __cpp_constinit >= 201907L
#  define _CCCL_CONSTINIT constinit
#else // ^^^ has constinit ^^^ / vvv no constinit vvv
#  define _CCCL_CONSTINIT _CCCL_REQUIRE_CONSTANT_INITIALIZATION
#endif // ^^^ no constinit ^^^

// nvcc and nvrtc don't implement multiarg operator[] even in C++23 mode
#if __cpp_multidimensional_subscript >= 202211L && !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_CUDA_COMPILER(NVRTC)
#  define _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() 1
#else // ^^^ has multiarg operator[] ^^^ / vvv no multiarg operator[] vvv
#  define _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() 0
#endif // ^^^ no mutiarg operator[] ^^^

// if consteval requires C++23, but most compilers support it even in C++20 mode while emitting some warnings. Those are
// silenced in prologue/epilogue. nvcc is happy about using it in C++20 since 13.0, but only when compiling host code.
// nvc++ requires libstdc++ at least 12 to support if consteval.
#if _CCCL_STD_VER == 2020                                  \
  && (_CCCL_COMPILER(GCC, >=, 12) || _CCCL_COMPILER(CLANG) \
      || (_CCCL_COMPILER(NVHPC) && _CCCL_HOST_STD_LIB(LIBSTDCXX, >=, 12)))
#  define _CCCL_HAS_IF_CONSTEVAL_IN_CXX20() 1
#else
#  define _CCCL_HAS_IF_CONSTEVAL_IN_CXX20() 0
#endif

// nvcc before 13 doesn't support if consteval at all. Since 13, it accepts if consteval in host code (clang doesn't
// work) and since 13.1 it works in device code, too.
#if _CCCL_CUDA_COMPILER(NVCC, <, 13) || (_CCCL_CUDA_COMPILER(NVCC, <, 13, 1) && _CCCL_DEVICE_COMPILATION()) \
  || (_CCCL_CUDA_COMPILER(NVCC) && _CCCL_COMPILER(CLANG))
#  undef _CCCL_HAS_IF_CONSTEVAL_IN_CXX20
#  define _CCCL_HAS_IF_CONSTEVAL_IN_CXX20() 0
#endif // ^^^ disable if consteval in c++20 for nvcc ^^^

#if __cpp_if_consteval >= 202106L || _CCCL_HAS_IF_CONSTEVAL_IN_CXX20()
#  define _CCCL_IF_CONSTEVAL             if consteval
#  define _CCCL_IF_CONSTEVAL_DEFAULT     _CCCL_IF_CONSTEVAL
#  define _CCCL_IF_NOT_CONSTEVAL         if !consteval
#  define _CCCL_IF_NOT_CONSTEVAL_DEFAULT _CCCL_IF_NOT_CONSTEVAL
#elif defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
#  if _CCCL_HOST_COMPILATION() && _CCCL_COMPILER(GCC)
#    define _CCCL_BEGIN_IF_CONSTEVAL_SUPPRESS() _CCCL_DIAG_PUSH _CCCL_DIAG_SUPPRESS_GCC("-Wtautological-compare")
#    define _CCCL_END_IF_CONSTEVAL_SUPPRESS()   _CCCL_DIAG_POP
#  else // ^^^ _CCCL_HOST_COMPILATION() && _CCCL_COMPILER(GCC) ^^^ /
        // vvv !_CCCL_HOST_COMPILATION() || ! _CCCL_COMPILER(GCC) vvv
#    define _CCCL_BEGIN_IF_CONSTEVAL_SUPPRESS()
#    define _CCCL_END_IF_CONSTEVAL_SUPPRESS()
#  endif // ^^^ !_CCCL_HOST_COMPILATION() || ! _CCCL_COMPILER(GCC) ^^^

#  define _CCCL_IF_CONSTEVAL \
    _CCCL_BEGIN_IF_CONSTEVAL_SUPPRESS() if (_CCCL_BUILTIN_IS_CONSTANT_EVALUATED()) _CCCL_END_IF_CONSTEVAL_SUPPRESS()
#  define _CCCL_IF_CONSTEVAL_DEFAULT _CCCL_IF_CONSTEVAL
#  define _CCCL_IF_NOT_CONSTEVAL \
    _CCCL_BEGIN_IF_CONSTEVAL_SUPPRESS() if (!_CCCL_BUILTIN_IS_CONSTANT_EVALUATED()) _CCCL_END_IF_CONSTEVAL_SUPPRESS()
#  define _CCCL_IF_NOT_CONSTEVAL_DEFAULT _CCCL_IF_NOT_CONSTEVAL
#else // ^^^ has is constant evaluated ^^^ / vvv no is constant evaluated vvv
#  define _CCCL_IF_CONSTEVAL             if constexpr (false)
#  define _CCCL_IF_CONSTEVAL_DEFAULT     if constexpr (true)
#  define _CCCL_IF_NOT_CONSTEVAL         if constexpr (true)
#  define _CCCL_IF_NOT_CONSTEVAL_DEFAULT if constexpr (false)
#endif // ^^^ no is constant evaluated ^^^

#if _CCCL_STD_VER >= 2020 && __cpp_char8_t >= 201811L
#  define _CCCL_HAS_CHAR8_T() 1
#else // ^^^ has char8_t ^^^ / vvv no char8_t vvv
#  define _CCCL_HAS_CHAR8_T() 0
#endif // ^^^ no char8_t ^^^

// We currently do not support any of the STL wchar facilities
#define _CCCL_HAS_WCHAR_T() 0

// Fixme: replace the condition with (!_CCCL_DEVICE_COMPILATION())
// FIXME: Enable this for clang-cuda in a followup
#if !_CCCL_CUDA_COMPILATION()
#  define _CCCL_HAS_LONG_DOUBLE() 1
#else // ^^^ !_CCCL_CUDA_COMPILATION() ^^^ / vvv _CCCL_CUDA_COMPILATION() vvv
#  define _CCCL_HAS_LONG_DOUBLE() 0
#endif // ^^^ _CCCL_CUDA_COMPILATION() ^^^

#endif // __CCCL_DIALECT_H
