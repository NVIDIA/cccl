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

#if defined(_CCCL_COMPILER_MSVC)
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
#else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv
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
#endif // !_CCCL_COMPILER_MSVC

#if _CCCL_STD_VER >= 2014
#  define _CCCL_CONSTEXPR_CXX14 constexpr
#else // ^^^ C++14 ^^^ / vvv C++11 vvv
#  define _CCCL_CONSTEXPR_CXX14
#endif // _CCCL_STD_VER <= 2011

#if _CCCL_STD_VER >= 2017
#  define _CCCL_CONSTEXPR_CXX17 constexpr
#else // ^^^ C++17 ^^^ / vvv C++14 vvv
#  define _CCCL_CONSTEXPR_CXX17
#endif // _CCCL_STD_VER <= 2014

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

#if _CCCL_STD_VER >= 2017 && defined(__cpp_if_constexpr)
#  define _CCCL_IF_CONSTEXPR      if constexpr
#  define _CCCL_ELSE_IF_CONSTEXPR else if constexpr
#else // ^^^ C++17 ^^^ / vvv C++14 vvv
#  define _CCCL_IF_CONSTEXPR      if
#  define _CCCL_ELSE_IF_CONSTEXPR else if
#endif // _CCCL_STD_VER <= 2014

#if _CCCL_STD_VER >= 2017
#  define _CCCL_TRAIT(__TRAIT, ...) __TRAIT##_v<__VA_ARGS__>
#else // ^^^ C++17 ^^^ / vvv C++14 vvv
#  define _CCCL_TRAIT(__TRAIT, ...) __TRAIT<__VA_ARGS__>::value
#endif // _CCCL_STD_VER <= 2014

// In nvcc prior to 11.3 global variables could not be marked constexpr
#if defined(_CCCL_CUDACC_BELOW_11_3)
#  define _CCCL_CONSTEXPR_GLOBAL const
#else // ^^^ _CCCL_CUDACC_BELOW_11_3 ^^^ / vvv !_CCCL_CUDACC_BELOW_11_3 vvv
#  define _CCCL_CONSTEXPR_GLOBAL constexpr
#endif // !_CCCL_CUDACC_BELOW_11_3

// Inline variables are only available from C++17 onwards
#if _CCCL_STD_VER >= 2017 && defined(__cpp_inline_variables) && (__cpp_inline_variables >= 201606L)
#  define _CCCL_INLINE_VAR inline
#else // ^^^ C++14 ^^^ / vvv C++17 vvv
#  define _CCCL_INLINE_VAR
#endif // _CCCL_STD_VER <= 2014

// We need to treat host and device separately
#if defined(__CUDA_ARCH__)
#  define _CCCL_GLOBAL_CONSTANT _CCCL_DEVICE _CCCL_CONSTEXPR_GLOBAL
#else // ^^^ __CUDA_ARCH__ ^^^ / vvv !__CUDA_ARCH__ vvv
#  define _CCCL_GLOBAL_CONSTANT _CCCL_INLINE_VAR constexpr
#endif // __CUDA_ARCH__

#endif // __CCCL_DIALECT_H
