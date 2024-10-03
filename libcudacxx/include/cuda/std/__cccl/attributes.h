//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_ATTRIBUTES_H
#define __CCCL_ATTRIBUTES_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/dialect.h>

#ifdef __has_attribute
#  define _CCCL_HAS_ATTRIBUTE(__x) __has_attribute(__x)
#else // ^^^ __has_attribute ^^^ / vvv !__has_attribute vvv
#  define _CCCL_HAS_ATTRIBUTE(__x) 0
#endif // !__has_attribute

#ifdef __has_cpp_attribute
#  define _CCCL_HAS_CPP_ATTRIBUTE(__x) __has_cpp_attribute(__x)
#else // ^^^ __has_cpp_attribute ^^^ / vvv !__has_cpp_attribute vvv
#  define _CCCL_HAS_CPP_ATTRIBUTE(__x) 0
#endif // !__has_cpp_attribute

// Use a function like macro to imply that it must be followed by a semicolon
#if _CCCL_STD_VER >= 2017 && _CCCL_HAS_CPP_ATTRIBUTE(fallthrough)
#  define _CCCL_FALLTHROUGH() [[fallthrough]]
#elif defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_FALLTHROUGH() ((void) 0)
#elif _CCCL_HAS_CPP_ATTRIBUTE(clang::fallthrough)
#  define _CCCL_FALLTHROUGH() [[clang::fallthrough]]
#elif defined(_CCCL_COMPILER_NVHPC)
#  define _CCCL_FALLTHROUGH()
#elif _CCCL_HAS_ATTRIBUTE(fallthough) || _GNUC_VER >= 700
#  define _CCCL_FALLTHROUGH() __attribute__((__fallthrough__))
#else
#  define _CCCL_FALLTHROUGH() ((void) 0)
#endif

#if _CCCL_HAS_CPP_ATTRIBUTE(msvc::no_unique_address)
// MSVC implements [[no_unique_address]] as a silent no-op currently.
// (If/when MSVC breaks its C++ ABI, it will be changed to work as intended.)
// However, MSVC implements [[msvc::no_unique_address]] which does what
// [[no_unique_address]] is supposed to do, in general.
#  define _CCCL_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#elif defined(_CCCL_CUDACC_BELOW_11_3) || (_CCCL_HAS_CPP_ATTRIBUTE(no_unique_address) < 201803L)
#  define _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
#  define _CCCL_NO_UNIQUE_ADDRESS
#elif _CCCL_HAS_CPP_ATTRIBUTE(no_unique_address)
#  define _CCCL_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#  define _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
#  define _CCCL_NO_UNIQUE_ADDRESS
#endif

#if _CCCL_HAS_CPP_ATTRIBUTE(nodiscard) || (defined(_CCCL_COMPILER_MSVC) && _CCCL_STD_VER >= 2017)
#  define _CCCL_NODISCARD [[nodiscard]]
#else // ^^^ has nodiscard ^^^ / vvv no nodiscard vvv
#  define _CCCL_NODISCARD
#endif // no nodiscard

// NVCC below 11.3 does not support nodiscard on friend operators
// It always fails with clang
#if defined(_CCCL_CUDACC_BELOW_11_3) || defined(_CCCL_COMPILER_CLANG)
#  define _CCCL_NODISCARD_FRIEND friend
#else // ^^^ _CCCL_CUDACC_BELOW_11_3 ^^^ / vvv !_CCCL_CUDACC_BELOW_11_3 vvv
#  define _CCCL_NODISCARD_FRIEND _CCCL_NODISCARD friend
#endif // !_CCCL_CUDACC_BELOW_11_3 && !_CCCL_COMPILER_CLANG

// NVCC below 11.3 does not support attributes on alias declarations
#ifdef _CCCL_CUDACC_BELOW_11_3
#  define _CCCL_ALIAS_ATTRIBUTE(...)
#else // ^^^ _CCCL_CUDACC_BELOW_11_3 ^^^ / vvv !_CCCL_CUDACC_BELOW_11_3 vvv
#  define _CCCL_ALIAS_ATTRIBUTE(...) __VA_ARGS__
#endif // !_CCCL_CUDACC_BELOW_11_3

#if defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_NORETURN __declspec(noreturn)
#elif _CCCL_HAS_CPP_ATTRIBUTE(noreturn)
#  define _CCCL_NORETURN [[noreturn]]
#else
#  define _CCCL_NORETURN __attribute__((noreturn))
#endif

#endif // __CCCL_ATTRIBUTES_H
