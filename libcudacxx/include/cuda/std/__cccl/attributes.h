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

#ifdef __has_declspec_attribute
#  define _CCCL_HAS_DECLSPEC_ATTRIBUTE(__x) __has_declspec_attribute(__x)
#else // ^^^ __has_declspec_attribute ^^^ / vvv !__has_declspec_attribute vvv
#  define _CCCL_HAS_DECLSPEC_ATTRIBUTE(__x) 0
#endif // !__has_declspec_attribute

// MSVC needs extra help with empty base classes
#if _CCCL_COMPILER(MSVC) || _CCCL_HAS_DECLSPEC_ATTRIBUTE(empty_bases)
#  define _CCCL_DECLSPEC_EMPTY_BASES __declspec(empty_bases)
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#  define _CCCL_DECLSPEC_EMPTY_BASES
#endif // !_CCCL_COMPILER(MSVC)

#if _CCCL_HAS_ATTRIBUTE(__nodebug__)
#  define _CCCL_NODEBUG __attribute__((__nodebug__))
#else // ^^^ _CCCL_HAS_ATTRIBUTE(__nodebug__) ^^^ / vvv !_CCCL_HAS_ATTRIBUTE(__nodebug__) vvv
#  define _CCCL_NODEBUG
#endif // !_CCCL_HAS_ATTRIBUTE(__nodebug__)

// The nodebug attribute flattens aliases down to the actual type rather typename meow<T>::type
#if _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_NODEBUG_ALIAS _CCCL_NODEBUG
#else // ^^^ _CCCL_CUDA_COMPILER(CLANG) ^^^ / vvv !_CCCL_CUDA_COMPILER(CLANG) vvv
#  define _CCCL_NODEBUG_ALIAS
#endif // !_CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_COMPILER(MSVC) || _CCCL_HAS_CPP_ATTRIBUTE(no_unique_address) < 201803L
// MSVC implementation has lead to multiple issues with silent runtime corruption when passing data into kernels
#  define _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
#  define _CCCL_NO_UNIQUE_ADDRESS
#elif _CCCL_HAS_CPP_ATTRIBUTE(no_unique_address)
#  define _CCCL_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#  define _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
#  define _CCCL_NO_UNIQUE_ADDRESS
#endif

// Passing objects with nested [[no_unique_address]] to kernels leads to data corruption
// This happens up to clang18
#if !defined(_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS) && _CCCL_COMPILER(CLANG)
#  define _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
#endif // !_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS && _CCCL_COMPILER(CLANG)

// It always fails with clang
#if _CCCL_COMPILER(CLANG)
#  define _CCCL_NODISCARD_FRIEND friend
#else
#  define _CCCL_NODISCARD_FRIEND [[nodiscard]] friend
#endif

#if _CCCL_COMPILER(MSVC) // vvv _CCCL_COMPILER(MSVC) vvv
#  define _CCCL_RESTRICT __restrict
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#  define _CCCL_RESTRICT __restrict__
#endif // ^^^ !_CCCL_COMPILER(MSVC) ^^^

#if _CCCL_HAS_CPP_ATTRIBUTE(assume)
#  define _CCCL_ASSUME(...) [[assume(__VA_ARGS__)]]
#elif _CCCL_CUDA_COMPILER(NVCC) && _CCCL_COMPILER(NVHPC)
#  define _CCCL_ASSUME(...) \
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__builtin_assume(__VA_ARGS__);), (_CCCL_BUILTIN_ASSUME(__VA_ARGS__);))
#else
#  define _CCCL_ASSUME(...) _CCCL_BUILTIN_ASSUME(__VA_ARGS__)
#endif

#if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 5)
#  define _CCCL_PURE __nv_pure__
#elif _CCCL_HAS_CPP_ATTRIBUTE(gnu::pure)
#  define _CCCL_PURE [[gnu::pure]]
#elif _CCCL_COMPILER(MSVC)
#  define _CCCL_PURE __declspec(noalias)
#else
#  define _CCCL_PURE
#endif

#if !_CCCL_COMPILER(MSVC) // _CCCL_HAS_CPP_ATTRIBUTE(const) doesn't work with MSVC
#  if _CCCL_HAS_CPP_ATTRIBUTE(gnu::const)
#    define _CCCL_CONST [[gnu::const]]
#  else
#    define _CCCL_CONST _CCCL_PURE
#  endif
#else
#  define _CCCL_CONST _CCCL_PURE
#endif

#if _CCCL_HAS_CPP_ATTRIBUTE(clang::no_specializations)
#  define _CCCL_NO_SPECIALIZATIONS_BECAUSE(_MSG)   [[clang::no_specializations(_MSG)]]
#  define _CCCL_HAS_ATTRIBUTE_NO_SPECIALIZATIONS() 1
#elif _CCCL_HAS_CPP_ATTRIBUTE(msvc::no_specializations)
#  define _CCCL_NO_SPECIALIZATIONS_BECAUSE(_MSG)   [[msvc::no_specializations(_MSG)]]
#  define _CCCL_HAS_ATTRIBUTE_NO_SPECIALIZATIONS() 1
#else // ^^^ has attribute no_specializations ^^^ / vvv hasn't attribute no_specializations vvv
#  define _CCCL_NO_SPECIALIZATIONS_BECAUSE(_MSG)
#  define _CCCL_HAS_ATTRIBUTE_NO_SPECIALIZATIONS() 0
#endif // ^^^ hasn't attribute no_specializations ^^^

#define _CCCL_NO_SPECIALIZATIONS \
  _CCCL_NO_SPECIALIZATIONS_BECAUSE("Users are not allowed to specialize this cccl entity")

#endif // __CCCL_ATTRIBUTES_H
