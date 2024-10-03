//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_BUILTIN_H
#define __CCCL_BUILTIN_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! This file consolidates all compiler builtin detection for CCCL.
//!
//! To work around older compilers not supporting `__has_builtin` we use `_CCCL_CHECK_BUILTIN` that detects more
//! cases
//!
//! * We work around old clang versions (before clang-10) not supporting __has_builtin via _CCCL_CHECK_BUILTIN
//! * We work around old intel versions (before 2021.3)   not supporting __has_builtin via _CCCL_CHECK_BUILTIN
//! * We work around old nvhpc versions (before 2022.11)  not supporting __has_builtin via _CCCL_CHECK_BUILTIN
//! * MSVC needs manual handling, has no real way of checking builtins so all is manual
//! * GCC  needs manual handling, before gcc-10 as that finally supports __has_builtin
//!
//! In case compiler support for a builtin is advertised but leads to regressions we explicitly #undef the macro
//!
//! Finally, because `_CCCL_CHECK_BUILTIN` may lead to false positives, we move detection of new builtins over towards
//! just using _CCCL_HAS_BUILTIN

#ifdef __has_builtin
#  define _CCCL_HAS_BUILTIN(__x) __has_builtin(__x)
#else // ^^^ __has_builtin ^^^ / vvv !__has_builtin vvv
#  define _CCCL_HAS_BUILTIN(__x) 0
#endif // !__has_builtin

#ifdef __has_feature
#  define _CCCL_HAS_FEATURE(__x) __has_feature(__x)
#else // ^^^ __has_feature ^^^ / vvv !__has_feature vvv
#  define _CCCL_HAS_FEATURE(__x) 0
#endif // !__has_feature

// '__is_identifier' returns '0' if '__x' is a reserved identifier provided by the compiler and '1' otherwise.
#ifdef __is_identifier
#  define _CCCL_IS_IDENTIFIER(__x) __is_identifier(__x)
#else // ^^^ __is_identifier ^^^ / vvv !__is_identifier vvv
#  define _CCCL_IS_IDENTIFIER(__x) 1
#endif // !__is_identifier

#define _CCCL_HAS_KEYWORD(__x) !(_CCCL_IS_IDENTIFIER(__x))

// https://bugs.llvm.org/show_bug.cgi?id=44517
#define _CCCL_CHECK_BUILTIN(__x) (_CCCL_HAS_BUILTIN(__##__x) || _CCCL_HAS_KEYWORD(__##__x) || _CCCL_HAS_FEATURE(__x))

// NVCC has issues with function pointers
#if _CCCL_HAS_BUILTIN(__add_lvalue_reference) && defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_BUILTIN_ADD_LVALUE_REFERENCE(...) __add_lvalue_reference(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__add_lvalue_reference)

// NVCC has issues with function pointers
#if _CCCL_HAS_BUILTIN(__add_pointer) && defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_BUILTIN_ADD_POINTER(...) __add_pointer(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__add_pointer)

// NVCC has issues with function pointers
#if _CCCL_HAS_BUILTIN(__add_rvalue_reference) && defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_BUILTIN_ADD_RVALUE_REFERENCE(...) __add_rvalue_reference(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__add_rvalue_reference)

// TODO: Enable using the builtin __array_rank when https://llvm.org/PR57133 is resolved
#if 0 // _CCCL_CHECK_BUILTIN(array_rank)
#  define _CCCL_BUILTIN_ARRAY_RANK(...) __array_rank(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(array_rank)

#if _CCCL_HAS_BUILTIN(__array_extent)
#  define _CCCL_BUILTIN_ARRAY_EXTENT(...) __array_extent(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(array_extent)

// nvhpc has a bug where it supports __builtin_addressof but does not mark it via _CCCL_CHECK_BUILTIN
#if _CCCL_CHECK_BUILTIN(builtin_addressof) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 70000) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVHPC)
#  define _CCCL_BUILTIN_ADDRESSOF(...) __builtin_addressof(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_addressof)

#if _CCCL_CHECK_BUILTIN(builtin_assume)
#  define _CCCL_BUILTIN_ASSUME(...) __builtin_assume(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_assume)

// NVCC prior to 11.2 cannot handle __builtin_assume
#if defined(_CCCL_CUDACC_BELOW_11_2)
#  undef _CCCL_BUILTIN_ASSUME
#endif // _CCCL_CUDACC_BELOW_11_2

// NVCC prior to 11.2 does not understand __builtin_assume
#if defined(_CCCL_CUDACC_BELOW_11_2)
#  undef _CCCL_BUILTIN_ASSUME
#endif // _CCCL_CUDACC_BELOW_11_2

// MSVC supports __builtin_bit_cast from 19.25 on
#if _CCCL_CHECK_BUILTIN(builtin_bit_cast) || (defined(_CCCL_COMPILER_MSVC) && _CCCL_MSVC_VERSION > 1925)
#  define _CCCL_BUILTIN_BIT_CAST(...) __builtin_bit_cast(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_bit_cast)

// clang prior to clang-10 supports __builtin_bit_cast but it is not a constant expression
// NVCC prior to 11.7 cannot parse __builtin_bit_cast
#if (defined(_CCCL_COMPILER_CLANG) && _CCCL_CLANG_VERSION < 100000) || defined(_CCCL_CUDACC_BELOW_11_7)
#  undef _CCCL_BUILTIN_BIT_CAST
#endif // clang < 10 || nvcc < 11.7

#if _CCCL_CHECK_BUILTIN(builtin_contant_p) || defined(_CCCL_COMPILER_GCC)
#  define _CCCL_BUILTIN_CONSTANT_P(...) __builtin_constant_p(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_contant_p)

#if _CCCL_CHECK_BUILTIN(builtin_expect) || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_GCC)
#  define _CCCL_BUILTIN_EXPECT(...) __builtin_expect(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_expect)

#if _CCCL_CHECK_BUILTIN(builtin_is_constant_evaluated) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 90000) \
  || (defined(_CCCL_COMPILER_MSVC) && _CCCL_MSVC_VERSION > 1924 && !defined(_CCCL_CUDACC_BELOW_11_3))
#  define _CCCL_BUILTIN_IS_CONSTANT_EVALUATED(...) __builtin_is_constant_evaluated(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_is_constant_evaluated)

// NVCC and NVRTC in C++11 mode freaks out about `__builtin_is_constant_evaluated`.
#if _CCCL_STD_VER < 2014 \
  && (defined(_CCCL_CUDA_COMPILER_NVCC) || defined(_CCCL_COMPILER_NVRTC) || defined(_CCCL_COMPILER_NVHPC))
#  undef _CCCL_BUILTIN_IS_CONSTANT_EVALUATED
#endif // _CCCL_STD_VER < 2014 && _CCCL_CUDA_COMPILER_NVCC

#if (_CCCL_CHECK_BUILTIN(builtin_launder) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 70000))
#  define _CCCL_BUILTIN_LAUNDER(...) __builtin_launder(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_launder) && gcc >= 7

// NVCC prior to 11.3 and clang prior to clang-10 accept __builtin_launder but break with a compiler error about an
// invalid return type
#if (defined(_CCCL_COMPILER_CLANG) && _CCCL_CLANG_VERSION < 100000) || defined(_CCCL_CUDACC_BELOW_11_3)
#  undef _CCCL_BUILTIN_LAUNDER
#endif // clang < 10 || nvcc < 11.3

#if _CCCL_CHECK_BUILTIN(__builtin_operator_new) && _CCCL_CHECK_BUILTIN(__builtin_operator_delete) \
  && defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_BUILTIN_OPERATOR_DELETE(...) __builtin_operator_delete(__VA_ARGS__)
#  define _CCCL_BUILTIN_OPERATOR_NEW(...)    __builtin_operator_new(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(__builtin_operator_new) && _CCCL_CHECK_BUILTIN(__builtin_operator_delete)

#if _CCCL_HAS_BUILTIN(__decay) && defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_BUILTIN_DECAY(...) __decay(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__decay) && clang-cuda

#if _CCCL_CHECK_BUILTIN(has_nothrow_assign) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_HAS_NOTHROW_ASSIGN(...) __has_nothrow_assign(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_nothrow_assign) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(has_nothrow_constructor) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_HAS_NOTHROW_CONSTRUCTOR(...) __has_nothrow_constructor(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_nothrow_constructor) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(has_nothrow_copy) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_HAS_NOTHROW_COPY(...) __has_nothrow_copy(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_nothrow_copy) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(has_trivial_constructor) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_HAS_TRIVIAL_CONSTRUCTOR(...) __has_trivial_constructor(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_trivial_constructor) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(has_trivial_destructor) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_HAS_TRIVIAL_DESTRUCTOR(...) __has_trivial_destructor(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_trivial_destructor) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(has_unique_object_representations) \
  || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 70000)
#  define _CCCL_BUILTIN_HAS_UNIQUE_OBJECT_REPRESENTATIONS(...) __has_unique_object_representations(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_unique_object_representations) && gcc >= 7.0

#if _CCCL_CHECK_BUILTIN(has_virtual_destructor) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_HAS_VIRTUAL_DESTRUCTOR(...) __has_virtual_destructor(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_virtual_destructor) && gcc >= 4.3

#if _CCCL_HAS_BUILTIN(__integer_pack)
#  define _CCCL_BUILTIN_INTEGER_PACK(...) __integer_pack(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__integer_pack)

#if _CCCL_CHECK_BUILTIN(is_aggregate) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 70000) \
  || (defined(_CCCL_COMPILER_MSVC) && _CCCL_MSVC_VERSION > 1914) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_AGGREGATE(...) __is_aggregate(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_aggregate) && gcc >= 7.0

#if _CCCL_CHECK_BUILTIN(is_array)
#  define _CCCL_BUILTIN_IS_ARRAY(...) __is_array(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_array)

// clang prior to clang-19 gives wrong results for __is_array of _Tp[0]
#if defined(_CCCL_COMPILER_CLANG) && _CCCL_CLANG_VERSION < 190000
#  undef _CCCL_BUILTIN_IS_ARRAY
#endif // clang < 19

#if _CCCL_CHECK_BUILTIN(is_assignable) || defined(_CCCL_COMPILER_MSVC) \
  || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 90000)
#  define _CCCL_BUILTIN_IS_ASSIGNABLE(...) __is_assignable(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_assignable) && gcc >= 9.0

#if _CCCL_CHECK_BUILTIN(is_base_of) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_BASE_OF(...) __is_base_of(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_base_of) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(is_class) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_CLASS(...) __is_class(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_class) && gcc >= 4.3

#if _CCCL_HAS_BUILTIN(__is_compound)
#  define _CCCL_BUILTIN_IS_COMPOUND(...) __is_compound(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_compound)

#if _CCCL_HAS_BUILTIN(__is_const)
#  define _CCCL_BUILTIN_IS_CONST(...) __is_const(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_const)

#if _CCCL_CHECK_BUILTIN(is_constructible) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 80000) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_CONSTRUCTIBLE(...) __is_constructible(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_constructible) && gcc >= 8.0

#if _CCCL_CHECK_BUILTIN(is_convertible_to) || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_CONVERTIBLE_TO(...) __is_convertible_to(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_convertible_to)

#if _CCCL_CHECK_BUILTIN(is_destructible) || defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_BUILTIN_IS_DESTRUCTIBLE(...) __is_destructible(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_destructible)

#if _CCCL_CHECK_BUILTIN(is_empty) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_EMPTY(...) __is_empty(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_empty) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(is_enum) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_ENUM(...) __is_enum(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_enum) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(is_final) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40700) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_FINAL(...) __is_final(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_final) && gcc >= 4.7

#if _CCCL_CHECK_BUILTIN(is_function)
#  define _CCCL_BUILTIN_IS_FUNCTION(...) __is_function(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_function)

// All current versions of NVCC give wrong results with __is_function
#if defined(_CCCL_CUDA_COMPILER_NVCC)
#  undef _CCCL_BUILTIN_IS_FUNCTION
#endif // _CCCL_CUDA_COMPILER_NVCC

#if _CCCL_CHECK_BUILTIN(is_fundamental)
#  define _CCCL_BUILTIN_IS_FUNDAMENTAL(...) __is_fundamental(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_fundamental)

// clang prior to clang-10 gives wrong results for __is_fundamental
#if defined(_CCCL_COMPILER_CLANG) && _CCCL_CLANG_VERSION < 100000
#  undef _CCCL_BUILTIN_IS_FUNDAMENTAL
#endif // clang < 10

#if _CCCL_HAS_BUILTIN(__is_integral)
#  define _CCCL_BUILTIN_IS_INTEGRAL(...) __is_integral(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_integral)

#if _CCCL_CHECK_BUILTIN(is_literal_type) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40600) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_LITERAL(...) __is_literal_type(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_literal_type) && gcc >= 4.6

#if _CCCL_CHECK_BUILTIN(is_lvalue_reference)
#  define _CCCL_BUILTIN_IS_LVALUE_REFERENCE(...) __is_lvalue_reference(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_lvalue_reference)

// NVCC prior to 11.3 cannot parse __is_lvalue_reference
#if defined(_CCCL_CUDACC_BELOW_11_3)
#  undef _CCCL_BUILTIN_IS_LVALUE_REFERENCE
#endif // _CCCL_CUDACC_BELOW_11_3

#if _CCCL_HAS_BUILTIN(__is_member_function_pointer)
#  define _CCCL_BUILTIN_IS_MEMBER_FUNCTION_POINTER(...) __is_member_function_pointer(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_member_function_pointer)

#if _CCCL_HAS_BUILTIN(__is_member_object_pointer)
#  define _CCCL_BUILTIN_IS_MEMBER_OBJECT_POINTER(...) __is_member_object_pointer(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_member_object_pointer)

#if _CCCL_HAS_BUILTIN(__is_member_pointer)
#  define _CCCL_BUILTIN_IS_MEMBER_POINTER(...) __is_member_pointer(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_member_pointer)

#if _CCCL_CHECK_BUILTIN(is_nothrow_assignable) || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_NOTHROW_ASSIGNABLE(...) __is_nothrow_assignable(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_nothrow_assignable)

#if _CCCL_CHECK_BUILTIN(is_nothrow_constructible) || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_NOTHROW_CONSTRUCTIBLE(...) __is_nothrow_constructible(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_nothrow_constructible)

#if _CCCL_CHECK_BUILTIN(is_nothrow_destructible) || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE(...) __is_nothrow_destructible(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_nothrow_destructible)

#if _CCCL_CHECK_BUILTIN(is_object)
#  define _CCCL_BUILTIN_IS_OBJECT(...) __is_object(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_object)

// NVCC prior to 11.3 cannot parse __is_object
#if defined(_CCCL_CUDACC_BELOW_11_3)
#  undef _CCCL_BUILTIN_IS_OBJECT
#endif // _CCCL_CUDACC_BELOW_11_3

#if _CCCL_CHECK_BUILTIN(is_pod) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_POD(...) __is_pod(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_pod) && gcc >= 4.3

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__is_pointer)
#  define _CCCL_BUILTIN_IS_POINTER(...) __is_pointer(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_pointer)

#if _CCCL_CHECK_BUILTIN(is_polymorphic) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_POLYMORPHIC(...) __is_polymorphic(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_polymorphic) && gcc >= 4.3

#if _CCCL_HAS_BUILTIN(__is_reference)
#  define _CCCL_BUILTIN_IS_REFERENCE(...) __is_reference(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_reference)

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__is_referenceable)
#  define _CCCL_BUILTIN_IS_REFERENCEABLE(...) __is_referenceable(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_referenceable)

#if _CCCL_HAS_BUILTIN(__is_rvalue_reference)
#  define _CCCL_BUILTIN_IS_RVALUE_REFERENCE(...) __is_rvalue_reference(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_rvalue_reference)

#if _CCCL_CHECK_BUILTIN(is_same) && !defined(_CCCL_CUDA_COMPILER_NVCC)
#  define _CCCL_BUILTIN_IS_SAME(...) __is_same(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_same)

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__is_scalar)
#  define _CCCL_BUILTIN_IS_SCALAR(...) __is_scalar(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_scalar)

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__is_signed)
#  define _CCCL_BUILTIN_IS_SIGNED(...) __is_signed(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_signed)

#if _CCCL_CHECK_BUILTIN(is_standard_layout) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40700) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_STANDARD_LAYOUT(...) __is_standard_layout(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_standard_layout) && gcc >= 4.7

#if _CCCL_CHECK_BUILTIN(is_trivial) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40500) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_TRIVIAL(...) __is_trivial(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_trivial) && gcc >= 4.5

#if _CCCL_CHECK_BUILTIN(is_trivially_assignable) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 50100) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_TRIVIALLY_ASSIGNABLE(...) __is_trivially_assignable(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_trivially_assignable) && gcc >= 5.1

#if _CCCL_CHECK_BUILTIN(is_trivially_constructible) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 50100) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_TRIVIALLY_CONSTRUCTIBLE(...) __is_trivially_constructible(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_trivially_constructible) && gcc >= 5.1

#if _CCCL_CHECK_BUILTIN(is_trivially_copyable) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 50100) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_TRIVIALLY_COPYABLE(...) __is_trivially_copyable(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_trivially_copyable) && gcc >= 5.1

#if _CCCL_CHECK_BUILTIN(is_trivially_destructible) || defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_BUILTIN_IS_TRIVIALLY_DESTRUCTIBLE(...) __is_trivially_destructible(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_trivially_destructible)

#if _CCCL_CHECK_BUILTIN(is_union) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_UNION(...) __is_union(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_union) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(is_unsigned)
#  define _CCCL_BUILTIN_IS_UNSIGNED(...) __is_unsigned(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_unsigned)

// NVCC prior to 11.3 cannot parse __is_unsigned
#if defined(_CCCL_CUDACC_BELOW_11_3)
#  undef _CCCL_BUILTIN_IS_UNSIGNED
#endif // _CCCL_CUDACC_BELOW_11_3

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__is_void)
#  define _CCCL_BUILTIN_IS_VOID(...) __is_void(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_void)

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__is_volatile)
#  define _CCCL_BUILTIN_IS_VOLATILE(...) __is_volatile(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_volatile)

#if _CCCL_CHECK_BUILTIN(isfinite)
#  define _CCCL_BUILTIN_ISFINITE(...) __builtin_isfinite(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(isfinite)

#if _CCCL_CHECK_BUILTIN(isinf)
#  define _CCCL_BUILTIN_ISINF(...) __builtin_isinf(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(isinf)

#if _CCCL_CHECK_BUILTIN(isnan)
#  define _CCCL_BUILTIN_ISNAN(...) __builtin_isnan(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(isnan)

#if _CCCL_CHECK_BUILTIN(make_integer_seq) || (defined(_CCCL_COMPILER_MSVC) && _CCCL_MSVC_VERSION >= 1923)
#  define _CCCL_BUILTIN_MAKE_INTEGER_SEQ(...) __make_integer_seq<__VA_ARGS__>
#endif // _CCCL_CHECK_BUILTIN(make_integer_seq)

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__make_signed)
#  define _CCCL_BUILTIN_MAKE_SIGNED(...) __make_signed(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__make_signed)

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__make_unsigned)
#  define _CCCL_BUILTIN_MAKE_UNSIGNED(...) __make_unsigned(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__make_unsigned)

#if _CCCL_HAS_BUILTIN(__remove_all_extents) && defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_BUILTIN_REMOVE_ALL_EXTENTS(...) __remove_all_extents(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_all_extents)

#if _CCCL_HAS_BUILTIN(__remove_const) && defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_BUILTIN_REMOVE_CONST(...) __remove_const(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_const)

#if _CCCL_HAS_BUILTIN(__remove_cv) && defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_BUILTIN_REMOVE_CV(...) __remove_cv(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_cv)

#if _CCCL_HAS_BUILTIN(__remove_cvref) && defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_BUILTIN_REMOVE_CVREF(...) __remove_cvref(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_cvref)

#if _CCCL_HAS_BUILTIN(__remove_extent) && defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_BUILTIN_REMOVE_EXTENT(...) __remove_extent(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_extent)

#if _CCCL_HAS_BUILTIN(__remove_pointer) && defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_BUILTIN_REMOVE_POINTER(...) __remove_pointer(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_pointer)

#if _CCCL_HAS_BUILTIN(__remove_reference)
#  define _CCCL_BUILTIN_REMOVE_REFERENCE_T(...) __remove_reference(__VA_ARGS__)
#elif _CCCL_HAS_BUILTIN(__remove_reference_t) && defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_BUILTIN_REMOVE_REFERENCE_T(...) __remove_reference_t(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_reference_t)

#if _CCCL_HAS_BUILTIN(__remove_volatile) && defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_BUILTIN_REMOVE_VOLATILE(...) __remove_volatile(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_volatile)

#if _CCCL_HAS_BUILTIN(__type_pack_element)
#  define _CCCL_BUILTIN_TYPE_PACK_ELEMENT(...) __type_pack_element<__VA_ARGS__>
#endif // _CCCL_HAS_BUILTIN(__type_pack_element)

// NVCC prior to 12.0 have trouble with pack expansion into __type_pack_element in an alias template
#if defined(_CCCL_CUDACC_BELOW_12_0)
#  undef _CCCL_BUILTIN_TYPE_PACK_ELEMENT
#endif // _CCCL_CUDACC_BELOW_12_0

#if _CCCL_CHECK_BUILTIN(underlying_type) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40700) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_UNDERLYING_TYPE(...) __underlying_type(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(underlying_type) && gcc >= 4.7

#endif // __CCCL_BUILTIN_H
