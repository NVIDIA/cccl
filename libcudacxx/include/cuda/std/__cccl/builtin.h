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

#ifndef __has_builtin
#  define __has_builtin(__x) 0
#endif // !__has_builtin

#ifndef __has_feature
#  define __has_feature(__x) 0
#endif // !__has_feature

// '__is_identifier' returns '0' if '__x' is a reserved identifier provided by the compiler and '1' otherwise.
#ifndef __is_identifier
#  define __is_identifier(__x) 1
#endif // !__is_identifier

#define __has_keyword(__x) !(__is_identifier(__x))

// https://bugs.llvm.org/show_bug.cgi?id=44517
#define __check_builtin(__x) (__has_builtin(__##__x) || __has_keyword(__##__x) || __has_feature(__x))

// We work around old clang versions (before clang-10) not supporting __has_builtin via __check_builtin
// We work around old intel versions (before 2021.3)   not supporting __has_builtin via __check_builtin
// We work around old nvhpc versions (before 2022.11)  not supporting __has_builtin via __check_builtin
// MSVC needs manual handling, has no real way of checking builtins so all is manual
// GCC  needs manual handling, before gcc-10 as that finally supports __has_builtin

#if __check_builtin(array_rank)
#  define _CCCL_BUILTIN_ARRAY_RANK(...) __array_rank(__VA_ARGS__)
#endif // __check_builtin(array_rank)

// nvhpc has a bug where it supports __builtin_addressof but does not mark it via __check_builtin
#if __check_builtin(builtin_addressof) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 70000) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVHPC)
#  define _CCCL_BUILTIN_ADDRESSOF(...) __builtin_addressof(__VA_ARGS__)
#endif // __check_builtin(builtin_addressof)

// MSVC supports __builtin_bit_cast from 19.25 on
// clang-9 supports __builtin_bit_cast but it is not a constant expression
#if (__check_builtin(builtin_bit_cast) || (defined(_CCCL_COMPILER_MSVC) && _CCCL_MSVC_VERSION > 1925)) \
  && !defined(_CCCL_CUDACC_BELOW_11_7) && !(defined(_CCCL_COMPILER_CLANG) && _CCCL_CLANG_VERSION < 100000)
#  define _CCCL_BUILTIN_BIT_CAST(...) __builtin_bit_cast(__VA_ARGS__)
#endif // __check_builtin(builtin_bit_cast)

#if __check_builtin(builtin_is_constant_evaluated) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 90000) \
  || (defined(_CCCL_COMPILER_MSVC) && _CCCL_MSVC_VERSION > 1924 && !defined(_CCCL_CUDACC_BELOW_11_3))
#  define _CCCL_BUILTIN_IS_CONSTANT_EVALUATED(...) __builtin_is_constant_evaluated(__VA_ARGS__)
#endif // __check_builtin(builtin_is_constant_evaluated)

// NVCC and NVRTC in C++11 mode freaks out about `__builtin_is_constant_evaluated`.
#if _CCCL_STD_VER < 2014 \
  && (defined(_CCCL_CUDA_COMPILER_NVCC) || defined(_CCCL_COMPILER_NVRTC) || defined(_CCCL_COMPILER_NVHPC))
#  undef _CCCL_BUILTIN_IS_CONSTANT_EVALUATED
#endif // _CCCL_STD_VER < 2014 && _CCCL_CUDA_COMPILER_NVCC

#if __check_builtin(builtin_launder) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 70000)
#  define _CCCL_BUILTIN_LAUNDER(...) __builtin_launder(__VA_ARGS__)
#endif // __check_builtin(builtin_launder)

// Disabled due to libstdc++ conflict
#if 0 // __check_builtin(decay)
#  define _CCCL_BUILTIN_DECAY(...) __decay(__VA_ARGS__)
#endif // __check_builtin(decay)

#if __check_builtin(has_nothrow_assign) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_HAS_NOTHROW_ASSIGN(...) __has_nothrow_assign(__VA_ARGS__)
#endif // __check_builtin(has_nothrow_assign)

#if __check_builtin(has_nothrow_constructor) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_HAS_NOTHROW_CONSTRUCTOR(...) __has_nothrow_constructor(__VA_ARGS__)
#endif // __check_builtin(has_nothrow_constructor)

#if __check_builtin(has_nothrow_copy) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_HAS_NOTHROW_COPY(...) __has_nothrow_copy(__VA_ARGS__)
#endif // __check_builtin(has_nothrow_copy)

#if __check_builtin(has_trivial_constructor) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_HAS_TRIVIAL_CONSTRUCTOR(...) __has_trivial_constructor(__VA_ARGS__)
#endif // __check_builtin(has_trivial_constructor)

#if __check_builtin(has_trivial_destructor) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_HAS_TRIVIAL_DESTRUCTOR(...) __has_trivial_destructor(__VA_ARGS__)
#endif // __check_builtin(has_trivial_destructor)

#if __check_builtin(has_unique_object_representations) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 70000)
#  define _CCCL_BUILTIN_HAS_UNIQUE_OBJECT_REPRESENTATIONS(...) __has_unique_object_representations(__VA_ARGS__)
#endif // __check_builtin(has_unique_object_representations)

#if __check_builtin(has_virtual_destructor) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_HAS_VIRTUAL_DESTRUCTOR(...) __has_virtual_destructor(__VA_ARGS__)
#endif // __check_builtin(has_virtual_destructor)

#if __check_builtin(is_aggregate) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 70000) \
  || (defined(_CCCL_COMPILER_MSVC) && _CCCL_MSVC_VERSION > 1914) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_AGGREGATE(...) __is_aggregate(__VA_ARGS__)
#endif // __check_builtin(is_aggregate)

#if __check_builtin(is_array)
#  define _CCCL_BUILTIN_IS_ARRAY(...) __is_array(__VA_ARGS__)
#endif // __check_builtin(is_array)

// TODO: Clang incorrectly reports that __is_array is true for T[0].
//       Re-enable the branch once https://llvm.org/PR54705 is fixed.
#ifndef _LIBCUDACXX_USE_IS_ARRAY_FALLBACK
#  if defined(_CCCL_COMPILER_CLANG)
#    define _LIBCUDACXX_USE_IS_ARRAY_FALLBACK
#  endif // _CCCL_COMPILER_CLANG
#endif // !_LIBCUDACXX_USE_IS_ARRAY_FALLBACK

#if __check_builtin(is_assignable) || defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_BUILTIN_IS_ASSIGNABLE(...) __is_assignable(__VA_ARGS__)
#endif // __check_builtin(is_assignable)

#if __check_builtin(is_base_of) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_BASE_OF(...) __is_base_of(__VA_ARGS__)
#endif // __check_builtin(is_base_of)

#if __check_builtin(is_class) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_CLASS(...) __is_class(__VA_ARGS__)
#endif // __check_builtin(is_class)

#if __check_builtin(is_constructible) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 80000) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_CONSTRUCTIBLE(...) __is_constructible(__VA_ARGS__)
#endif // __check_builtin(is_constructible)

#if __check_builtin(is_convertible_to) || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_CONVERTIBLE_TO(...) __is_convertible_to(__VA_ARGS__)
#endif // __check_builtin(is_convertible_to)

#if __check_builtin(is_destructible) || defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_BUILTIN_IS_DESTRUCTIBLE(...) __is_destructible(__VA_ARGS__)
#endif // __check_builtin(is_destructible)

#if __check_builtin(is_empty) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_EMPTY(...) __is_empty(__VA_ARGS__)
#endif // __check_builtin(is_empty)

#if __check_builtin(is_enum) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_ENUM(...) __is_enum(__VA_ARGS__)
#endif // __check_builtin(is_enum)

#if __check_builtin(is_final) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40700) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_FINAL(...) __is_final(__VA_ARGS__)
#endif // __check_builtin(is_final)

#if __check_builtin(is_function) && !defined(_CCCL_CUDA_COMPILER_NVCC)
#  define _CCCL_BUILTIN_IS_FUNCTION(...) __is_function(__VA_ARGS__)
#endif // __check_builtin(is_function)

#if __check_builtin(is_literal_type) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40600) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_LITERAL(...) __is_literal_type(__VA_ARGS__)
#endif // __check_builtin(is_literal_type)

#if __check_builtin(is_lvalue_reference)
#  define _CCCL_BUILTIN_IS_LVALUE_REFERENCE(...) __is_lvalue_reference(__VA_ARGS__)
#endif // __check_builtin(is_lvalue_reference)

#ifndef _LIBCUDACXX_USE_IS_LVALUE_REFERENCE_FALLBACK
#  if defined(_CCCL_CUDACC_BELOW_11_3)
#    define _LIBCUDACXX_USE_IS_LVALUE_REFERENCE_FALLBACK
#  endif // nvcc < 11.3
#endif // !_LIBCUDACXX_USE_IS_LVALUE_REFERENCE_FALLBACK

#if __check_builtin(is_nothrow_assignable) || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_NOTHROW_ASSIGNABLE(...) __is_nothrow_assignable(__VA_ARGS__)
#endif // __check_builtin(is_nothrow_assignable)

#if __check_builtin(is_nothrow_constructible) || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_NOTHROW_CONSTRUCTIBLE(...) __is_nothrow_constructible(__VA_ARGS__)
#endif // __check_builtin(is_nothrow_constructible)

#if __check_builtin(is_nothrow_destructible) || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE(...) __is_nothrow_destructible(__VA_ARGS__)
#endif // __check_builtin(is_nothrow_destructible)

#if __check_builtin(is_object)
#  define _CCCL_BUILTIN_IS_OBJECT(...) __is_object(__VA_ARGS__)
#endif // __check_builtin(is_object)

#ifndef _LIBCUDACXX_USE_IS_OBJECT_FALLBACK
#  if defined(_CCCL_CUDACC_BELOW_11_3)
#    define _LIBCUDACXX_USE_IS_OBJECT_FALLBACK
#  endif // _CCCL_CUDACC_BELOW_11_3
#endif // !_LIBCUDACXX_USE_IS_OBJECT_FALLBACK

#if __check_builtin(is_pod) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_POD(...) __is_pod(__VA_ARGS__)
#endif // __check_builtin(is_pod)

// libstdc++ defines this as a function, breaking functionality
#if 0 // __check_builtin(is_pointer)
#  define _CCCL_BUILTIN_IS_POINTER(...) __is_pointer(__VA_ARGS__)
#endif // __check_builtin(is_pointer)

#if __check_builtin(is_polymorphic) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_POLYMORPHIC(...) __is_polymorphic(__VA_ARGS__)
#endif // __check_builtin(is_polymorphic)

#if __check_builtin(is_reference)
#  define _CCCL_BUILTIN_IS_REFERENCE(...) __is_reference(__VA_ARGS__)
#endif // __check_builtin(is_reference)

// Disabled due to libstdc++ conflict
#if 0 // __check_builtin(is_referenceable)
#  define _CCCL_BUILTIN_IS_REFERENCEABLE(...) __is_referenceable(__VA_ARGS__)
#endif // __check_builtin(is_referenceable)

#if __check_builtin(is_rvalue_reference)
#  define _CCCL_BUILTIN_IS_RVALUE_REFERENCE(...) __is_rvalue_reference(__VA_ARGS__)
#endif // __check_builtin(is_rvalue_reference)

#if __check_builtin(is_same) && !defined(_CCCL_CUDA_COMPILER_NVCC)
#  define _CCCL_BUILTIN_IS_SAME(...) __is_same(__VA_ARGS__)
#endif // __check_builtin(is_same)

// libstdc++ defines this as a function, breaking functionality
#if 0 // __check_builtin(is_scalar)
#  define _CCCL_BUILTIN_IS_SCALAR(...) __is_scalar(__VA_ARGS__)
#endif // __check_builtin(is_scalar)

// libstdc++ defines this as a function, breaking functionality
#if 0 // __check_builtin(is_signed)
#  define _CCCL_BUILTIN_IS_SIGNED(...) __is_signed(__VA_ARGS__)
#endif // __check_builtin(is_signed)

#if __check_builtin(is_standard_layout) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40700) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_STANDARD_LAYOUT(...) __is_standard_layout(__VA_ARGS__)
#endif // __check_builtin(is_standard_layout)

#if __check_builtin(is_trivial) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40500) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_TRIVIAL(...) __is_trivial(__VA_ARGS__)
#endif // __check_builtin(is_trivial)

#if __check_builtin(is_trivially_assignable) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 50100) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_TRIVIALLY_ASSIGNABLE(...) __is_trivially_assignable(__VA_ARGS__)
#endif // __check_builtin(is_trivially_assignable)

#if __check_builtin(is_trivially_constructible) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 50100) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_TRIVIALLY_CONSTRUCTIBLE(...) __is_trivially_constructible(__VA_ARGS__)
#endif // __check_builtin(is_trivially_constructible)

#if __check_builtin(is_trivially_copyable) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 50100) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_TRIVIALLY_COPYABLE(...) __is_trivially_copyable(__VA_ARGS__)
#endif // __check_builtin(is_trivially_copyable)

#if __check_builtin(is_trivially_destructible) || defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_BUILTIN_IS_TRIVIALLY_DESTRUCTIBLE(...) __is_trivially_destructible(__VA_ARGS__)
#endif // __check_builtin(is_trivially_destructible)

#if __check_builtin(is_union) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40300) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_IS_UNION(...) __is_union(__VA_ARGS__)
#endif // __check_builtin(is_union)

#if __check_builtin(is_unsigned)
#  define _CCCL_BUILTIN_IS_UNSIGNED(...) __is_unsigned(__VA_ARGS__)
#endif // __check_builtin(is_unsigned)

#ifndef _LIBCUDACXX_USE_IS_UNSIGNED_FALLBACK
#  if defined(_CCCL_CUDACC_BELOW_11_3)
#    define _LIBCUDACXX_USE_IS_UNSIGNED_FALLBACK
#  endif // _CCCL_CUDACC_BELOW_11_3
#endif // !_LIBCUDACXX_USE_IS_UNSIGNED_FALLBACK

// libstdc++ defines this as a function, breaking functionality
#if 0 // __check_builtin(is_void)
#  define _CCCL_BUILTIN_IS_VOID(...) __is_void(__VA_ARGS__)
#endif // __check_builtin(is_void)

// Disabled due to libstdc++ conflict
#if 0 // __check_builtin(make_signed)
#  define _CCCL_BUILTIN_MAKE_SIGNED(...) __make_signed(__VA_ARGS__)
#endif // __check_builtin(make_signed)

// Disabled due to libstdc++ conflict
#if 0 // __check_builtin(make_unsigned)
#  define _CCCL_BUILTIN_MAKE_UNSIGNED(...) __make_unsigned(__VA_ARGS__)
#endif // __check_builtin(make_unsigned)

// Disabled due to libstdc++ conflict
#if 0 // __check_builtin(remove_all_extents)
#  define _CCCL_BUILTIN_REMOVE_ALL_EXTENTS(...) __remove_all_extents(__VA_ARGS__)
#endif // __check_builtin(remove_all_extents)

// Disabled due to libstdc++ conflict
#if 0 // __check_builtin(remove_const)
#  define _CCCL_BUILTIN_REMOVE_CONST(...) __remove_const(__VA_ARGS__)
#endif // __check_builtin(remove_const)

// Disabled due to libstdc++ conflict
#if 0 // __check_builtin(remove_cv)
#  define _CCCL_BUILTIN_REMOVE_CV(...) __remove_cv(__VA_ARGS__)
#endif // __check_builtin(remove_cv)

// Disabled due to libstdc++ conflict
#if 0 // __check_builtin(remove_cvref)
#  define _CCCL_BUILTIN_REMOVE_CVREF(...) __remove_cvref(__VA_ARGS__)
#endif // __check_builtin(remove_cvref)

// Disabled due to libstdc++ conflict
#if 0 // __check_builtin(remove_extent)
#  define _CCCL_BUILTIN_REMOVE_EXTENT(...) __remove_extent(__VA_ARGS__)
#endif // __check_builtin(remove_extent)

// Disabled due to libstdc++ conflict
#if 0 // __check_builtin(remove_pointer)
#  define _CCCL_BUILTIN_REMOVE_POINTER(...) __remove_pointer(__VA_ARGS__)
#endif // __check_builtin(remove_pointer)

// Disabled due to libstdc++ conflict
#if 0 // __check_builtin(remove_reference_t)
#  define _CCCL_BUILTIN_REMOVE_REFERENCE_T(...) __remove_reference_t(__VA_ARGS__)
#endif // __check_builtin(remove_reference_t)

// Disabled due to libstdc++ conflict
#if 0 // __check_builtin(remove_volatile)
#  define _CCCL_BUILTIN_REMOVE_VOLATILE(...) __remove_volatile(__VA_ARGS__)
#endif // __check_builtin(remove_volatile)

#if __check_builtin(underlying_type) || (defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION >= 40700) \
  || defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_BUILTIN_UNDERLYING_TYPE(...) __underlying_type(__VA_ARGS__)
#endif // __check_builtin(underlying_type)

#endif // __CCCL_BUILTIN_H
