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
#if __cpp_if_constexpr < 201606L
#  define _CCCL_NO_IF_CONSTEXPR
#endif // !defined(__cpp_if_constexpr)

// concepts are only available from C++20 onwards
#if _CCCL_STD_VER <= 2017 || __cpp_concepts < 201907L
#  define _CCCL_NO_CONCEPTS
#endif // _CCCL_STD_VER <= 2017 || __cpp_concepts < 201907L

// CTAD is only available from C++17 onwards
#if __cpp_deduction_guides < 201611L
#  define _CCCL_NO_DEDUCTION_GUIDES
#endif // __cpp_deduction_guides < 201611L

// Fold expressions are only available from C++17 onwards
#if __cpp_fold_expressions < 201603L
#  define _CCCL_NO_FOLD_EXPRESSIONS
#endif // __cpp_fold_expressions < 201603L

// Inline variables are only available from C++17 onwards
#if __cpp_inline_variables < 201606L
#  define _CCCL_NO_INLINE_VARIABLES
#endif // __cpp_inline_variables < 201606L

// noexcept function types are only available from C++17 onwards
#if __cpp_noexcept_function_type < 201510L
#  define _CCCL_NO_NOEXCEPT_FUNCTION_TYPE
#endif // __cpp_noexcept_function_type < 201510L

// Declaring a non-type template parameters with auto is only available from C++17 onwards
#if __cpp_nontype_template_parameter_auto < 201606L
#  define _CCCL_NO_NONTYPE_TEMPLATE_PARAMETER_AUTO
#endif // __cpp_nontype_template_parameter_auto < 201606L

// Three way comparison is only available from C++20 onwards
#if _CCCL_STD_VER <= 2017 || __cpp_impl_three_way_comparison < 201907L
#  define _CCCL_NO_THREE_WAY_COMPARISON
#endif // _CCCL_STD_VER <= 2017 || __cpp_impl_three_way_comparison < 201907L

// Variable templates are only available from C++14 onwards and require some compiler support
#if __cpp_variable_templates < 201304L
#  define _CCCL_NO_VARIABLE_TEMPLATES
#endif // __cpp_variable_templates < 201304L

///////////////////////////////////////////////////////////////////////////////
// Conditionally use certain language features depending on availability
///////////////////////////////////////////////////////////////////////////////

#if defined(_CCCL_NO_INLINE_VARIABLES)
#  define _CCCL_INLINE_VAR
#else // ^^^ _CCCL_NO_INLINE_VARIABLES ^^^ / vvv !_CCCL_NO_INLINE_VARIABLES vvv
#  define _CCCL_INLINE_VAR inline
#endif // !_CCCL_NO_INLINE_VARIABLES

#if defined(_CCCL_NO_NOEXCEPT_FUNCTION_TYPE)
#  define _CCCL_FUNCTION_TYPE_NOEXCEPT
#else // ^^^ _CCCL_NO_NOEXCEPT_FUNCTION_TYPE ^^^ / vvv !_CCCL_NO_NOEXCEPT_FUNCTION_TYPE vvv
#  define _CCCL_FUNCTION_TYPE_NOEXCEPT noexcept
#endif // !_CCCL_NO_NOEXCEPT_FUNCTION_TYPE

#if defined(_CCCL_NO_NONTYPE_TEMPLATE_PARAMETER_AUTO)
#  define _CCCL_NTTP_AUTO unsigned long long int
#else // ^^^ _CCCL_NO_NONTYPE_TEMPLATE_PARAMETER_AUTO ^^^ / vvv !_CCCL_NO_NONTYPE_TEMPLATE_PARAMETER_AUTO vvv
#  define _CCCL_NTTP_AUTO auto
#endif // !_CCCL_NO_NONTYPE_TEMPLATE_PARAMETER_AUTO

// Variable templates are more efficient most of the time, so we want to use them rather than structs when possible
#if defined(_CCCL_NO_VARIABLE_TEMPLATES)
#  define _CCCL_TRAIT(__TRAIT, ...) __TRAIT<__VA_ARGS__>::value
#else // ^^^ _CCCL_NO_VARIABLE_TEMPLATES ^^^ / vvv !_CCCL_NO_VARIABLE_TEMPLATES vvv
#  define _CCCL_TRAIT(__TRAIT, ...) __TRAIT##_v<__VA_ARGS__>
#endif // !_CCCL_NO_VARIABLE_TEMPLATES

// We need to treat host and device separately
#if defined(__CUDA_ARCH__)
#  define _CCCL_GLOBAL_CONSTANT _CCCL_DEVICE constexpr
#else // ^^^ __CUDA_ARCH__ ^^^ / vvv !__CUDA_ARCH__ vvv
#  define _CCCL_GLOBAL_CONSTANT _CCCL_INLINE_VAR constexpr
#endif // __CUDA_ARCH__

#endif // __CCCL_DIALECT_H
