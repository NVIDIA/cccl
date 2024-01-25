/*
 *  Copyright 2020 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file cpp_dialect.h
 *  \brief Detect the version of the C++ standard used by the compiler.
 */

#pragma once

#include <thrust/detail/config/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/config/compiler.h>

// Deprecation warnings may be silenced by defining the following macros. These
// may be combined.
// - THRUST_IGNORE_DEPRECATED_CPP_DIALECT:
//   Ignore all deprecated C++ dialects and outdated compilers.
// - THRUST_IGNORE_DEPRECATED_CPP_11:
//   Ignore deprecation warnings when compiling with C++11. C++03 and outdated
//   compilers will still issue warnings.
// - THRUST_IGNORE_DEPRECATED_COMPILER
//   Ignore deprecation warnings when using deprecated compilers. Compiling
//   with C++03 and C++11 will still issue warnings.

// Check for the CUB opt-outs as well:
#if !defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT) && \
     defined(CUB_IGNORE_DEPRECATED_CPP_DIALECT)
#  define    THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#endif
#if !defined(THRUST_IGNORE_DEPRECATED_CPP_11) && \
     defined(CUB_IGNORE_DEPRECATED_CPP_11)
#  define    THRUST_IGNORE_DEPRECATED_CPP_11
#endif
#if !defined(THRUST_IGNORE_DEPRECATED_COMPILER) && \
     defined(CUB_IGNORE_DEPRECATED_COMPILER)
#  define    THRUST_IGNORE_DEPRECATED_COMPILER
#endif

#ifdef THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#  define THRUST_IGNORE_DEPRECATED_CPP_11
#  define THRUST_IGNORE_DEPRECATED_COMPILER
#endif

#define THRUST_CPP_DIALECT _CCCL_STD_VER

// Define THRUST_COMPILER_DEPRECATION macro:
#if defined(_CCCL_COMPILER_MSVC)
#  define THRUST_COMP_DEPR_IMPL(msg) \
    __pragma(message(__FILE__ ":" THRUST_COMP_DEPR_IMPL0(__LINE__) ": warning: " #msg))
#  define THRUST_COMP_DEPR_IMPL0(x) THRUST_COMP_DEPR_IMPL1(x)
#  define THRUST_COMP_DEPR_IMPL1(x) #x
#else // clang / gcc:
#  define THRUST_COMP_DEPR_IMPL(msg) THRUST_COMP_DEPR_IMPL0(GCC warning #msg)
#  define THRUST_COMP_DEPR_IMPL0(expr) _Pragma(#expr)
#  define THRUST_COMP_DEPR_IMPL1 /* intentionally blank */
#endif

#define THRUST_COMPILER_DEPRECATION(REQ) \
  THRUST_COMP_DEPR_IMPL(Thrust requires at least REQ. Define THRUST_IGNORE_DEPRECATED_COMPILER to suppress this message.)

#define THRUST_COMPILER_DEPRECATION_SOFT(REQ, CUR) \
  THRUST_COMP_DEPR_IMPL(Thrust requires at least REQ. CUR is deprecated but still supported. CUR support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.)

#ifndef THRUST_IGNORE_DEPRECATED_COMPILER

// Compiler checks:
#  if defined(_CCCL_COMPILER_GCC) && THRUST_GCC_VERSION < 50000
     THRUST_COMPILER_DEPRECATION(GCC 5.0);
#  elif defined(_CCCL_COMPILER_CLANG) && THRUST_CLANG_VERSION < 70000
     THRUST_COMPILER_DEPRECATION(Clang 7.0);
#  elif defined(_CCCL_COMPILER_MSVC) && THRUST_MSVC_VERSION < 1910
     // <2017. Hard upgrade message:
     THRUST_COMPILER_DEPRECATION(MSVC 2019 (19.20/16.0/14.20));
#  elif defined(_CCCL_COMPILER_MSVC) && THRUST_MSVC_VERSION < 1920
     // >=2017, <2019. Soft deprecation message:
     THRUST_COMPILER_DEPRECATION_SOFT(MSVC 2019 (19.20/16.0/14.20), MSVC 2017);
#  endif

#endif // THRUST_IGNORE_DEPRECATED_COMPILER

#ifndef THRUST_IGNORE_DEPRECATED_DIALECT

// Dialect checks:
#  if _CCCL_STD_VER < 2011
     // <C++11. Hard upgrade message:
     THRUST_COMPILER_DEPRECATION(C++14);
#  elif _CCCL_STD_VER == 2011 && !defined(THRUST_IGNORE_DEPRECATED_CPP_11)
     // =C++11. Soft upgrade message:
     THRUST_COMPILER_DEPRECATION_SOFT(C++14, C++11);
#  endif

#endif // THRUST_IGNORE_DEPRECATED_DIALECT

#undef THRUST_COMPILER_DEPRECATION_SOFT
#undef THRUST_COMPILER_DEPRECATION
#undef THRUST_COMP_DEPR_IMPL
#undef THRUST_COMP_DEPR_IMPL0
#undef THRUST_COMP_DEPR_IMPL1
