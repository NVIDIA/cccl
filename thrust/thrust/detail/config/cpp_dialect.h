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

#include <thrust/detail/config/compiler.h> // IWYU pragma: export

// Deprecation warnings may be silenced by defining the following macros. These
// may be combined.
// - CCCL_IGNORE_DEPRECATED_CPP_DIALECT:
//   Ignore all deprecated C++ dialects and outdated compilers.
// - CCCL_IGNORE_DEPRECATED_CPP_11:
//   Ignore deprecation warnings when compiling with C++11. C++03 and outdated
//   compilers will still issue warnings.
// - CCCL_IGNORE_DEPRECATED_CPP_14:
//   Ignore deprecation warnings when compiling with C++14. C++03 and outdated
//   compilers will still issue warnings.
// - CCCL_IGNORE_DEPRECATED_COMPILER
//   Ignore deprecation warnings when using deprecated compilers. Compiling
//   with C++03, C++11 and C++14 will still issue warnings.

#define THRUST_CPP_DIALECT _CCCL_STD_VER

// Define THRUST_COMPILER_DEPRECATION macro:
#if _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define THRUST_COMP_DEPR_IMPL(msg) _CCCL_PRAGMA(message(__FILE__ ":" _CCCL_TO_STRING(__LINE__) ": warning: " #msg))
#else // clang / gcc:
#  define THRUST_COMP_DEPR_IMPL(msg) _CCCL_PRAGMA(GCC warning #msg)
#endif

// clang-format off
#define THRUST_COMPILER_DEPRECATION(REQ) \
  THRUST_COMP_DEPR_IMPL(Thrust requires at least REQ. Define CCCL_IGNORE_DEPRECATED_COMPILER to suppress this message.)

#define THRUST_COMPILER_DEPRECATION_SOFT(REQ, CUR)                                                        \
  THRUST_COMP_DEPR_IMPL(                                                                                  \
    Thrust requires at least REQ. CUR is deprecated but still supported. CUR support will be removed in a \
      future release. Define CCCL_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.)
// clang-format on

#ifndef CCCL_IGNORE_DEPRECATED_COMPILER

// Compiler checks:
#  if _CCCL_COMPILER(GCC, <, 5)
THRUST_COMPILER_DEPRECATION(GCC 5.0);
#  elif _CCCL_COMPILER(CLANG, <, 7)
THRUST_COMPILER_DEPRECATION(Clang 7.0);
#  elif _CCCL_COMPILER(MSVC, <, 19, 10)
// <2017. Hard upgrade message:
THRUST_COMPILER_DEPRECATION(MSVC 2019(19.20 / 16.0 / 14.20));
#  elif _CCCL_COMPILER(MSVC2017)
// >=2017, <2019. Soft deprecation message:
THRUST_COMPILER_DEPRECATION_SOFT(MSVC 2019(19.20 / 16.0 / 14.20), MSVC 2017);
#  endif

#endif // CCCL_IGNORE_DEPRECATED_COMPILER

#if _CCCL_STD_VER < 2011
// <C++11. Hard upgrade message:
THRUST_COMPILER_DEPRECATION(C++ 17);
#elif _CCCL_STD_VER == 2011 && !defined(CCCL_IGNORE_DEPRECATED_CPP_11)
// =C++11. Soft upgrade message:
THRUST_COMPILER_DEPRECATION_SOFT(C++ 17, C++ 11);
#elif _CCCL_STD_VER == 2014 && !defined(CCCL_IGNORE_DEPRECATED_CPP_14)
// =C++14. Soft upgrade message:
THRUST_COMPILER_DEPRECATION_SOFT(C++ 17, C++ 14);
#endif // _CCCL_STD_VER >= 2017

#undef THRUST_COMPILER_DEPRECATION_SOFT
#undef THRUST_COMPILER_DEPRECATION
#undef THRUST_COMP_DEPR_IMPL
#undef THRUST_COMP_DEPR_IMPL0
#undef THRUST_COMP_DEPR_IMPL1
