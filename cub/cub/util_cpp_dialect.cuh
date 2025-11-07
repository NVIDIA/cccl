// SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! Detect the version of the C++ standard used by the compiler.

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

// Deprecation warnings may be silenced by defining the following macros. These
// may be combined.
// - CCCL_IGNORE_DEPRECATED_COMPILER
//   Ignore deprecation warnings when using deprecated compilers. Compiling
//   with deprecated C++ dialects will still issue warnings.

//! Deprecated [Since 3.0]
#  define CUB_CPP_DIALECT _CCCL_STD_VER

// Define CUB_COMPILER_DEPRECATION macro:
#  if _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#    define CUB_COMP_DEPR_IMPL(msg) _CCCL_PRAGMA(message(__FILE__ ":" _CCCL_TO_STRING(__LINE__) ": warning: " #msg))
#  else // clang / gcc:
#    define CUB_COMP_DEPR_IMPL(msg) _CCCL_PRAGMA(GCC warning #msg)
#  endif

// Compiler checks:
// clang-format off
#  define CUB_COMPILER_DEPRECATION(REQ) \
    CUB_COMP_DEPR_IMPL(CUB requires at least REQ. Define CCCL_IGNORE_DEPRECATED_COMPILER to suppress this message.)

#  define CUB_COMPILER_DEPRECATION_SOFT(REQ, CUR)                                                        \
    CUB_COMP_DEPR_IMPL(                                                                                  \
      CUB requires at least REQ. CUR is deprecated but still supported. CUR support will be removed in a \
        future release. Define CCCL_IGNORE_DEPRECATED_COMPILER to suppress this message.)
// clang-format on

#  ifndef CCCL_IGNORE_DEPRECATED_COMPILER
#    if _CCCL_COMPILER(GCC, <, 7)
CUB_COMPILER_DEPRECATION(GCC 7.0);
#    elif _CCCL_COMPILER(CLANG, <, 7)
CUB_COMPILER_DEPRECATION(Clang 7.0);
#    elif _CCCL_COMPILER(MSVC, <, 19, 10)
// <2017. Hard upgrade message:
CUB_COMPILER_DEPRECATION(MSVC 2019(19.20 / 16.0 / 14.20));
#    endif
#  endif // CCCL_IGNORE_DEPRECATED_COMPILER

#  undef CUB_COMPILER_DEPRECATION_SOFT
#  undef CUB_COMPILER_DEPRECATION

// C++17 dialect check:
#  ifndef CCCL_IGNORE_DEPRECATED_CPP_DIALECT
#    if _CCCL_STD_VER < 2017
#      error CUB requires at least C++17. Define CCCL_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
#    endif // _CCCL_STD_VER < 2017
#  endif

#  undef CUB_COMP_DEPR_IMPL

#endif // !_CCCL_DOXYGEN_INVOKED
