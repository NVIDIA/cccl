//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_DIAGNOSTIC_H
#define __CCCL_DIAGNOSTIC_H

#include "../__cccl/compiler.h"
#include "../__cccl/system_header.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Enable us to selectively silence host compiler warnings
#define _CCCL_TOSTRING2(_STR) #_STR
#define _CCCL_TOSTRING(_STR)  _CCCL_TOSTRING2(_STR)
#ifdef _CCCL_COMPILER_CLANG
#  define _CCCL_DIAG_PUSH                _Pragma("clang diagnostic push")
#  define _CCCL_DIAG_POP                 _Pragma("clang diagnostic pop")
#  define _CCCL_DIAG_SUPPRESS_CLANG(str) _Pragma(_CCCL_TOSTRING(clang diagnostic ignored str))
#  define _CCCL_DIAG_SUPPRESS_GCC(str)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str)
#elif defined(_CCCL_COMPILER_GCC) || defined(_CCCL_COMPILER_ICC)
#  define _CCCL_DIAG_PUSH _Pragma("GCC diagnostic push")
#  define _CCCL_DIAG_POP  _Pragma("GCC diagnostic pop")
#  define _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str) _Pragma(_CCCL_TOSTRING(GCC diagnostic ignored str))
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str)
#elif defined(_CCCL_COMPILER_NVHPC)
#  define _CCCL_DIAG_PUSH _Pragma("diagnostic push")
#  define _CCCL_DIAG_POP  _Pragma("diagnostic pop")
#  define _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str) _Pragma(_CCCL_TOSTRING(diag_suppress str))
#  define _CCCL_DIAG_SUPPRESS_MSVC(str)
#elif defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_DIAG_PUSH __pragma(warning(push))
#  define _CCCL_DIAG_POP  __pragma(warning(pop))
#  define _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str) __pragma(warning(disable : str))
#else
#  define _CCCL_DIAG_PUSH
#  define _CCCL_DIAG_POP
#  define _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str)
#endif

// Convenient shortcuts to silence common warnings
#if defined(_CCCL_COMPILER_CLANG)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH \
    _CCCL_DIAG_PUSH                           \
    _CCCL_DIAG_SUPPRESS_CLANG("-Wdeprecated") \
    _CCCL_DIAG_SUPPRESS_CLANG("-Wdeprecated-declarations")
#  define _CCCL_SUPPRESS_DEPRECATED_POP _CCCL_DIAG_POP
#elif defined(_CCCL_COMPILER_GCC) || defined(_CCCL_COMPILER_ICC)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH \
    _CCCL_DIAG_PUSH                           \
    _CCCL_DIAG_SUPPRESS_GCC("-Wdeprecated")   \
    _CCCL_DIAG_SUPPRESS_GCC("-Wdeprecated-declarations")
#  define _CCCL_SUPPRESS_DEPRECATED_POP _CCCL_DIAG_POP
#elif defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH \
    _CCCL_DIAG_PUSH                           \
    _CCCL_DIAG_SUPPRESS_MSVC(4996)
#  define _CCCL_SUPPRESS_DEPRECATED_POP _CCCL_DIAG_POP
#else // !_CCCL_COMPILER_CLANG && !_CCCL_COMPILER_GCC
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH
#  define _CCCL_SUPPRESS_DEPRECATED_POP
#endif // !_CCCL_COMPILER_CLANG && !_CCCL_COMPILER_GCC

// Enable us to selectively silence cuda compiler warnings
#if defined(_CCCL_CUDA_COMPILER)
#  if defined(_CCCL_CUDA_COMPILER_CLANG)
#    define _CCCL_NV_DIAG_SUPPRESS(_WARNING)
#    define _CCCL_NV_DIAG_DEFAULT(_WARNING)
#  elif defined(__NVCC_DIAG_PRAGMA_SUPPORT__)
#    if defined(_CCCL_COMPILER_MSVC)
#      define _CCCL_NV_DIAG_SUPPRESS(_WARNING) __pragma(_CCCL_TOSTRING(nv_diag_suppress _WARNING))
#      define _CCCL_NV_DIAG_DEFAULT(_WARNING)  __pragma(_CCCL_TOSTRING(nv_diag_default _WARNING))
#    else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv
#      define _CCCL_NV_DIAG_SUPPRESS(_WARNING) \
        _Pragma(_CCCL_TOSTRING(nv_diagnostic push)) _Pragma(_CCCL_TOSTRING(nv_diag_suppress _WARNING))
#      define _CCCL_NV_DIAG_DEFAULT(_WARNING) _Pragma(_CCCL_TOSTRING(nv_diagnostic pop))
#    endif // !_CCCL_COMPILER_MSVC
#  elif defined(_CCCL_COMPILER_NVHPC)
#    define _CCCL_NV_DIAG_SUPPRESS(_WARNING) \
      _Pragma(_CCCL_TOSTRING(diagnostic push)) _Pragma(_CCCL_TOSTRING(diag_suppress _WARNING))
#    define _CCCL_NV_DIAG_DEFAULT(_WARNING) _Pragma(_CCCL_TOSTRING(diagnostic pop))
#  else // ^^^ __NVCC_DIAG_PRAGMA_SUPPORT__ ^^^ / vvv !__NVCC_DIAG_PRAGMA_SUPPORT__ vvv
#    if defined(_CCCL_COMPILER_MSVC_2017) // MSVC 2017 has issues with restoring the warning
#      define _CCCL_NV_DIAG_SUPPRESS(_WARNING) __pragma(_CCCL_TOSTRING(diag_suppress _WARNING))
#      define _CCCL_NV_DIAG_DEFAULT(_WARNING)
#    elif defined(_CCCL_COMPILER_MSVC)
#      define _CCCL_NV_DIAG_SUPPRESS(_WARNING) __pragma(_CCCL_TOSTRING(diag_suppress _WARNING))
#      define _CCCL_NV_DIAG_DEFAULT(_WARNING)  __pragma(_CCCL_TOSTRING(diag_default _WARNING))
#    else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv
#      define _CCCL_NV_DIAG_SUPPRESS(_WARNING) _Pragma(_CCCL_TOSTRING(diag_suppress _WARNING))
#      define _CCCL_NV_DIAG_DEFAULT(_WARNING)  _Pragma(_CCCL_TOSTRING(diag_default _WARNING))
#    endif // !_CCCL_COMPILER_MSVC
#  endif // !__NVCC_DIAG_PRAGMA_SUPPORT__
#else // ^^^ _CCCL_CUDA_COMPILER ^^^ / vvv !_CCCL_CUDA_COMPILER vvv
#  define _CCCL_NV_DIAG_SUPPRESS(_WARNING)
#  define _CCCL_NV_DIAG_DEFAULT(_WARNING)
#endif // other compilers

#endif // __CCCL_DIAGNOSTIC_H
