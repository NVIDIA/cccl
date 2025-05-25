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

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Enable us to selectively silence host compiler warnings
#if _CCCL_COMPILER(CLANG)
#  define _CCCL_DIAG_PUSH                _CCCL_PRAGMA(clang diagnostic push)
#  define _CCCL_DIAG_POP                 _CCCL_PRAGMA(clang diagnostic pop)
#  define _CCCL_DIAG_SUPPRESS_CLANG(str) _CCCL_PRAGMA(clang diagnostic ignored str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str)
#elif _CCCL_COMPILER(GCC)
#  define _CCCL_DIAG_PUSH _CCCL_PRAGMA(GCC diagnostic push)
#  define _CCCL_DIAG_POP  _CCCL_PRAGMA(GCC diagnostic pop)
#  define _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str) _CCCL_PRAGMA(GCC diagnostic ignored str)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str)
#elif _CCCL_COMPILER(NVHPC)
#  define _CCCL_DIAG_PUSH _CCCL_PRAGMA(diagnostic push)
#  define _CCCL_DIAG_POP  _CCCL_PRAGMA(diagnostic pop)
#  define _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str) _CCCL_PRAGMA(diag_suppress str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str)
#elif _CCCL_COMPILER(MSVC)
#  define _CCCL_DIAG_PUSH _CCCL_PRAGMA(warning(push))
#  define _CCCL_DIAG_POP  _CCCL_PRAGMA(warning(pop))
#  define _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str) _CCCL_PRAGMA(warning(disable : str))
#else
#  define _CCCL_DIAG_PUSH
#  define _CCCL_DIAG_POP
#  define _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str)
#endif

// Convenient shortcuts to silence common warnings
#if _CCCL_COMPILER(CLANG)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH      \
    _CCCL_DIAG_PUSH                           \
    _CCCL_DIAG_SUPPRESS_CLANG("-Wdeprecated") \
    _CCCL_DIAG_SUPPRESS_CLANG("-Wdeprecated-declarations")
#  define _CCCL_SUPPRESS_DEPRECATED_POP _CCCL_DIAG_POP
#elif _CCCL_COMPILER(GCC)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH    \
    _CCCL_DIAG_PUSH                         \
    _CCCL_DIAG_SUPPRESS_GCC("-Wdeprecated") \
    _CCCL_DIAG_SUPPRESS_GCC("-Wdeprecated-declarations")
#  define _CCCL_SUPPRESS_DEPRECATED_POP _CCCL_DIAG_POP
#elif _CCCL_COMPILER(NVHPC)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH         \
    _CCCL_DIAG_PUSH                              \
    _CCCL_DIAG_SUPPRESS_NVHPC(deprecated_entity) \
    _CCCL_DIAG_SUPPRESS_NVHPC(deprecated_entity_with_custom_message)
#  define _CCCL_SUPPRESS_DEPRECATED_POP _CCCL_DIAG_POP
#elif _CCCL_COMPILER(MSVC)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH \
    _CCCL_DIAG_PUSH                      \
    _CCCL_DIAG_SUPPRESS_MSVC(4996)
#  define _CCCL_SUPPRESS_DEPRECATED_POP _CCCL_DIAG_POP
#else // !_CCCL_COMPILER(CLANG) && !_CCCL_COMPILER(GCC) && !_CCCL_COMPILER(NVHPC) && !_CCCL_COMPILER(MSVC)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH
#  define _CCCL_SUPPRESS_DEPRECATED_POP
#endif // !_CCCL_COMPILER(CLANG) && !_CCCL_COMPILER(GCC) && !_CCCL_COMPILER(NVHPC) && !_CCCL_COMPILER(MSVC)

// Enable us to selectively silence cuda compiler warnings
#if _CCCL_HAS_CUDA_COMPILER()
#  if _CCCL_CUDA_COMPILER(CLANG)
#    define _CCCL_NV_DIAG_SUPPRESS(_WARNING)
#    define _CCCL_NV_DIAG_DEFAULT(_WARNING)
#  elif defined(__NVCC_DIAG_PRAGMA_SUPPORT__)
#    if _CCCL_COMPILER(MSVC)
#      define _CCCL_NV_DIAG_SUPPRESS(_WARNING) _CCCL_PRAGMA(nv_diag_suppress _WARNING)
#      define _CCCL_NV_DIAG_DEFAULT(_WARNING)  _CCCL_PRAGMA(nv_diag_default _WARNING)
#    else // ^^^ _CCCL_COMPILER_{MSVC}^^^ / vvv !_CCCL_COMPILER_{MSVC} vvv
#      define _CCCL_NV_DIAG_SUPPRESS(_WARNING) _CCCL_PRAGMA(nv_diagnostic push) _CCCL_PRAGMA(nv_diag_suppress _WARNING)
#      define _CCCL_NV_DIAG_DEFAULT(_WARNING)  _CCCL_PRAGMA(nv_diagnostic pop)
#    endif // !_CCCL_COMPILER(MSVC)
#  elif _CCCL_COMPILER(NVHPC)
#    define _CCCL_NV_DIAG_SUPPRESS(_WARNING) _CCCL_PRAGMA(diagnostic push) _CCCL_PRAGMA(diag_suppress _WARNING)
#    define _CCCL_NV_DIAG_DEFAULT(_WARNING)  _CCCL_PRAGMA(diagnostic pop)
#  else // ^^^ __NVCC_DIAG_PRAGMA_SUPPORT__ ^^^ / vvv !__NVCC_DIAG_PRAGMA_SUPPORT__ vvv
#    if _CCCL_COMPILER(GCC) // these compilers have issues with restoring the warning
#      define _CCCL_NV_DIAG_SUPPRESS(_WARNING) _CCCL_PRAGMA(diag_suppress _WARNING)
#      define _CCCL_NV_DIAG_DEFAULT(_WARNING)
#    else // ^^^ _CCCL_COMPILER(GCC) ^^^ / vvv !_CCCL_COMPILER(GCC) vvv
#      define _CCCL_NV_DIAG_SUPPRESS(_WARNING) _CCCL_PRAGMA(diag_suppress _WARNING)
#      define _CCCL_NV_DIAG_DEFAULT(_WARNING)  _CCCL_PRAGMA(diag_default _WARNING)
#    endif // !_CCCL_COMPILER(GCC)
#  endif // !__NVCC_DIAG_PRAGMA_SUPPORT__
#else // ^^^ _CCCL_HAS_CUDA_COMPILER() ^^^ / vvv !_CCCL_HAS_CUDA_COMPILER() vvv
#  define _CCCL_NV_DIAG_SUPPRESS(_WARNING)
#  define _CCCL_NV_DIAG_DEFAULT(_WARNING)
#endif // ^^^ !_CCCL_HAS_CUDA_COMPILER() ^^^

#if _CCCL_COMPILER(MSVC)
#  define _CCCL_HAS_PRAGMA_MSVC_WARNING
#  if !defined(_LIBCUDACXX_DISABLE_PRAGMA_MSVC_WARNING)
#    define _CCCL_USE_PRAGMA_MSVC_WARNING
#  endif // !_LIBCUDACXX_DISABLE_PRAGMA_MSVC_WARNING
#endif // !_CCCL_COMPILER(MSVC)

#endif // __CCCL_DIAGNOSTIC_H
