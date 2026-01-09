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

#include <cuda/std/__cccl/preprocessor.h>

// By default all of the diagnostic suppression macros expand to nothing.
#define _CCCL_DIAG_SUPPRESS_CLANG(_WARNING)
#define _CCCL_DIAG_SUPPRESS_GCC(_WARNING)
#define _CCCL_DIAG_SUPPRESS_NV(_WARNING)
#define _CCCL_DIAG_SUPPRESS_NVHPC(_WARNING)
#define _CCCL_DIAG_SUPPRESS_MSVC(_WARNING)

// Each host compiler must define _CCCL_DIAG_SUPPRESS_MEOW(W), _CCCL_DIAG_PUSH_HOST, _CCCL_DIAG_POP_HOST and
// _CCCL_SUPPRESS_DEPRECATED_HOST macros.
#if _CCCL_COMPILER(CLANG)
#  undef _CCCL_DIAG_SUPPRESS_CLANG
#  define _CCCL_DIAG_SUPPRESS_CLANG(_WARNING) _CCCL_PRAGMA(clang diagnostic ignored _WARNING)
#  define _CCCL_DIAG_PUSH_HOST                _CCCL_PRAGMA(clang diagnostic push)
#  define _CCCL_DIAG_POP_HOST                 _CCCL_PRAGMA(clang diagnostic pop)
#  define _CCCL_SUPPRESS_DEPRECATED_HOST      _CCCL_DIAG_SUPPRESS(CLANG, "-Wdeprecated", "-Wdeprecated-declarations")
#elif _CCCL_COMPILER(GCC)
#  undef _CCCL_DIAG_SUPPRESS_GCC
#  define _CCCL_DIAG_SUPPRESS_GCC(_WARNING) _CCCL_PRAGMA(GCC diagnostic ignored _WARNING)
#  define _CCCL_DIAG_PUSH_HOST              _CCCL_PRAGMA(GCC diagnostic push)
#  define _CCCL_DIAG_POP_HOST               _CCCL_PRAGMA(GCC diagnostic pop)
#  define _CCCL_SUPPRESS_DEPRECATED_HOST    _CCCL_DIAG_SUPPRESS(GCC, "-Wdeprecated", "-Wdeprecated-declarations")
#elif _CCCL_COMPILER(NVHPC)
#  undef _CCCL_DIAG_SUPPRESS_NVHPC
#  define _CCCL_DIAG_SUPPRESS_NVHPC(_WARNING) _CCCL_PRAGMA(diag_suppress _WARNING)
#  define _CCCL_DIAG_PUSH_HOST                _CCCL_PRAGMA(diagnostic push)
#  define _CCCL_DIAG_POP_HOST                 _CCCL_PRAGMA(diagnostic pop)
#  define _CCCL_SUPPRESS_DEPRECATED_HOST \
    _CCCL_DIAG_SUPPRESS(NVHPC, deprecated_entity, deprecated_entity_with_custom_message)
#elif _CCCL_COMPILER(MSVC)
#  undef _CCCL_DIAG_SUPPRESS_MSVC
#  define _CCCL_DIAG_SUPPRESS_MSVC(_WARNING) _CCCL_PRAGMA(warning(disable : _WARNING))
#  define _CCCL_DIAG_PUSH_HOST               _CCCL_PRAGMA(warning(push))
#  define _CCCL_DIAG_POP_HOST                _CCCL_PRAGMA(warning(pop))
#  define _CCCL_SUPPRESS_DEPRECATED_HOST     _CCCL_DIAG_SUPPRESS(MSVC, 4996)
#else // ^^^ known host compilers ^^^ / vvv unknown or no host compilers vvv
#  define _CCCL_DIAG_PUSH_HOST
#  define _CCCL_DIAG_POP_HOST
#  define _CCCL_SUPPRESS_DEPRECATED_HOST
#endif // ^^^ unknown or no host compilers ^^^

// For nvcc and nvrtc we need to add support for nv diagnostics and append them to the host diagnostics macros.
// Other device compilers and host only compilers will expand to just the host macros.
#if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_CUDA_COMPILER(NVRTC)
#  undef _CCCL_DIAG_SUPPRESS_NV
#  define _CCCL_DIAG_SUPPRESS_NV(_WARNING) _CCCL_PRAGMA(nv_diag_suppress _WARNING)
#  define _CCCL_DIAG_PUSH_NV               _CCCL_PRAGMA(nv_diagnostic push)
#  define _CCCL_DIAG_POP_NV                _CCCL_PRAGMA(nv_diagnostic pop)
#  define _CCCL_SUPPRESS_DEPRECATED_NV     _CCCL_DIAG_SUPPRESS(NV, 1444, 20199)

// Don't use host diagnostics for device compilation.
#  if _CCCL_DEVICE_COMPILATION()
#    define _CCCL_DIAG_PUSH           _CCCL_DIAG_PUSH_NV
#    define _CCCL_DIAG_POP            _CCCL_DIAG_POP_NV
#    define _CCCL_SUPPRESS_DEPRECATED _CCCL_SUPPRESS_DEPRECATED_NV
#  else // ^^^ _CCCL_DEVICE_COMPILATION() ^^^ / vvv !_CCCL_DEVICE_COMPILATION() vvv
#    define _CCCL_DIAG_PUSH           _CCCL_DIAG_PUSH_NV _CCCL_DIAG_PUSH_HOST
#    define _CCCL_DIAG_POP            _CCCL_DIAG_POP_HOST _CCCL_DIAG_POP_NV
#    define _CCCL_SUPPRESS_DEPRECATED _CCCL_SUPPRESS_DEPRECATED_NV _CCCL_SUPPRESS_DEPRECATED_HOST
#  endif // ^^^ !_CCCL_DEVICE_COMPILATION() ^^^
#else // ^^^ nvcc or nvrtc ^^^ / vvv other cuda compilers and host only compilation vvv
#  define _CCCL_DIAG_PUSH           _CCCL_DIAG_PUSH_HOST
#  define _CCCL_DIAG_POP            _CCCL_DIAG_POP_HOST
#  define _CCCL_SUPPRESS_DEPRECATED _CCCL_SUPPRESS_DEPRECATED_HOST
#endif // ^^^ other cuda compilers and host only compilation ^^^

#define _CCCL_DIAG_SUPPRESS(_COMPILER, ...) _CCCL_PP_FOR_EACH(_CCCL_DIAG_SUPPRESS_##_COMPILER, __VA_ARGS__)
#define _CCCL_DIAG_PUSH_AND_SUPPRESS(_COMPILER, ...) \
  _CCCL_DIAG_PUSH _CCCL_PP_FOR_EACH(_CCCL_DIAG_SUPPRESS_##_COMPILER, __VA_ARGS__)
#define _CCCL_DIAG_PUSH_AND_SUPPRESS_DEPRECATED _CCCL_DIAG_PUSH _CCCL_SUPPRESS_DEPRECATED
#define _CCCL_DIAG_POP_DEPRECATED               _CCCL_DIAG_POP

#endif // __CCCL_DIAGNOSTIC_H
