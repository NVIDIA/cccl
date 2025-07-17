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
#  define _CCCL_DIAG_PUSH                     _CCCL_PRAGMA(clang diagnostic push)
#  define _CCCL_DIAG_POP                      _CCCL_PRAGMA(clang diagnostic pop)
#  define _CCCL_DIAG_SUPPRESS_CLANG(_WARNING) _CCCL_PRAGMA(clang diagnostic ignored _WARNING)
#  define _CCCL_DIAG_SUPPRESS_GCC(_WARNING)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(_WARNING)
#  define _CCCL_DIAG_SUPPRESS_MSVC(_WARNING)
#elif _CCCL_COMPILER(GCC)
#  define _CCCL_DIAG_PUSH _CCCL_PRAGMA(GCC diagnostic push)
#  define _CCCL_DIAG_POP  _CCCL_PRAGMA(GCC diagnostic pop)
#  define _CCCL_DIAG_SUPPRESS_CLANG(_WARNING)
#  define _CCCL_DIAG_SUPPRESS_GCC(_WARNING) _CCCL_PRAGMA(GCC diagnostic ignored _WARNING)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(_WARNING)
#  define _CCCL_DIAG_SUPPRESS_MSVC(_WARNING)
#elif _CCCL_COMPILER(NVHPC)
#  define _CCCL_DIAG_PUSH _CCCL_PRAGMA(diagnostic push)
#  define _CCCL_DIAG_POP  _CCCL_PRAGMA(diagnostic pop)
#  define _CCCL_DIAG_SUPPRESS_CLANG(_WARNING)
#  define _CCCL_DIAG_SUPPRESS_GCC(_WARNING)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(_WARNING) _CCCL_PRAGMA(diag_suppress _WARNING)
#  define _CCCL_DIAG_SUPPRESS_MSVC(_WARNING)
#elif _CCCL_COMPILER(MSVC)
#  define _CCCL_DIAG_PUSH _CCCL_PRAGMA(warning(push))
#  define _CCCL_DIAG_POP  _CCCL_PRAGMA(warning(pop))
#  define _CCCL_DIAG_SUPPRESS_CLANG(_WARNING)
#  define _CCCL_DIAG_SUPPRESS_GCC(_WARNING)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(_WARNING)
#  define _CCCL_DIAG_SUPPRESS_MSVC(_WARNING) _CCCL_PRAGMA(warning(disable : _WARNING))
#else
#  define _CCCL_DIAG_PUSH
#  define _CCCL_DIAG_POP
#  define _CCCL_DIAG_SUPPRESS_CLANG(_WARNING)
#  define _CCCL_DIAG_SUPPRESS_GCC(_WARNING)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(_WARNING)
#  define _CCCL_DIAG_SUPPRESS_MSVC(_WARNING)
#endif

// Enable us to selectively silence cuda compiler warnings
#if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_COMPILER(NVRTC)
#  if defined(__NVCC_DIAG_PRAGMA_SUPPORT__)
#    define _CCCL_NV_DIAG_PUSH()               _CCCL_PRAGMA(nv_diagnostic push)
#    define _CCCL_NV_DIAG_POP()                _CCCL_PRAGMA(nv_diagnostic pop)
#    define _CCCL_DIAG_SUPPRESS_NVCC(_WARNING) _CCCL_PRAGMA(nv_diag_suppress _WARNING)
#    define _CCCL_BEGIN_NV_DIAG_SUPPRESS(...) \
      _CCCL_NV_DIAG_PUSH() _CCCL_PP_FOR_EACH(_CCCL_DIAG_SUPPRESS_NVCC, __VA_ARGS__)
#    define _CCCL_END_NV_DIAG_SUPPRESS() _CCCL_NV_DIAG_POP()
#  else // ^^^ __NVCC_DIAG_PRAGMA_SUPPORT__ ^^^ / vvv !__NVCC_DIAG_PRAGMA_SUPPORT__ vvv
#    define _CCCL_NV_DIAG_PUSH()               _CCCL_PRAGMA(diagnostic push)
#    define _CCCL_NV_DIAG_POP()                _CCCL_PRAGMA(diagnostic pop)
#    define _CCCL_DIAG_SUPPRESS_NVCC(_WARNING) _CCCL_PRAGMA(diag_suppress _WARNING)
#    define _CCCL_BEGIN_NV_DIAG_SUPPRESS(...) \
      _CCCL_NV_DIAG_PUSH() _CCCL_PP_FOR_EACH(_CCCL_DIAG_SUPPRESS_NVCC, __VA_ARGS__)
#    define _CCCL_END_NV_DIAG_SUPPRESS() _CCCL_NV_DIAG_POP()
#  endif // !__NVCC_DIAG_PRAGMA_SUPPORT__
#else // ^^^ _CCCL_CUDA_COMPILER(NVCC) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVCC) vvv
#  define _CCCL_NV_DIAG_PUSH()
#  define _CCCL_NV_DIAG_POP()
#  define _CCCL_DIAG_SUPPRESS_NVCC(_WARNING)
#  define _CCCL_BEGIN_NV_DIAG_SUPPRESS(...)
#  define _CCCL_END_NV_DIAG_SUPPRESS()
#endif // !_CCCL_CUDA_COMPILER(NVCC)

// Convenient shortcuts to silence common warnings
#if _CCCL_COMPILER(CLANG)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH                   \
    _CCCL_DIAG_PUSH                                        \
    _CCCL_DIAG_SUPPRESS_CLANG("-Wdeprecated")              \
    _CCCL_DIAG_SUPPRESS_CLANG("-Wdeprecated-declarations") \
    _CCCL_BEGIN_NV_DIAG_SUPPRESS(1444, 20199)
#  define _CCCL_SUPPRESS_DEPRECATED_POP _CCCL_NV_DIAG_POP() _CCCL_DIAG_POP
#elif _CCCL_COMPILER(GCC)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH                 \
    _CCCL_DIAG_PUSH                                      \
    _CCCL_DIAG_SUPPRESS_GCC("-Wdeprecated")              \
    _CCCL_DIAG_SUPPRESS_GCC("-Wdeprecated-declarations") \
    _CCCL_BEGIN_NV_DIAG_SUPPRESS(1444, 20199)
#  define _CCCL_SUPPRESS_DEPRECATED_POP _CCCL_NV_DIAG_POP() _CCCL_DIAG_POP
#elif _CCCL_COMPILER(NVHPC)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH                             \
    _CCCL_DIAG_PUSH                                                  \
    _CCCL_DIAG_SUPPRESS_NVHPC(deprecated_entity)                     \
    _CCCL_DIAG_SUPPRESS_NVHPC(deprecated_entity_with_custom_message) \
    _CCCL_BEGIN_NV_DIAG_SUPPRESS(1444, 20199)
#  define _CCCL_SUPPRESS_DEPRECATED_POP _CCCL_NV_DIAG_POP() _CCCL_DIAG_POP
#elif _CCCL_COMPILER(MSVC)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH \
    _CCCL_DIAG_PUSH                      \
    _CCCL_DIAG_SUPPRESS_MSVC(4996)       \
    _CCCL_BEGIN_NV_DIAG_SUPPRESS(1444)
#  define _CCCL_SUPPRESS_DEPRECATED_POP _CCCL_NV_DIAG_POP() _CCCL_DIAG_POP
#elif _CCCL_COMPILER(NVRTC)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH _CCCL_BEGIN_NV_DIAG_SUPPRESS(1444, 20199)
#  define _CCCL_SUPPRESS_DEPRECATED_POP  _CCCL_NV_DIAG_POP()
#else // unknown compiler
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH
#  define _CCCL_SUPPRESS_DEPRECATED_POP
#endif // unknown compiler

#if _CCCL_COMPILER(MSVC)
#  define _CCCL_HAS_PRAGMA_MSVC_WARNING
#  if !defined(_LIBCUDACXX_DISABLE_PRAGMA_MSVC_WARNING)
#    define _CCCL_USE_PRAGMA_MSVC_WARNING
#  endif // !_LIBCUDACXX_DISABLE_PRAGMA_MSVC_WARNING

// https://github.com/microsoft/STL/blob/master/stl/inc/yvals_core.h#L353
// warning C4100: 'quack': unreferenced formal parameter
// warning C4127: conditional expression is constant
// warning C4180: qualifier applied to function type has no meaning; ignored
// warning C4197: 'purr': top-level volatile in cast is ignored
// warning C4324: 'roar': structure was padded due to alignment specifier
// warning C4455: literal suffix identifiers that do not start with an underscore are reserved
// warning C4503: 'hum': decorated name length exceeded, name was truncated
// warning C4522: 'woof' : multiple assignment operators specified
// warning C4668: 'meow' is not defined as a preprocessor macro, replacing with '0' for '#if/#elif'
// warning C4800: 'boo': forcing value to bool 'true' or 'false' (performance warning)
// warning C4996: 'meow': was declared deprecated
#  define _CCCL_MSVC_DISABLED_WARNINGS 4100 4127 4180 4197 4296 4324 4455 4503 4522 4668 4800 4996 /**/
#  define _CCCL_MSVC_WARNINGS_PUSH \
    _CCCL_PRAGMA(warning(push)) _CCCL_PRAGMA(warning(disable : _CCCL_MSVC_DISABLED_WARNINGS))
#  define _CCCL_MSVC_WARNINGS_POP _CCCL_PRAGMA(warning(pop))
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#  define _CCCL_MSVC_WARNINGS_PUSH
#  define _CCCL_MSVC_WARNINGS_POP
#endif // !_CCCL_COMPILER(MSVC)

#ifndef _CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO
#  if _CCCL_COMPILER(NVRTC)
#    define _CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO
#  endif
#endif // _CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO

#if defined(_CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO)
#  define _CCCL_PUSH_MACROS _CCCL_MSVC_WARNINGS_PUSH
#  define _CCCL_POP_MACROS  _CCCL_MSVC_WARNINGS_POP
#else // ^^^ _CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO ^^^ / vvv !_CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO vvv
#  define _CCCL_PUSH_MACROS         \
    _CCCL_PRAGMA(push_macro("min")) \
    _CCCL_PRAGMA(push_macro("max")) _CCCL_PRAGMA(push_macro("interface")) _CCCL_MSVC_WARNINGS_PUSH
#  define _CCCL_POP_MACROS         \
    _CCCL_PRAGMA(pop_macro("min")) \
    _CCCL_PRAGMA(pop_macro("max")) _CCCL_PRAGMA(pop_macro("interface")) _CCCL_MSVC_WARNINGS_POP
#endif // !_CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO

#endif // __CCCL_DIAGNOSTIC_H
