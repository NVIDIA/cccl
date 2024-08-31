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
#define _CCCL_TOSTRING2(_STR) #_STR
#define _CCCL_TOSTRING(_STR)  _CCCL_TOSTRING2(_STR)
#ifdef _CCCL_COMPILER_CLANG
#  define _CCCL_DIAG_PUSH                _Pragma("clang diagnostic push")
#  define _CCCL_DIAG_POP                 _Pragma("clang diagnostic pop")
#  define _CCCL_DIAG_SUPPRESS_CLANG(str) _Pragma(_CCCL_TOSTRING(clang diagnostic ignored str))
#  define _CCCL_DIAG_SUPPRESS_GCC(str)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str)
#  define _CCCL_DIAG_SUPPRESS_ICC(str)
#elif defined(_CCCL_COMPILER_GCC)
#  define _CCCL_DIAG_PUSH _Pragma("GCC diagnostic push")
#  define _CCCL_DIAG_POP  _Pragma("GCC diagnostic pop")
#  define _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str) _Pragma(_CCCL_TOSTRING(GCC diagnostic ignored str))
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str)
#  define _CCCL_DIAG_SUPPRESS_ICC(str)
#elif defined(_CCCL_COMPILER_ICC)
#  define _CCCL_DIAG_PUSH _Pragma("GCC diagnostic push")
#  define _CCCL_DIAG_POP  _Pragma("GCC diagnostic pop")
#  define _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str) _Pragma(_CCCL_TOSTRING(GCC diagnostic ignored str))
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str)
#  define _CCCL_DIAG_SUPPRESS_ICC(str) _Pragma(_CCCL_TOSTRING(warning disable str))
#elif defined(_CCCL_COMPILER_NVHPC)
#  define _CCCL_DIAG_PUSH _Pragma("diagnostic push")
#  define _CCCL_DIAG_POP  _Pragma("diagnostic pop")
#  define _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str) _Pragma(_CCCL_TOSTRING(diag_suppress str))
#  define _CCCL_DIAG_SUPPRESS_MSVC(str)
#  define _CCCL_DIAG_SUPPRESS_ICC(str)
#elif defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_DIAG_PUSH __pragma(warning(push))
#  define _CCCL_DIAG_POP  __pragma(warning(pop))
#  define _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str) __pragma(warning(disable : str))
#  define _CCCL_DIAG_SUPPRESS_ICC(str)
#else
#  define _CCCL_DIAG_PUSH
#  define _CCCL_DIAG_POP
#  define _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define _CCCL_DIAG_SUPPRESS_GCC(str)
#  define _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define _CCCL_DIAG_SUPPRESS_MSVC(str)
#  define _CCCL_DIAG_SUPPRESS_ICC(str)
#endif

// Convenient shortcuts to silence common warnings
#if defined(_CCCL_COMPILER_CLANG)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH      \
    _CCCL_DIAG_PUSH                           \
    _CCCL_DIAG_SUPPRESS_CLANG("-Wdeprecated") \
    _CCCL_DIAG_SUPPRESS_CLANG("-Wdeprecated-declarations")
#  define _CCCL_SUPPRESS_DEPRECATED_POP _CCCL_DIAG_POP
#elif defined(_CCCL_COMPILER_GCC) || defined(_CCCL_COMPILER_ICC)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH    \
    _CCCL_DIAG_PUSH                         \
    _CCCL_DIAG_SUPPRESS_GCC("-Wdeprecated") \
    _CCCL_DIAG_SUPPRESS_GCC("-Wdeprecated-declarations")
#  define _CCCL_SUPPRESS_DEPRECATED_POP _CCCL_DIAG_POP
#elif defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_SUPPRESS_DEPRECATED_PUSH \
    _CCCL_DIAG_PUSH                      \
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
#    elif defined(_CCCL_COMPILER_ICC) // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv _CCCL_COMPILER_ICCvvv
#      define _CCCL_NV_DIAG_SUPPRESS(_WARNING) _Pragma(_CCCL_TOSTRING(nv_diag_suppress _WARNING))
#      define _CCCL_NV_DIAG_DEFAULT(_WARNING)  _Pragma(_CCCL_TOSTRING(nv_diag_default _WARNING))
#    else // ^^^ _CCCL_COMPILER_ICC^^^ / vvv !_CCCL_COMPILER_{MSVC,ICC} vvv
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

#if defined(_CCCL_COMPILER_MSVC)
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
#  define _CCCL_MSVC_WARNINGS_PUSH     __pragma(warning(push)) __pragma(warning(disable : _CCCL_MSVC_DISABLED_WARNINGS))
#  define _CCCL_MSVC_WARNINGS_POP      __pragma(warning(pop))
#else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv
#  define _CCCL_MSVC_WARNINGS_PUSH
#  define _CCCL_MSVC_WARNINGS_POP
#endif // !_CCCL_COMPILER_MSVC

#ifndef _CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO
#  if defined(_CCCL_COMPILER_MSVC_2017) || defined(_CCCL_COMPILER_NVRTC)
#    define _CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO
#  endif
#endif // _CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO

#if defined(_CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO)
#  define _CCCL_PUSH_MACROS _CCCL_MSVC_WARNINGS_PUSH
#  define _CCCL_POP_MACROS  _CCCL_MSVC_WARNINGS_POP
#else // ^^^ _CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO ^^^ / vvv !_CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO vvv
#  if defined(_CCCL_COMPILER_MSVC)
#    define _CCCL_PUSH_MACROS __pragma(push_macro("min")) __pragma(push_macro("max")) _CCCL_MSVC_WARNINGS_PUSH
#    define _CCCL_POP_MACROS  __pragma(pop_macro("min")) __pragma(pop_macro("max")) _CCCL_MSVC_WARNINGS_POP
#  else
#    define _CCCL_PUSH_MACROS _Pragma("push_macro(\"min\")") _Pragma("push_macro(\"max\")")
#    define _CCCL_POP_MACROS  _Pragma("pop_macro(\"min\")") _Pragma("pop_macro(\"max\")")
#  endif
#endif // !_CCCL_HAS_NO_PRAGMA_PUSH_POP_MACRO

#endif // __CCCL_DIAGNOSTIC_H
