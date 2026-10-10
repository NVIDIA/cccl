//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _CUDA_STD___INTERNAL_THREAD_API_H
#define _CUDA_STD___INTERNAL_THREAD_API_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Each thread API is detected on its own, so more than one may be true. A CUDA
// translation unit on Linux has both the CUDA and pthread APIs: device code uses
// the CUDA thread API, and the host can still call pthread.
//
// `_CCCL_HAS_THREAD_API(API)` is 1 when that API is available and 0 otherwise.
// `API` is `EXTERNAL`, `CUDA`, `WIN32`, or `PTHREAD`.

#define _CCCL_HAS_THREAD_API_EXTERNAL() 0
#define _CCCL_HAS_THREAD_API_CUDA()     0
#define _CCCL_HAS_THREAD_API_WIN32()    0
#define _CCCL_HAS_THREAD_API_PTHREAD()  0

#if _CCCL_COMPILER(NVRTC) || defined(__EMSCRIPTEN__) || _CCCL_HOSTJIT()
#  undef _CCCL_HAS_THREAD_API_EXTERNAL
#  define _CCCL_HAS_THREAD_API_EXTERNAL() 1
#endif // _CCCL_COMPILER(NVRTC) || defined(__EMSCRIPTEN__) || _CCCL_HOSTJIT()

#if (_CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC)) || _CCCL_HAS_THREAD_API_EXTERNAL()
#  undef _CCCL_HAS_THREAD_API_CUDA
#  define _CCCL_HAS_THREAD_API_CUDA() 1
#endif // (_CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC)) || _CCCL_HAS_THREAD_API_EXTERNAL()

#if _CCCL_HAS_THREAD_API_EXTERNAL()
// No windows thread API
#elif _CCCL_OS(WINDOWS)
#  undef _CCCL_HAS_THREAD_API_WIN32
#  define _CCCL_HAS_THREAD_API_WIN32() 1
#endif // (!_CCCL_COMPILER(NVRTC) && _CCCL_OS(WINDOWS))

#if _CCCL_HAS_THREAD_API_EXTERNAL()
// No pthread
#elif defined(__GNU__) || _CCCL_OS(LINUX) || _CCCL_OS(APPLE) || _CCCL_OS(QNX)
#  undef _CCCL_HAS_THREAD_API_PTHREAD
#  define _CCCL_HAS_THREAD_API_PTHREAD() 1
#elif defined(__MINGW32__) && __has_include(<pthread.h>)
#  undef _CCCL_HAS_THREAD_API_PTHREAD
#  define _CCCL_HAS_THREAD_API_PTHREAD() 1
#endif // ^^^ pthread thread API

#define _CCCL_HAS_THREAD_API(_API) _CCCL_HAS_THREAD_API_##_API()

#if !_CCCL_HAS_THREAD_API(EXTERNAL) && !_CCCL_HAS_THREAD_API(CUDA) && !_CCCL_HAS_THREAD_API(WIN32) \
  && !_CCCL_HAS_THREAD_API(PTHREAD)
#  define _CCCL_UNSUPPORTED_THREAD_API
#endif // no thread API

#ifndef __STDCPP_THREADS__
#  define __STDCPP_THREADS__ 1
#endif // __STDCPP_THREADS__

#endif // _CUDA_STD___INTERNAL_THREAD_API_H
