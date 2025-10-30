//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

// Thread API
#ifndef _CCCL_HAS_THREAD_API_EXTERNAL
#  if _CCCL_COMPILER(NVRTC) || defined(__EMSCRIPTEN__)
#    define _CCCL_HAS_THREAD_API_EXTERNAL
#  endif
#endif // _CCCL_HAS_THREAD_API_EXTERNAL

#ifndef _CCCL_HAS_THREAD_API_CUDA
#  if ((_CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC)) || defined(__EMSCRIPTEN__))
#    define _CCCL_HAS_THREAD_API_CUDA
#  endif // ((_CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC)) || defined(__EMSCRIPTEN__))
#endif // _CCCL_HAS_THREAD_API_CUDA

#ifndef _CCCL_HAS_THREAD_API_WIN32
#  if _CCCL_COMPILER(MSVC) && !defined(_CCCL_HAS_THREAD_API_CUDA)
#    define _CCCL_HAS_THREAD_API_WIN32
#  endif // _CCCL_COMPILER(MSVC) && !defined(_CCCL_HAS_THREAD_API_CUDA)
#endif // _CCCL_HAS_THREAD_API_WIN32

#if !defined(_CCCL_HAS_THREAD_API_PTHREAD) && !defined(_CCCL_HAS_THREAD_API_WIN32) \
  && !defined(_CCCL_HAS_THREAD_API_EXTERNAL)
#  if defined(__GNU__) || _CCCL_OS(LINUX) || _CCCL_OS(APPLE) || _CCCL_OS(QNX) \
    || (defined(__MINGW32__) && _CCCL_HAS_INCLUDE(<pthread.h>))
#    define _CCCL_HAS_THREAD_API_PTHREAD
#  elif defined(_WIN32)
#    define _CCCL_HAS_THREAD_API_WIN32
#  else
#    define _CCCL_UNSUPPORTED_THREAD_API
#  endif // _CCCL_HAS_THREAD_API
#endif

#ifndef __STDCPP_THREADS__
#  define __STDCPP_THREADS__ 1
#endif // __STDCPP_THREADS__

#endif // _CUDA_STD___INTERNAL_THREAD_API_H
