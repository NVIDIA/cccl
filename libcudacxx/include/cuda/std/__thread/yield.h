// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___THREAD_YIELD_H
#define _CUDA_STD___THREAD_YIELD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_THREAD_API(CUDA)
// nothing to include
#elif _CCCL_HAS_THREAD_API(PTHREAD)
#  include <sched.h>
#elif _CCCL_HAS_THREAD_API(WIN32)
#  include <windows.h>
#endif // _CCCL_HAS_THREAD_API(WIN32)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HOST_COMPILATION()
_CCCL_HOST_API inline void __cccl_thread_yield_host()
{
#  if _CCCL_HAS_THREAD_API(PTHREAD)
  ::sched_yield();
#  elif _CCCL_HAS_THREAD_API(WIN32)
  ::SwitchToThread();
#  endif // _CCCL_HAS_THREAD_API(WIN32)
}
#endif // _CCCL_HOST_COMPILATION()

_CCCL_HOST_DEVICE_API inline void __cccl_thread_yield()
{
  NV_IF_TARGET(NV_IS_HOST, ::cuda::std::__cccl_thread_yield_host();)
}

#if _CCCL_HOST_ARCH(ARM64) && _CCCL_OS(LINUX)
#  define __LIBCUDACXX_ASM_THREAD_YIELD (asm volatile("yield" :: :);)
#elif _CCCL_HOST_ARCH(X86_64) && _CCCL_OS(LINUX)
#  define __LIBCUDACXX_ASM_THREAD_YIELD (asm volatile("pause" :: :);)
#else // ^^^  _CCCL_HOST_ARCH(X86_64) ^^^ / vvv ! _CCCL_HOST_ARCH(X86_64) vvv
#  define __LIBCUDACXX_ASM_THREAD_YIELD (;)
#endif // ! _CCCL_HOST_ARCH(X86_64)

_CCCL_HOST_DEVICE_API
inline void __cccl_thread_yield_processor(){NV_IF_TARGET(NV_IS_HOST, __LIBCUDACXX_ASM_THREAD_YIELD)}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___THREAD_YIELD_H
