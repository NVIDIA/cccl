// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___THREAD_POLL_H
#define _CUDA_STD___THREAD_POLL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__chrono/high_resolution_clock.h>
#include <cuda/std/__thread/sleep_for.h>
#include <cuda/std/__thread/yield.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#define _LIBCUDACXX_POLLING_COUNT 16

template <class _Fn>
_CCCL_HOST_DEVICE_API inline bool __cccl_thread_poll_with_backoff(
  _Fn&& __f, ::cuda::std::chrono::nanoseconds __max = ::cuda::std::chrono::nanoseconds::zero())
{
  ::cuda::std::chrono::high_resolution_clock::time_point const __start =
    ::cuda::std::chrono::high_resolution_clock::now();
  for (int __count = 0;;)
  {
    if (__f())
    {
      return true;
    }
    if (__count < _LIBCUDACXX_POLLING_COUNT)
    {
      if (__count > (_LIBCUDACXX_POLLING_COUNT >> 1))
      {
        ::cuda::std::__cccl_thread_yield_processor();
      }
      __count += 1;
      continue;
    }
    ::cuda::std::chrono::high_resolution_clock::duration const __elapsed =
      ::cuda::std::chrono::high_resolution_clock::now() - __start;
    if (__max != ::cuda::std::chrono::nanoseconds::zero() && __max < __elapsed)
    {
      return false;
    }
    ::cuda::std::chrono::nanoseconds const __step = __elapsed / 4;
    if (__step >= ::cuda::std::chrono::milliseconds(1))
    {
      ::cuda::std::__cccl_thread_sleep_for(::cuda::std::chrono::milliseconds(1));
    }
    else if (__step >= ::cuda::std::chrono::microseconds(10))
    {
      ::cuda::std::__cccl_thread_sleep_for(__step);
    }
    else
    {
      ::cuda::std::__cccl_thread_yield();
    }
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___THREAD_POLL_H
