//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MUTEX_MUTEX_H
#define _CUDA_STD___MUTEX_MUTEX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__thread/threading_support.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <thread_scope _Sco = thread_scope::thread_scope_device>
struct mutex
{
  _CCCL_HIDE_FROM_ABI constexpr mutex() noexcept = default;

  mutex(const mutex&)            = delete;
  mutex& operator=(const mutex&) = delete;

  static constexpr thread_scope _Scope = _Sco;

  _CCCL_HOST_DEVICE_API void lock()
  {
    ::cuda::std::__cccl_mutex_lock<_Sco>(&__mut);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_lock() noexcept
  {
    return ::cuda::std::__cccl_mutex_trylock<_Sco>(&__mut);
  }

  _CCCL_HOST_DEVICE_API void unlock() noexcept
  {
    ::cuda::std::__cccl_mutex_unlock<_Sco>(&__mut);
  }

private:
  ::cuda::std::__cccl_mutex_t __mut{};
};

_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using mutex = ::cuda::mutex<thread_scope::thread_scope_device>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MUTEX_MUTEX_H
