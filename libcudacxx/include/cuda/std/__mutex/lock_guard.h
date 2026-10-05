//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MUTEX_LOCK_GUARD_H
#define _CUDA_STD___MUTEX_LOCK_GUARD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__mutex/tag_types.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Mutex>
class lock_guard
{
public:
  using mutex_type = _Mutex;

  // Owns the lock for the lifetime of the guard. Discarding the temporary would unlock immediately.
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit lock_guard(mutex_type& __mut)
      : __mut(__mut)
  {
    __mut.lock();
  }

  // __mut is already locked by this thread. The guard unlocks it on destruction.
  [[nodiscard]] _CCCL_HOST_DEVICE_API lock_guard(mutex_type& __mut, adopt_lock_t)
      : __mut(__mut)
  {}

  _CCCL_HOST_DEVICE_API ~lock_guard()
  {
    __mut.unlock();
  }

  lock_guard(const lock_guard&)            = delete;
  lock_guard& operator=(const lock_guard&) = delete;

private:
  mutex_type& __mut;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MUTEX_LOCK_GUARD_H
