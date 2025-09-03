// -*- C++ -*-
//===--------------------------- latch -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___LATCH_LATCH_H
#define _CUDA_STD___LATCH_LATCH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/atomic>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <thread_scope _Sco = thread_scope_system>
class __latch_base
{
  __atomic_impl<ptrdiff_t, _Sco> __counter;

public:
  _CCCL_API constexpr explicit __latch_base(ptrdiff_t __expected)
      : __counter(__expected)
  {}

  _CCCL_HIDE_FROM_ABI ~__latch_base()          = default;
  __latch_base(const __latch_base&)            = delete;
  __latch_base& operator=(const __latch_base&) = delete;

  _CCCL_API inline void count_down(ptrdiff_t __update = 1)
  {
    _CCCL_ASSERT(__update > 0, "");
    auto const __old = __counter.fetch_sub(__update, memory_order_release);
    _CCCL_ASSERT(__old >= __update, "");
    if (__old == __update)
    {
      __counter.notify_all();
    }
  }
  _CCCL_API inline bool try_wait() const noexcept
  {
    return __counter.load(memory_order_acquire) == 0;
  }
  _CCCL_API inline void wait() const
  {
    while (1)
    {
      auto const __current = __counter.load(memory_order_acquire);
      if (__current == 0)
      {
        return;
      }
      __counter.wait(__current, memory_order_relaxed);
    }
  }
  _CCCL_API inline void arrive_and_wait(ptrdiff_t __update = 1)
  {
    count_down(__update);
    wait();
  }

  _CCCL_API static constexpr ptrdiff_t max() noexcept
  {
    return numeric_limits<ptrdiff_t>::max();
  }
};

using latch = __latch_base<>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif //_CUDA_STD___LATCH_LATCH_H
