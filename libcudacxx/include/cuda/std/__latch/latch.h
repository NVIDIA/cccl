// -*- C++ -*-
//===--------------------------- latch -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___LATCH_LATCH_H
#define _LIBCUDACXX___LATCH_LATCH_H

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

_CCCL_PUSH_MACROS

#ifdef _LIBCUDACXX_HAS_NO_THREADS
#  error <latch> is not supported on this single threaded system
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_CUDA_ABI_VERSION < 3
#  define _LIBCUDACXX_LATCH_ALIGNMENT alignas(64)
#else
#  define _LIBCUDACXX_LATCH_ALIGNMENT
#endif

template <thread_scope _Sco = thread_scope_system>
class __latch_base
{
  _LIBCUDACXX_LATCH_ALIGNMENT __atomic_impl<ptrdiff_t, _Sco> __counter;

public:
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __latch_base(ptrdiff_t __expected)
      : __counter(__expected)
  {}

  _CCCL_HIDE_FROM_ABI ~__latch_base()          = default;
  __latch_base(const __latch_base&)            = delete;
  __latch_base& operator=(const __latch_base&) = delete;

  _LIBCUDACXX_HIDE_FROM_ABI void count_down(ptrdiff_t __update = 1)
  {
    _CCCL_ASSERT(__update > 0, "");
    auto const __old = __counter.fetch_sub(__update, memory_order_release);
    _CCCL_ASSERT(__old >= __update, "");
    if (__old == __update)
    {
      __counter.notify_all();
    }
  }
  _LIBCUDACXX_HIDE_FROM_ABI bool try_wait() const noexcept
  {
    return __counter.load(memory_order_acquire) == 0;
  }
  _LIBCUDACXX_HIDE_FROM_ABI void wait() const
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
  _LIBCUDACXX_HIDE_FROM_ABI void arrive_and_wait(ptrdiff_t __update = 1)
  {
    count_down(__update);
    wait();
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t max() noexcept
  {
    return numeric_limits<ptrdiff_t>::max();
  }
};

using latch = __latch_base<>;

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_POP_MACROS

#endif //_LIBCUDACXX___LATCH_LATCH_H
