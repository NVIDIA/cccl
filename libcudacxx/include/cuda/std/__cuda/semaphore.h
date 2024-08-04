// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_SEMAPHORE_H
#define _LIBCUDACXX___CUDA_SEMAPHORE_H

#include <cuda/std/detail/__config>
#include <cuda/atomic>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA


template <thread_scope _Sco, ptrdiff_t __least_max_value = INT_MAX>
class __fair_semaphore
{
private:
  const ptrdiff_t max_;
  cuda::atomic<ptrdiff_t, _Sco> current;
  cuda::atomic<ptrdiff_t, _Sco> tickets;

public:

    _LIBCUDACXX_INLINE_VISIBILITY bool try_acquire()
  {
    ptrdiff_t available_tickets = max_ - tickets.load();
    return (available_tickets > 0) ? true : false;
  }

  _LIBCUDACXX_INLINE_VISIBILITY static constexpr ptrdiff_t max() noexcept
  {
    return _CUDA_VSTD::numeric_limits<ptrdiff_t>::max();
  }

  _LIBCUDACXX_INLINE_VISIBILITY void acquire()
  {
    while (!try_acquire())
    {
      int wait_iterations = 1;
      volatile int ctr = 0;
      const int max_iterations = 1024;

      for (int i = 0; i < wait_iterations; ++i)
      {
        ctr++;
      }
      wait_iterations = (wait_iterations * 2 >= max_iterations) ? max_iterations : wait_iterations * 2;
    }
    ptrdiff_t t = tickets.fetch_add(1, cuda::memory_order_relaxed);

  }

  _LIBCUDACXX_INLINE_VISIBILITY bool __acquire_slow_timed(
    _CUDA_VSTD::chrono::nanoseconds const& __rel_time) {
    return __libcpp_thread_poll_with_backoff(
      [this]() {
        ptrdiff_t const remaining = max_ - tickets.load(memory_order_acquire);
        return remaining > 0 && __fetch_add_if_slow(remaining);
      },
      __rel_time);
  }

  _LIBCUDACXX_INLINE_VISIBILITY void release(ptrdiff_t update = 1)
  {
    tickets.fetch_sub(update, memory_order_release);
    current.fetch_add(update, memory_order_relaxed);
    if (update > 1)
    {
      tickets.notify_all();
      current.notify_all();
    }
    else
    {
      tickets.notify_one();
      current.notify_one();
    }
  }

  _LIBCUDACXX_INLINE_VISIBILITY bool __fetch_add_if_slow(ptrdiff_t remaining)
  {
    while (remaining > 0)
    {
      if (tickets.compare_exchange_weak(remaining, remaining + 1,
        memory_order_acquire,
        memory_order_relaxed))
      {
        return true;
      }
    }
    return false;
  }

  template <class Rep, class Period>
  _LIBCUDACXX_INLINE_VISIBILITY bool try_acquire_for(_CUDA_VSTD::chrono::duration<Rep, Period> const& __rel_time)
  {
    if (try_acquire())
    {
      return true;
    }
    else
    {
      return __acquire_slow_timed(__rel_time);
    }
  }

  template <class Clock, class Duration>
  _LIBCUDACXX_INLINE_VISIBILITY bool try_acquire_until(_CUDA_VSTD::chrono::time_point<Clock, Duration> const& __abs_time)
  {
    if (try_acquire())
    {
      return true;
    }
    else
    {
      return __acquire_slow_timed(__abs_time - Clock::now());
    }
  }

  _LIBCUDACXX_INLINE_VISIBILITY constexpr __fair_semaphore(ptrdiff_t count = 0) : tickets(0),
                                                                        current(0),
                                                                        max_(count)
  {
    assert(count > 0); // per https://en.cppreference.com/w/cpp/thread/counting_semaphore/counting_semaphore
  }

  ~__fair_semaphore() = default;
  __fair_semaphore(const __fair_semaphore &) = delete;
  __fair_semaphore operator=(const __fair_semaphore &) = delete;

};

template <thread_scope _Sco, ptrdiff_t Count = INT_MAX>
using counting_semaphore = __fair_semaphore<_Sco, Count>;

template <thread_scope _Sco>
using binary_semaphore = __fair_semaphore<_Sco, 1>;

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___CUDA_SEMAPHORE_H
