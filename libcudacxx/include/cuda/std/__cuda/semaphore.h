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

void simple_backoff() {
  int wait_iterations = 1;
  volatile int ctr = 0;
  const int max_iterations = 1024;

  for (int i = 0; i < wait_iterations; ++i)
  {
    ctr++;
  }
  wait_iterations = (wait_iterations * 2 >= max_iterations) ? max_iterations : wait_iterations * 2;
}


template <thread_scope _Sco, ptrdiff_t __least_max_value = INT_MAX>
class __fair_semaphore
{
private:
  const ptrdiff_t max_;
  cuda::atomic<ptrdiff_t, _Sco> current;
  cuda::atomic<ptrdiff_t, _Sco> tickets;

public:
    static_assert(__least_max_value >= 0, "The least maximum value must be a positive number");

    _LIBCUDACXX_INLINE_VISIBILITY bool try_acquire()
  {
    ptrdiff_t available_tickets = max_ - tickets.load();
    return (available_tickets > 0) ? true : false;
  }
  
  _LIBCUDACXX_INLINE_VISIBILITY bool our_turn(ptrdiff_t ticket_num)
  {
    ptrdiff_t serving_ticket = current.load();
    return ticket_num == serving_ticket;
  }

  _LIBCUDACXX_INLINE_VISIBILITY static constexpr ptrdiff_t max() noexcept
  {
    return _CUDA_VSTD::numeric_limits<ptrdiff_t>::max();
  }

  // should block until it's their turn. Right? 
  _LIBCUDACXX_INLINE_VISIBILITY void acquire()
  { 
    // if there aren't any tickets available we block
    while(!try_acquire()) {
      simple_backoff();
    }
    ptrdiff_t t = tickets.fetch_add(1, cuda::memory_order_relaxed);
    // now we're in line
    // give the newly added value
    while (!our_turn(t+1)) 
    {
      simple_backoff();
    }
    // blocking has concluded at this point
  }

  _LIBCUDACXX_INLINE_VISIBILITY bool __acquire_slow_timed(
    _CUDA_VSTD::chrono::nanoseconds const& __rel_time) {

    return __libcpp_thread_poll_with_backoff(
      [this]() {
        ptrdiff_t const remaining = max_ - tickets.load(memory_order_acquire);
        bool success = false;
        // potential to refine this so that time between loading
        // tickets and time between return doesn't allow possibility of
        // another thread to alter remaining?
        while(remaining > 0 && !success) {
          success = tickets.compare_exchange_weak(remaining, 
            remaining + 1,
            memory_order_acquire,
            memory_order_relaxed);
        }
        // keep looping until we finally succeed. Now that it's been added
        // we need to make sure it's our turn
        while(!our_turn(remaining+1)) {
          simple_backoff();
        }
        return true;

        // I don't think we should do the below code because it'll terminate 
        // potentially early 
        // return our_turn(remaining+1);
      },
      __rel_time);
  }

  _LIBCUDACXX_INLINE_VISIBILITY void release(ptrdiff_t update = 1)
  {
    tickets.fetch_sub(update, memory_order_release);
    current.fetch_add(update, memory_order_relaxed);
    if (update > 1)
    {
      // if the update is one, we update one at a time
      // do we need to use CG or something to emulate conditions for waking up 
      // threads? 
      for(int i = 0; i<update; i++) {
        tickets.notify_one();
        current.notify_one();
      }
    }
    else
    {
      tickets.notify_one();
      current.notify_one();
    }
  }

  template <class Rep, class Period>
  _LIBCUDACXX_INLINE_VISIBILITY bool try_acquire_for(_CUDA_VSTD::chrono::duration<Rep, Period> const& __rel_time)
  {
    return __acquire_slow_timed(__rel_time); 
  }

  template <class Clock, class Duration>
  _LIBCUDACXX_INLINE_VISIBILITY bool try_acquire_until(_CUDA_VSTD::chrono::time_point<Clock, Duration> const& __abs_time)
  {
    return __acquire_slow_timed(__abs_time - Clock::now());  
  }

  _LIBCUDACXX_INLINE_VISIBILITY constexpr explicit __fair_semaphore(ptrdiff_t count = 0) : tickets(0),
                                                                        current(0),
                                                                        max_(count)
  {
    assert(count >= 0); // per https://en.cppreference.com/w/cpp/thread/counting_semaphore/counting_semaphore
    // i noticed it's not in the standard to have a default value, imo should be removed
    assert(count <= max());
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
