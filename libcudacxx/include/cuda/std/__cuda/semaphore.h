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

void simple_backoff()
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

template <thread_scope _Sco, ptrdiff_t __least_max_value = INT_MAX>
class counting_semaphore : public _CUDA_VSTD::__semaphore_base<__least_max_value, _Sco>
{
  static_assert(__least_max_value <= _CUDA_VSTD::__semaphore_base<__least_max_value, _Sco>::max(), "");

public:
  _LIBCUDACXX_INLINE_VISIBILITY constexpr counting_semaphore(ptrdiff_t __count = 0)
      : _CUDA_VSTD::__semaphore_base<__least_max_value, _Sco>(__count)
  {}
  ~counting_semaphore() = default;

  counting_semaphore(const counting_semaphore&)            = delete;
  counting_semaphore& operator=(const counting_semaphore&) = delete;
};

template <thread_scope _Sco, ptrdiff_t __least_max_value = INT_MAX>
struct fair_counting_semaphore
{
};

template <thread_scope _Sco>
struct binary_semaphore : fair_counting_semaphore<_Sco, 1>
{
public:
    _LIBCUDACXX_INLINE_VISIBILITY static constexpr ptrdiff_t max() noexcept
    {
        return static_cast<ptrdiff_t>(1);
    }

    _LIBCUDACXX_INLINE_VISIBILITY bool try_acquire()
    {
        ptrdiff_t ctr = counter.load(cuda::memory_order_relaxed);
        return try_acq_impl(ctr);
    }

    _LIBCUDACXX_INLINE_VISIBILITY void acquire()
    {
        ptrdiff_t ctr = counter.load(cuda::memory_order_relaxed);
        while (!try_acq_impl(ctr))
        {
            counter.wait(ctr);
        }
        int32_t cur_tickets = (ticketing.load(cuda::memory_order_relaxed) & 0xFFFFFFFF);
        auto limit = _CUDA_VSTD::numeric_limits<int32_t>::max();
        if (limit >= cur_tickets + 1)
        {
            auto prev = ticketing.fetch_add(1);
            while (!our_turn(prev + 1))
            {
                simple_backoff();
            }
        }
    }

    _LIBCUDACXX_INLINE_VISIBILITY bool __acquire_slow_timed(
        _CUDA_VSTD::chrono::nanoseconds const &__rel_time)
    {
        return __libcpp_thread_poll_with_backoff(
            [this]()
            {
                ptrdiff_t ctr = counter.load(cuda::memory_order_relaxed);
                while (!try_acq_impl(ctr))
                {
                    counter.wait(ctr);
                }
                int32_t cur_tickets = (ticketing.load(cuda::memory_order_relaxed) & 0xFFFFFFFF);
                auto limit = _CUDA_VSTD::numeric_limits<int32_t>::max();
                if (limit >= cur_tickets + 1)
                {
                    auto prev = ticketing.fetch_add(1);
                    while (!our_turn(prev + 1))
                    {
                        simple_backoff();
                    }
                    return true;
                }
                else
                {
                    return false;
                }
            },
            __rel_time);
    }

    _LIBCUDACXX_INLINE_VISIBILITY void release(ptrdiff_t update = 1)
    {
        counter.fetch_add(1);
        ticketing.fetch_add(static_cast<ptrdiff_t>(1) << 32, cuda::memory_order_relaxed);
        // we need to notify all because no guarantee of who
        // gets it
        counter.notify_all();
        ticketing.notify_all();
    }

    template <class Rep, class Period>
    _LIBCUDACXX_INLINE_VISIBILITY bool try_acquire_for(_CUDA_VSTD::chrono::duration<Rep, Period> const &__rel_time)
    {
        bool result = __acquire_slow_timed(__rel_time);
        if (result)
        {
            return true;
        }
        else
        {
            skipped_tickets.fetch_add(1, cuda::memory_order_relaxed);
            return false;
        }
    }

    template <class Clock, class Duration>
    _LIBCUDACXX_INLINE_VISIBILITY bool try_acquire_until(_CUDA_VSTD::chrono::time_point<Clock, Duration> const &__abs_time)
    {
        bool result = __acquire_slow_timed(__abs_time - Clock::now());
        if (result)
        {
            return true;
        }
        else
        {
            skipped_tickets.fetch_add(1, cuda::memory_order_relaxed);
            return false;
        }
    }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr explicit binary_semaphore(ptrdiff_t ctr = 1) : counter(ctr), ticketing(0)
    {
        assert(ctr == 1);
    }

    ~binary_semaphore() = default;
    binary_semaphore(const binary_semaphore &) = delete;
    binary_semaphore operator=(const binary_semaphore &) = delete;

private:
    cuda::atomic<ptrdiff_t, _Sco> counter;
    cuda::atomic<ptrdiff_t, _Sco> skipped_tickets;
    cuda::atomic<ptrdiff_t, _Sco> ticketing;

    _LIBCUDACXX_INLINE_VISIBILITY bool try_acq_impl(ptrdiff_t ctr)
    {

        if (counter.compare_exchange_weak(ctr,
                                          static_cast<ptrdiff_t>(ctr - 1),
                                          cuda::memory_order_acquire,
                                          cuda::memory_order_relaxed))
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    _LIBCUDACXX_INLINE_VISIBILITY bool our_turn(ptrdiff_t ticket_num)
    {
        auto tick = ticketing.load(cuda::memory_order_relaxed);
        int32_t serving_ticket = tick >> 32;
        auto skipped = skipped_tickets.load(cuda::memory_order_relaxed);
        return ((skipped + ticket_num) == serving_ticket);
    }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___CUDA_SEMAPHORE_H
