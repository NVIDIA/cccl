//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___SEMAPHORE_ATOMIC_SEMAPHORE_H
#define _LIBCUDACXX___SEMAPHORE_ATOMIC_SEMAPHORE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/atomic>
#include <cuda/std/chrono>
#include <cuda/std/cstdint>

_CCCL_PUSH_MACROS

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <thread_scope _Sco, ptrdiff_t __least_max_value>
class __atomic_semaphore
{
  __atomic_impl<ptrdiff_t, _Sco> __count;

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __fetch_sub_if_slow(ptrdiff_t __old)
  {
    while (__old != 0)
    {
      if (__count.compare_exchange_weak(__old, __old - 1, memory_order_acquire, memory_order_relaxed))
      {
        return true;
      }
    }
    return false;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __fetch_sub_if()
  {
    ptrdiff_t __old = __count.load(memory_order_acquire);
    if (__old == 0)
    {
      return false;
    }
    if (__count.compare_exchange_weak(__old, __old - 1, memory_order_acquire, memory_order_relaxed))
    {
      return true;
    }
    return __fetch_sub_if_slow(__old); // fail only if not __available
  }

  _LIBCUDACXX_HIDE_FROM_ABI void __wait_slow()
  {
    while (1)
    {
      ptrdiff_t const __old = __count.load(memory_order_acquire);
      if (__old != 0)
      {
        break;
      }
      __count.wait(__old, memory_order_relaxed);
    }
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __acquire_slow_timed(chrono::nanoseconds const& __rel_time)
  {
    return _CUDA_VSTD::__cccl_thread_poll_with_backoff(
      [this]() {
        ptrdiff_t const __old = __count.load(memory_order_acquire);
        return __old != 0 && __fetch_sub_if_slow(__old);
      },
      __rel_time);
  }

public:
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t max() noexcept
  {
    return numeric_limits<ptrdiff_t>::max();
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __atomic_semaphore(ptrdiff_t __count) noexcept
      : __count(__count)
  {}

  _CCCL_HIDE_FROM_ABI ~__atomic_semaphore() = default;

  __atomic_semaphore(__atomic_semaphore const&)            = delete;
  __atomic_semaphore& operator=(__atomic_semaphore const&) = delete;

  _LIBCUDACXX_HIDE_FROM_ABI void release(ptrdiff_t __update = 1)
  {
    __count.fetch_add(__update, memory_order_release);
    if (__update > 1)
    {
      __count.notify_all();
    }
    else
    {
      __count.notify_one();
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI void acquire()
  {
    while (!try_acquire())
    {
      __wait_slow();
    }
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool try_acquire() noexcept
  {
    return __fetch_sub_if();
  }

  template <class Clock, class Duration>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool try_acquire_until(chrono::time_point<Clock, Duration> const& __abs_time)
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

  template <class Rep, class Period>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool try_acquire_for(chrono::duration<Rep, Period> const& __rel_time)
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
};

template <thread_scope _Sco>
class __atomic_semaphore<_Sco, 1>
{
  __atomic_impl<int, _Sco> __available;

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __acquire_slow_timed(chrono::nanoseconds const& __rel_time)
  {
    return _CUDA_VSTD::__cccl_thread_poll_with_backoff(
      [this]() {
        return try_acquire();
      },
      __rel_time);
  }

public:
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t max() noexcept
  {
    return 1;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __atomic_semaphore(ptrdiff_t __available)
      : __available(__available)
  {}

  _CCCL_HIDE_FROM_ABI ~__atomic_semaphore() = default;

  __atomic_semaphore(__atomic_semaphore const&)            = delete;
  __atomic_semaphore& operator=(__atomic_semaphore const&) = delete;

  _LIBCUDACXX_HIDE_FROM_ABI void release(ptrdiff_t __update = 1)
  {
    _CCCL_ASSERT(__update == 1, "");
    __available.store(1, memory_order_release);
    __available.notify_one();
    (void) __update;
  }

  _LIBCUDACXX_HIDE_FROM_ABI void acquire()
  {
    while (!try_acquire())
    {
      __available.wait(0, memory_order_relaxed);
    }
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool try_acquire() noexcept
  {
    return 1 == __available.exchange(0, memory_order_acquire);
  }

  template <class Clock, class Duration>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool try_acquire_until(chrono::time_point<Clock, Duration> const& __abs_time)
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

  template <class Rep, class Period>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool try_acquire_for(chrono::duration<Rep, Period> const& __rel_time)
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
};

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_POP_MACROS

#endif // _LIBCUDACXX___SEMAPHORE_ATOMIC_SEMAPHORE_H
