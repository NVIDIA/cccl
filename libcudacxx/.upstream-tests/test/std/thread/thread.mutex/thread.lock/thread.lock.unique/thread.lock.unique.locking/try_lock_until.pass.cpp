//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: pre-sm-70

// <mutex>

// template <class Mutex> class unique_lock;

// template <class Clock, class Duration>
//   bool try_lock_until(const chrono::time_point<Clock, Duration>& abs_time);

#include<cuda/std/mutex>
#include<cuda/std/cassert>

#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR bool try_lock_until_called = false;

struct mutex
{
    template <class Clock, class Duration>
    __host__ __device__ bool try_lock_until(const cuda::std::chrono::time_point<Clock, Duration>& abs_time)
    {
        typedef cuda::std::chrono::milliseconds ms;
        assert(Clock::now() - abs_time < ms(5));
        try_lock_until_called = !try_lock_until_called;
        return try_lock_until_called;
    }
    __host__ __device__ void unlock() {}
};

STATIC_TEST_GLOBAL_VAR mutex m;

#if defined(_LIBCUDACXX_HAS_NO_MONOTONIC_CLOCK)
typedef cuda::std::chrono::system_clock Clock;
#else
typedef cuda::std::chrono::steady_clock Clock;
#endif

int main(int, char**)
{
    cuda::std::unique_lock<mutex> lk(m, cuda::std::defer_lock);
    assert(lk.try_lock_until(Clock::now()) == true);
    assert(try_lock_until_called == true);
    assert(lk.owns_lock() == true);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        TEST_IGNORE_NODISCARD lk.try_lock_until(Clock::now());
        assert(false);
    }
    catch (cuda::std::system_error& e)
    {
        assert(e.code().value() == EDEADLK);
    }
#endif
    lk.unlock();
    assert(lk.try_lock_until(Clock::now()) == false);
    assert(try_lock_until_called == false);
    assert(lk.owns_lock() == false);
    lk.release();
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        TEST_IGNORE_NODISCARD lk.try_lock_until(Clock::now());
        assert(false);
    }
    catch (cuda::std::system_error& e)
    {
        assert(e.code().value() == EPERM);
    }
#endif

  return 0;
}
