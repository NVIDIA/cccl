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
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: pre-sm-70

// <mutex>

// template <class ...Mutex> class scoped_lock;

// explicit scoped_lock(mutex_type& m);

#include<cuda/std/mutex>
#include<cuda/std/cassert>
#include "test_macros.h"

struct TestMutex {
    bool locked = false;
    TestMutex() = default;
    __host__ __device__ ~TestMutex() { assert(!locked); }

    __host__ __device__ void lock() { assert(!locked); locked = true; }
    __host__ __device__ bool try_lock() { if (locked) return false; locked = true; return true; }
    __host__ __device__ void unlock() { assert(locked); locked = false; }

    TestMutex(TestMutex const&) = delete;
    TestMutex& operator=(TestMutex const&) = delete;
};

#if !defined(TEST_HAS_NO_EXCEPTIONS)
struct TestMutexThrows {
    bool locked = false;
    bool throws_on_lock = false;

    __host__ __device__ TestMutexThrows() = default;
    __host__ __device__ ~TestMutexThrows() { assert(!locked); }

    __host__ __device__ void lock() {
        assert(!locked);
        if (throws_on_lock) {
            throw 42;
        }
        locked = true;
    }

    __host__ __device__ bool try_lock() {
        if (locked) return false;
        lock();
        return true;
    }

    __host__ __device__ void unlock() { assert(locked); locked = false; }

    TestMutexThrows(TestMutexThrows const&) = delete;
    TestMutexThrows& operator=(TestMutexThrows const&) = delete;
};
#endif // !defined(TEST_HAS_NO_EXCEPTIONS)

int main(int, char**)
{
    {
        using LG = cuda::std::scoped_lock<>;
        LG lg;
        unused(lg);
    }
    {
        using LG = cuda::std::scoped_lock<TestMutex>;
        TestMutex m1;
        {
            LG lg(m1);
            assert(m1.locked);
        }
        assert(!m1.locked);
    }
    {
        using LG = cuda::std::scoped_lock<TestMutex, TestMutex>;
        TestMutex m1, m2;
        {
            LG lg(m1, m2);
            assert(m1.locked && m2.locked);
        }
        assert(!m1.locked && !m2.locked);
    }
    {
        using LG = cuda::std::scoped_lock<TestMutex, TestMutex, TestMutex>;
        TestMutex m1, m2, m3;
        {
            LG lg(m1, m2, m3);
            assert(m1.locked && m2.locked && m3.locked);
        }
        assert(!m1.locked && !m2.locked && !m3.locked);
    }
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    {
        using MT = TestMutexThrows;
        using LG = cuda::std::scoped_lock<MT>;
        MT m1;
        m1.throws_on_lock = true;
        try {
            LG lg(m1);
            assert(false);
        } catch (int) {}
        assert(!m1.locked);
    }
    {
        using MT = TestMutexThrows;
        using LG = cuda::std::scoped_lock<MT, MT>;
        MT m1, m2;
        m1.throws_on_lock = true;
        try {
            LG lg(m1, m2);
            assert(false);
        } catch (int) {}
        assert(!m1.locked && !m2.locked);
    }
    {
        using MT = TestMutexThrows;
        using LG = cuda::std::scoped_lock<MT, MT, MT>;
        MT m1, m2, m3;
        m2.throws_on_lock = true;
        try {
            LG lg(m1, m2, m3);
            assert(false);
        } catch (int) {}
        assert(!m1.locked && !m2.locked && !m3.locked);
    }
#endif

#if TEST_STD_VER >= 17
    {
    TestMutex m1, m2, m3;
        {
        cuda::std::scoped_lock sl{};
        static_assert((cuda::std::is_same<decltype(sl), cuda::std::scoped_lock<>>::value), "" );
        unused(sl);
        }
        {
        cuda::std::scoped_lock sl{m1};
        static_assert((cuda::std::is_same<decltype(sl), cuda::std::scoped_lock<decltype(m1)>>::value), "" );
        unused(sl);
        }
        {
        cuda::std::scoped_lock sl{m1, m2};
        static_assert((cuda::std::is_same<decltype(sl), cuda::std::scoped_lock<decltype(m1), decltype(m2)>>::value), "" );
        unused(sl);
        }
        {
        cuda::std::scoped_lock sl{m1, m2, m3};
        static_assert((cuda::std::is_same<decltype(sl), cuda::std::scoped_lock<decltype(m1), decltype(m2), decltype(m3)>>::value), "" );
        unused(sl);
        }
    }
#endif

  return 0;
}
