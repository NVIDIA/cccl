//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: pre-sm-70

// <mutex>

// struct once_flag;

// template<class Callable, class ...Args>
//   void call_once(once_flag& flag, Callable&& func, Args&&... args);

#include<cuda/mutex>
#include<cuda/std/cassert>
#include<cuda/std/chrono>

#include "test_macros.h"

#include "cuda_space_selector.h"
#include "heterogeneous_thread_handler.h"

typedef cuda::std::chrono::milliseconds ms;

__host__ __device__
void init0(int& init0_called)
{
    test_sleep_thread(ms(250));
    ++init0_called;
}

template<class OnceFlag>
__host__ __device__
void f0(OnceFlag* flg0, int* init0_called)
{
    cuda::std::call_once(*flg0, init0, *init0_called);
}

#ifndef TEST_HAS_NO_EXCEPTIONS
__host__ __device__
void init3(int& init3_called, int& init3_completed)
{
    ++init3_called;
    test_sleep_thread(ms(250));
    if (init3_called == 1)
#ifdef __CUDA_ARCH__
        _LIBCUDACXX_UNREACHABLE();
#else
        TEST_THROW(1);
#endif
    ++init3_completed;
}

template<class OnceFlag>
__host__ __device__
void f3(OnceFlag* flg3, int* init3_called, int* init3_completed)
{
    try
    {
        cuda::std::call_once(*flg3, init3, *init3_called, *init3_completed);
    }
    catch (...)
    {
    }
}
#endif // TEST_HAS_NO_EXCEPTIONS

#if TEST_STD_VER >= 11
struct init1
{
    int& init1_called;
    __host__ __device__ void operator()(int i) { init1_called += i; }
};

template<class OnceFlag>
__host__ __device__
void f1(OnceFlag* flg1, int* init1_called)
{
    cuda::std::call_once(*flg1, init1{*init1_called}, 1);
}

struct init2
{
    int& init2_called;
    __host__ __device__ void operator()(int i, int j) const {init2_called += i + j;}
};

template<class OnceFlag>
__host__ __device__
void f2(OnceFlag* flg2, int* init2_called)
{
    cuda::std::call_once(*flg2, init2{*init2_called}, 2, 3);
    cuda::std::call_once(*flg2, init2{*init2_called}, 4, 5);
}

#endif // TEST_STD_VER >= 11
__host__ __device__
void init41(int& init41_called)
{
    test_sleep_thread(ms(250));
    ++init41_called;
}

__host__ __device__
void init42(int& init42_called)
{
    test_sleep_thread(ms(250));
    ++init42_called;
}

template<class OnceFlag>
__host__ __device__
void f41(OnceFlag* flg41, OnceFlag* flg42, int* init41_called, int* init42_called)
{
    cuda::std::call_once(*flg41, init41, *init41_called);
    cuda::std::call_once(*flg42, init42, *init42_called);
}

template<class OnceFlag>
__host__ __device__
void f42(OnceFlag* flg41, OnceFlag* flg42, int* init41_called, int* init42_called)
{
    cuda::std::call_once(*flg41, init42, *init42_called);
    cuda::std::call_once(*flg42, init41, *init41_called);
}

#if TEST_STD_VER >= 11

class MoveOnly
{
    __host__ __device__ MoveOnly(const MoveOnly&);
public:
    __host__ __device__ MoveOnly() {}
    __host__ __device__ MoveOnly(MoveOnly&&) {}

    __host__ __device__ void operator()(MoveOnly&&) {}
};

class NonCopyable
{
    __host__ __device__ NonCopyable(const NonCopyable&);
public:
    __host__ __device__ NonCopyable() {}

    __host__ __device__ void operator()(int&) {}
};

// reference qualifiers on functions are a C++11 extension
struct RefQual
{
    __host__ __device__ void operator()( int& lv_called) & { ++lv_called; }
    __host__ __device__ void operator()( int& rv_called) && { ++rv_called; }
};

#endif // TEST_STD_VER >= 11

template<class OnceFlag,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__host__ __device__ void test() {
    Selector<OnceFlag, Initializer> sel;
    Selector<int, Initializer> sel_count;
    heterogeneous_thread_handler handler;

    // check basic functionality
    {
        SHARED OnceFlag* flg0;
        flg0 = sel.construct();

        SHARED int* init0_called;
        init0_called = sel_count.construct();

        handler.run_on_first_thread(f0<OnceFlag>, flg0, init0_called);
        handler.run_on_second_thread(f0<OnceFlag>, flg0, init0_called);
        handler.join_test_thread();
        handler.syncthreads();
        assert(*init0_called == 1);
    }

#ifndef TEST_HAS_NO_EXCEPTIONS
    // check basic exception safety
    {
        SHARED OnceFlag* flg0;
        flg0 = sel.construct();

        SHARED int* init3_called;
        init3_called = sel_count.construct();
        SHARED int* init3_completed;
        init3_completed = sel_count.construct();

        handler.run_on_first_thread( f3<OnceFlag>, flg3, init3_called, init3_completed);
        handler.run_on_second_thread(f3<OnceFlag>, flg3, init3_called, init3_completed);
        handler.join_test_thread();
        handler.syncthreads();

        assert(*init3_called == 2);
        assert(*init3_completed == 1);
    }
#endif
    // check deadlock avoidance
    {
        SHARED OnceFlag* flg41;
        flg41 = sel.construct();
        SHARED OnceFlag* flg42;
        flg42 = sel.construct();

        SHARED int* init41_called;
        init41_called = sel_count.construct();
        SHARED int* init42_called;
        init42_called = sel_count.construct();

        handler.run_on_first_thread( f41<OnceFlag>, flg41, flg42, init41_called, init42_called);
        handler.run_on_second_thread(f42<OnceFlag>, flg41, flg42, init41_called, init42_called);
        handler.join_test_thread();
        handler.syncthreads();

        assert(*init41_called == 1);
        assert(*init42_called == 1);
    }
#if TEST_STD_VER >= 11
    // check functors with 1 arg
    {
        SHARED OnceFlag* flg1;
        flg1 = sel.construct();

        SHARED int* init1_called;
        init1_called = sel_count.construct();

        handler.run_on_first_thread( f1<OnceFlag>, flg1, init1_called);
        handler.run_on_second_thread(f1<OnceFlag>, flg1, init1_called);
        handler.join_test_thread();
        handler.syncthreads();
        assert(*init1_called == 1);
    }
    // check functors with 2 args
    {
        SHARED OnceFlag* flg2;
        flg2 = sel.construct();

        SHARED int* init2_called;
        init2_called = sel_count.construct();

        handler.run_on_first_thread( f2<OnceFlag>, flg2, init2_called);
        handler.run_on_second_thread(f2<OnceFlag>, flg2, init2_called);
        handler.join_test_thread();
        handler.syncthreads();
        assert(*init2_called == 5);
    }

    {
        SHARED OnceFlag* f;
        f = sel.construct();

        cuda::call_once(*f, MoveOnly(), MoveOnly());
    }
    // check LWG2442: call_once() shouldn't DECAY_COPY()
    {
        SHARED OnceFlag* f;
        f = sel.construct();
        int i = 0;

        cuda::call_once(*f, NonCopyable(), i);
    }
    // reference qualifiers on functions are a C++11 extension
    {
        SHARED OnceFlag* f1, *f2;
        f1 = sel.construct();
        f2 = sel.construct();

        SHARED int* lv_called;
        lv_called = sel_count.construct();

        SHARED int* rv_called;
        rv_called = sel_count.construct();

        RefQual rq;
        cuda::call_once(*f1, rq, *lv_called);
        assert(*lv_called == 1);
        cuda::call_once(*f2, cuda::std::move(rq), *rv_called);
        assert(*rv_called == 1);
    }
#endif // TEST_STD_VER >= 11
}

int main(int, char**)
{
#ifndef __CUDA_ARCH__
    cuda_thread_count = 2;

    test<cuda::once_flag<cuda::thread_scope_system>, local_memory_selector>();
    test<cuda::once_flag<cuda::thread_scope_device>, local_memory_selector>();
    test<cuda::once_flag<cuda::thread_scope_block>,  local_memory_selector>();
    test<cuda::once_flag<cuda::thread_scope_thread>, local_memory_selector>();
#else
    test<cuda::once_flag<cuda::thread_scope_system>, shared_memory_selector>();
    test<cuda::once_flag<cuda::thread_scope_device>, shared_memory_selector>();
    test<cuda::once_flag<cuda::thread_scope_block>,  shared_memory_selector>();
    test<cuda::once_flag<cuda::thread_scope_thread>, shared_memory_selector>();

    test<cuda::once_flag<cuda::thread_scope_system>, global_memory_selector>();
    test<cuda::once_flag<cuda::thread_scope_device>, global_memory_selector>();
    test<cuda::once_flag<cuda::thread_scope_block>,  global_memory_selector>();
    test<cuda::once_flag<cuda::thread_scope_thread>, global_memory_selector>();
#endif

    return 0;
}
