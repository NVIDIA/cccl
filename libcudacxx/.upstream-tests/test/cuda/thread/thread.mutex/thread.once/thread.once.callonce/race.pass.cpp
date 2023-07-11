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

// This test is supposed to be run with ThreadSanitizer and verifies that
// call_once properly synchronizes user state, a data race that was fixed
// in r280621.

#include<cuda/mutex>
#include<cuda/std/cassert>
#include<cuda/std/chrono>

#include "test_macros.h"

#include "cuda_space_selector.h"
#include "heterogeneous_thread_handler.h"

typedef cuda::std::chrono::milliseconds ms;

__host__ __device__
void init0(int& global)
{
    ++global;
}

template<class OnceFlag>
__host__ __device__
void f0(OnceFlag* flg, int* global)
{
    cuda::call_once(*flg, init0, *global);
    assert(*global == 1);
}

template<class OnceFlag,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__host__ __device__ void test() {
    Selector<OnceFlag, Initializer> sel;
    SHARED OnceFlag* flg;
    flg = sel.construct();

    Selector<int, Initializer> sel_count;
    SHARED int* global;
    global = sel_count.construct();

    heterogeneous_thread_handler handler;
    handler.run_on_first_thread(f0<OnceFlag>, flg, global);
    handler.run_on_second_thread(f0<OnceFlag>, flg, global);
    handler.syncthreads();
    handler.join_test_thread();
    assert(*global == 1);
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
