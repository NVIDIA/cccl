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

// template <class Mutex> class lock_guard;

// lock_guard(mutex_type& m, adopt_lock_t);
#include<cuda/std/mutex>
#include<cuda/std/cassert>
#include<cuda/std/chrono>

#include "test_macros.h"

#include "cuda_space_selector.h"
#include "heterogeneous_thread_handler.h"

typedef cuda::std::chrono::system_clock Clock;
typedef Clock::time_point time_point;
typedef Clock::duration duration;
typedef cuda::std::chrono::milliseconds ms;
typedef cuda::std::chrono::nanoseconds ns;

#ifndef __CUDA_ARCH__
template<class Mutex>
__host__ void do_try_lock(Mutex* m) {
  assert(m->try_lock() == false);
}

template<class Mutex,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__host__ void test_host() {
    Selector<Mutex, Initializer> sel;
    SHARED Mutex* m;
    m = sel.construct();

    {
      m->lock();
      cuda::std::lock_guard<Mutex> lg(*m, cuda::std::adopt_lock);
      std::thread t(do_try_lock<Mutex>, m);
      t.join();
    }

    m->lock();
    m->unlock();
}
#endif

template<class Mutex,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__device__ void test_device() {
    Selector<Mutex, Initializer> sel;
    SHARED Mutex* m;
    m = sel.construct();

    Selector<cuda::std::atomic_flag, Initializer> sel_flag;
    SHARED cuda::std::atomic_flag* thread_ready;
    thread_ready = sel_flag.construct(false);

    if (threadIdx.x == 0) {
      m->lock();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      cuda::std::lock_guard<Mutex> lg(*m, cuda::std::adopt_lock);
      thread_ready->test_and_set(cuda::std::memory_order_relaxed);
      thread_ready->notify_one();
      thread_ready->wait(true, cuda::std::memory_order_relaxed);  // Waits until clear
    } else {
      thread_ready->wait(false, cuda::std::memory_order_relaxed); // Waits until test_and_set
      assert(m->try_lock() == false);
      thread_ready->clear(cuda::std::memory_order_relaxed);
      thread_ready->notify_one();
    }
    __syncthreads();

    m->lock();
    m->unlock();
}

int main(int, char**)
{
#ifndef __CUDA_ARCH__
    cuda_thread_count = 2;

    test_host<cuda::std::mutex, local_memory_selector>();
#else
    test_device<cuda::std::mutex, shared_memory_selector>();
    test_device<cuda::std::mutex, global_memory_selector>();
#endif

  return 0;
}
