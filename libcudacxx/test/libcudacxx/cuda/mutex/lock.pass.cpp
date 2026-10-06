//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/mutex>

// template <thread_scope Scope = thread_scope_device>
// struct cuda::mutex;
// using cuda::std::mutex = cuda::mutex<thread_scope_device>;

// UNSUPPORTED: libcpp-has-no-threads
// The host side launches through std::thread.

// UNSUPPORTED: pre-sm-70
// Device atomics used by mutex require sm_70.

// UNSUPPORTED: force-tile
// error: asm statement is not supported

#include <cuda/std/cassert>
#include <cuda/std/mutex>
#include <cuda/std/type_traits>

#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

static_assert(cuda::std::is_same_v<cuda::mutex<>, cuda::mutex<cuda::thread_scope_device>>, "");
static_assert(cuda::std::is_same_v<cuda::mutex<>, cuda::std::mutex>, "");
static_assert(cuda::mutex<cuda::thread_scope_thread>::_Scope == cuda::thread_scope_thread, "");
static_assert(cuda::mutex<cuda::thread_scope_block>::_Scope == cuda::thread_scope_block, "");
static_assert(cuda::mutex<cuda::thread_scope_cluster>::_Scope == cuda::thread_scope_cluster, "");
static_assert(cuda::mutex<cuda::thread_scope_device>::_Scope == cuda::thread_scope_device, "");
static_assert(cuda::mutex<cuda::thread_scope_system>::_Scope == cuda::thread_scope_system, "");

template <class Mutex>
struct exclusive_section
{
  Mutex* mutex;
  int* counter;

  TEST_HOST_DEVICE_FUNC void operator()()
  {
    mutex->lock();
    ++*counter;
    assert(*counter == 1);
    --*counter;
    mutex->unlock();
  }
};

template <typename Mutex, template <typename, typename> class Selector, typename Initializer = default_initializer>
TEST_HOST_DEVICE_FUNC void test()
{
  Selector<Mutex, Initializer> sel;
  SHARED Mutex* mutex;
  mutex = sel.construct();

  SHARED int counter_storage;
  int* counter = &counter_storage;
  execute_on_main_thread([&] {
    *counter = 0;
    {
      cuda::std::lock_guard<Mutex> guard(*mutex);
      assert(!mutex->try_lock());
    }
    assert(mutex->try_lock());
    assert(!mutex->try_lock());
    mutex->unlock();
  });

  auto worker = LAMBDA()
  {
    exclusive_section<Mutex> section{mutex, counter};
    section();
    section();
  };
  concurrent_agents_launch(worker, worker);

  execute_on_main_thread([&] {
    assert(*counter == 0);
    assert(mutex->try_lock());
    mutex->unlock();
  });
}

int main(int, char**)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    (cuda_thread_count = 2;

     test<cuda::std::mutex, local_memory_selector>();
     test<cuda::mutex<cuda::thread_scope_thread>, local_memory_selector>();
     test<cuda::mutex<cuda::thread_scope_block>, local_memory_selector>();
     test<cuda::mutex<cuda::thread_scope_device>, local_memory_selector>();
     test<cuda::mutex<cuda::thread_scope_cluster>, local_memory_selector>();
     test<cuda::mutex<cuda::thread_scope_system>, local_memory_selector>();))

  NV_IF_TARGET(
    NV_IS_DEVICE,
    (test<cuda::std::mutex, shared_memory_selector>();
     test<cuda::mutex<cuda::thread_scope_thread>, shared_memory_selector>();
     test<cuda::mutex<cuda::thread_scope_block>, shared_memory_selector>();
     test<cuda::mutex<cuda::thread_scope_device>, shared_memory_selector>();
     test<cuda::mutex<cuda::thread_scope_system>, shared_memory_selector>();

     test<cuda::std::mutex, global_memory_selector>();
     test<cuda::mutex<cuda::thread_scope_thread>, global_memory_selector>();
     test<cuda::mutex<cuda::thread_scope_block>, global_memory_selector>();
     test<cuda::mutex<cuda::thread_scope_device>, global_memory_selector>();
     test<cuda::mutex<cuda::thread_scope_system>, global_memory_selector>();))

  // Cluster atomics are not available below sm_90.
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (test<cuda::mutex<cuda::thread_scope_cluster>, shared_memory_selector>();
                test<cuda::mutex<cuda::thread_scope_cluster>, global_memory_selector>();))

  return 0;
}
