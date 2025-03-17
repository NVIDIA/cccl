//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-70

// <cuda/std/semaphore>

#include <cuda/std/chrono>
#include <cuda/std/semaphore>

#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <typename Semaphore, template <typename, typename> class Selector, typename Initializer = constructor_initializer>
__host__ __device__ void test()
{
  Selector<Semaphore, Initializer> sel;
  SHARED Semaphore* s;
  s = sel.construct(0);

  auto const start = cuda::std::chrono::high_resolution_clock::now();

  execute_on_main_thread([&] {
    assert(!s->try_acquire_until(start + cuda::std::chrono::milliseconds(250)));
    assert(!s->try_acquire_for(cuda::std::chrono::milliseconds(250)));
  });

  auto releaser = LAMBDA()
  {
    // cuda::std::this_thread::sleep_for(cuda::std::chrono::milliseconds(250));
    s->release();
    // cuda::std::this_thread::sleep_for(cuda::std::chrono::milliseconds(250));
    s->release();
  };

  auto acquirer = LAMBDA()
  {
    assert(s->try_acquire_until(start + cuda::std::chrono::seconds(2)));
    assert(s->try_acquire_for(cuda::std::chrono::seconds(2)));
  };

  concurrent_agents_launch(acquirer, releaser);

  execute_on_main_thread([&] {
    auto const end = cuda::std::chrono::high_resolution_clock::now();
    assert(end - start < cuda::std::chrono::seconds(10));
  });
}

int main(int, char**)
{
  NV_IF_ELSE_TARGET(
    NV_IS_HOST,
    (cuda_thread_count = 2;

     test<cuda::std::counting_semaphore<>, local_memory_selector>();
     test<cuda::counting_semaphore<cuda::thread_scope_block>, local_memory_selector>();
     test<cuda::counting_semaphore<cuda::thread_scope_device>, local_memory_selector>();
     test<cuda::counting_semaphore<cuda::thread_scope_system>, local_memory_selector>();),
    (test<cuda::std::counting_semaphore<>, shared_memory_selector>();
     test<cuda::counting_semaphore<cuda::thread_scope_block>, shared_memory_selector>();
     test<cuda::counting_semaphore<cuda::thread_scope_device>, shared_memory_selector>();
     test<cuda::counting_semaphore<cuda::thread_scope_system>, shared_memory_selector>();

     test<cuda::std::counting_semaphore<>, global_memory_selector>();
     test<cuda::counting_semaphore<cuda::thread_scope_block>, global_memory_selector>();
     test<cuda::counting_semaphore<cuda::thread_scope_device>, global_memory_selector>();
     test<cuda::counting_semaphore<cuda::thread_scope_system>, global_memory_selector>();))

  return 0;
}
