//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-80

// <cuda/std/barrier>

#include <cuda/std/barrier>

#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <typename Barrier, template <typename, typename> class Selector, typename Initializer = constructor_initializer>
__host__ __device__ int test(bool add_delay = false)
{
  printf("delay %s\r\n", add_delay ? "enabled" : "disabled");

  Selector<Barrier, Initializer> sel;
  SHARED Barrier* b;
  b            = sel.construct(2);
  auto delay   = cuda::std::chrono::nanoseconds(0);
  auto timeout = cuda::std::chrono::nanoseconds(100000000);

  if (add_delay)
  {
    delay = cuda::std::chrono::nanoseconds(100000);
  }

  auto time = cuda::std::chrono::high_resolution_clock::now();
  cuda::std::atomic_ref<decltype(time)> time_ref(time);

  auto measure = LAMBDA()->cuda::std::chrono::nanoseconds
  {
    return cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(
      cuda::std::chrono::high_resolution_clock::now() - time_ref.load());
  };

  {
    typename Barrier::arrival_token* tok = nullptr;
    execute_on_main_thread([&] {
      tok = new auto(b->arrive());
    });

    auto awaiter = LAMBDA()
    {
      time_ref = ::cuda::std::chrono::high_resolution_clock::now();
      while ((b->try_wait_for(cuda::std::move(*tok), delay) == false) && (measure() < timeout))
      {
      }
      printf("p1 barrier delay: %lluns\r\n", measure().count());
    };
    auto arriver = LAMBDA()
    {
      (void) b->arrive();
    };
    concurrent_agents_launch(awaiter, arriver);
    if (measure() > timeout)
    {
      printf("Deadlock detected in p1\r\n");
      return 1;
    }
  }
  {
    execute_on_main_thread([&] {
      auto tok2 = b->arrive(2);
      time_ref  = ::cuda::std::chrono::high_resolution_clock::now();
      while ((b->try_wait_for(cuda::std::move(tok2), delay) == false) && (measure() < timeout))
      {
      }
      printf("p2 barrier delay: %lluns\r\n", measure().count());
    });
    if (measure() > timeout)
    {
      printf("Deadlock detected in p2\r\n");
      return 1;
    }
  }
  return 0;
}

int main(int, char**)
{
  int failure = 0;
  NV_IF_TARGET(
    NV_IS_HOST,
    (cuda_thread_count = 2; failure |= test<cuda::barrier<cuda::thread_scope_block>, local_memory_selector>();
     failure |= test<cuda::barrier<cuda::thread_scope_block>, local_memory_selector>(true);),
    (failure |= test<cuda::barrier<cuda::thread_scope_block>, shared_memory_selector>();
     failure |= test<cuda::barrier<cuda::thread_scope_block>, global_memory_selector>();
     failure |= test<cuda::barrier<cuda::thread_scope_block>, shared_memory_selector>(true);
     failure |= test<cuda::barrier<cuda::thread_scope_block>, global_memory_selector>(true);))

  return failure;
}
