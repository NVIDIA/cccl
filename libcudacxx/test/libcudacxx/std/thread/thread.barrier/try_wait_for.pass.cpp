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
__host__ __device__ void test(bool add_delay = false)
{
  Selector<Barrier, Initializer> sel;
  SHARED Barrier* b;
  b          = sel.construct(2);
  auto delay = cuda::std::chrono::duration<int>(0);

  if (add_delay)
  {
    delay = cuda::std::chrono::duration<int>(1);
  }

  typename Barrier::arrival_token* tok = nullptr;
  execute_on_main_thread([&] {
    tok = new auto(b->arrive());
  });

  auto awaiter = LAMBDA()
  {
    while (b->try_wait_for(cuda::std::move(*tok), delay) == false)
    {
    }
  };
  auto arriver = LAMBDA()
  {
    (void) b->arrive();
  };
  concurrent_agents_launch(awaiter, arriver);

  execute_on_main_thread([&] {
    auto tok2 = b->arrive(2);
    while (b->try_wait_for(cuda::std::move(tok2), delay) == false)
    {
    }
  });
}

int main(int, char**)
{
  NV_IF_ELSE_TARGET(
    NV_IS_HOST,
    (
      // Required by concurrent_agents_launch to know how many we're launching
      cuda_thread_count = 2;

      test<cuda::barrier<cuda::thread_scope_block>, local_memory_selector>();
      test<cuda::barrier<cuda::thread_scope_block>, local_memory_selector>(true);),
    (test<cuda::barrier<cuda::thread_scope_block>, shared_memory_selector>();
     test<cuda::barrier<cuda::thread_scope_block>, global_memory_selector>();
     test<cuda::barrier<cuda::thread_scope_block>, shared_memory_selector>(true);
     test<cuda::barrier<cuda::thread_scope_block>, global_memory_selector>(true);))

  return 0;
}
