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

// <cuda/std/barrier>

#include <cuda/std/barrier>

#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <typename Barrier, template <typename, typename> class Selector, typename Initializer = constructor_initializer>
__host__ __device__ void test()
{
  Selector<Barrier, Initializer> sel;
  SHARED Barrier* b;
  b = sel.construct(2);

  auto worker = LAMBDA()
  {
    for (int i = 0; i < 10; ++i)
    {
      b->arrive_and_wait();
    }
  };

  concurrent_agents_launch(worker, worker);
}

int main(int, char**)
{
  NV_IF_ELSE_TARGET(
    NV_IS_HOST,
    (cuda_thread_count = 2;

     test<cuda::std::barrier<>, local_memory_selector>();
     test<cuda::barrier<cuda::thread_scope_block>, local_memory_selector>();
     test<cuda::barrier<cuda::thread_scope_device>, local_memory_selector>();
     test<cuda::barrier<cuda::thread_scope_system>, local_memory_selector>();),
    (test<cuda::std::barrier<>, shared_memory_selector>();
     test<cuda::barrier<cuda::thread_scope_block>, shared_memory_selector>();
     test<cuda::barrier<cuda::thread_scope_device>, shared_memory_selector>();
     test<cuda::barrier<cuda::thread_scope_system>, shared_memory_selector>();

     test<cuda::std::barrier<>, global_memory_selector>();
     test<cuda::barrier<cuda::thread_scope_block>, global_memory_selector>();
     test<cuda::barrier<cuda::thread_scope_device>, global_memory_selector>();
     test<cuda::barrier<cuda::thread_scope_system>, global_memory_selector>();))

  return 0;
}
