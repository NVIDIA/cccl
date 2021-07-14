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

#include <cuda/std/semaphore>

#include "test_macros.h"
#include "concurrent_agents.h"
#include "cuda_space_selector.h"

template<typename Semaphore,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__host__ __device__
void test()
{
  Selector<Semaphore, Initializer> sel;
  SHARED Semaphore * s;
  s = sel.construct(2);

#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  s->release();
  s->acquire();
#ifdef __CUDA_ARCH__
  }
#endif

  auto acquirer = LAMBDA (){
    s->acquire();
  };
  auto releaser = LAMBDA (){
    s->release(2);
  };

  concurrent_agents_launch(acquirer, releaser);

#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  s->acquire();
#ifdef __CUDA_ARCH__
  }
#endif
}

int main(int, char**)
{
#ifndef __CUDA_ARCH__
  cuda_thread_count = 2;

  test<cuda::std::counting_semaphore<>, local_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_block>, local_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_device>, local_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_system>, local_memory_selector>();
#else
  test<cuda::std::counting_semaphore<>, shared_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_block>, shared_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_device>, shared_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_system>, shared_memory_selector>();

  test<cuda::std::counting_semaphore<>, global_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_block>, global_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_device>, global_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_system>, global_memory_selector>();
#endif

  return 0;
}
