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

#include "test_macros.h"
#include "concurrent_agents.h"

#include "cuda_space_selector.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

template<typename Barrier,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__host__ __device__
void test(bool add_delay = false)
{
  Selector<Barrier, Initializer> sel;
  SHARED Barrier * b;
  b = sel.construct(2);
  bool phase = false;
  auto delay = cuda::std::chrono::duration<int>(0);

  if (add_delay)
    delay = cuda::std::chrono::duration<int>(1);

#ifdef __CUDA_ARCH__
  auto * tok = threadIdx.x == 0 ? new auto(b->arrive()) : nullptr;
#else
  auto * tok = new auto(b->arrive());
#endif
  unused(tok);

  auto awaiter = LAMBDA (){
    const auto until_time = cuda::std::chrono::system_clock::now() + delay;
    while(b->try_wait_parity_until(phase, until_time) == false) {}
  };
  auto arriver = LAMBDA (){
    (void)b->arrive();
  };
  concurrent_agents_launch(awaiter, arriver);

#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
   auto tok2 = b->arrive(2);
   unused(tok2);
   const auto until_time = cuda::std::chrono::system_clock::now() + delay;
   while(b->try_wait_parity_until(!phase, until_time) == false) {}
#ifdef __CUDA_ARCH__
  }
  __syncthreads();
#endif
}

int main(int, char**)
{
#ifndef __CUDA_ARCH__
  //Required by concurrent_agents_launch to know how many we're launching
  cuda_thread_count = 2;

  test<cuda::barrier<cuda::thread_scope_block>, local_memory_selector>();
  test<cuda::barrier<cuda::thread_scope_block>, local_memory_selector>(true);
#else
  test<cuda::barrier<cuda::thread_scope_block>, shared_memory_selector>();
  test<cuda::barrier<cuda::thread_scope_block>, global_memory_selector>();
  test<cuda::barrier<cuda::thread_scope_block>, shared_memory_selector>(true);
  test<cuda::barrier<cuda::thread_scope_block>, global_memory_selector>(true);
#endif
  return 0;
}
