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
// XFAIL: msvc-19.36 && c++20

// Compiler bug for being unable to convert __nv_hdl lambdas
// XFAIL: msvc-19.33
// XFAIL: windows && c++20

// <cuda/std/barrier>

#include <cuda/std/barrier>

#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <template <typename> class Barrier,
          template <typename, typename>
          class Selector,
          typename Initializer = constructor_initializer>
__host__ __device__ void test()
{
  global_memory_selector<int> int_sel;
  SHARED int* x;
  x = int_sel.construct(0);

  auto comp = LAMBDA()
  {
    *x += 1;
  };

  Selector<Barrier<decltype(comp)>, Initializer> sel;
  SHARED Barrier<decltype(comp)>* b;
  b = sel.construct(2, comp);

  auto worker = LAMBDA()
  {
    for (int i = 0; i < 10; ++i)
    {
      b->arrive_and_wait();
    }
    assert(*x == 10);
  };

  concurrent_agents_launch(worker, worker);

  assert(*x == 10);
}

template <typename Comp>
using std_barrier = cuda::std::barrier<Comp>;
template <typename Comp>
using block_barrier = cuda::barrier<cuda::thread_scope_block, Comp>;
template <typename Comp>
using device_barrier = cuda::barrier<cuda::thread_scope_device, Comp>;
template <typename Comp>
using system_barrier = cuda::barrier<cuda::thread_scope_system, Comp>;

int main(int, char**)
{
  NV_IF_ELSE_TARGET(
    NV_IS_HOST,
    (cuda_thread_count = 2;

     test<std_barrier, local_memory_selector>();
     test<block_barrier, local_memory_selector>();
     test<device_barrier, local_memory_selector>();
     test<system_barrier, local_memory_selector>();),
    (test<std_barrier, shared_memory_selector>(); test<block_barrier, shared_memory_selector>();
     test<device_barrier, shared_memory_selector>();
     test<system_barrier, shared_memory_selector>();

     test<std_barrier, global_memory_selector>();
     test<block_barrier, global_memory_selector>();
     test<device_barrier, global_memory_selector>();
     test<system_barrier, global_memory_selector>();))

  return 0;
}
