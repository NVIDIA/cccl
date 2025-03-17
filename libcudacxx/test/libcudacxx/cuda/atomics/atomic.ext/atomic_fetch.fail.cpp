//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/atomic>

#include <cuda/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T, template <typename, typename> class Selector, cuda::thread_scope>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    {
      typedef cuda::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t.fetch_min(4);
    }
    {
      typedef cuda::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t.fetch_max(4);
    }
    T tmp = T(0);
    {
      cuda::atomic_ref<T> t(tmp);
      t.fetch_min(4);
    }
    {
      cuda::atomic_ref<T> t(tmp);
      t.fetch_max(4);
    }
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST, TestFn<__half, local_memory_selector, cuda::thread_scope::thread_scope_thread>()();
    , NV_PROVIDES_SM_70, (TestFn<__half, local_memory_selector, cuda::thread_scope::thread_scope_thread>()();))

  NV_IF_TARGET(NV_IS_DEVICE,
               (TestFn<__half, shared_memory_selector, cuda::thread_scope::thread_scope_thread>()();
                TestFn<__half, global_memory_selector, cuda::thread_scope::thread_scope_thread>()();))

  return 0;
}
