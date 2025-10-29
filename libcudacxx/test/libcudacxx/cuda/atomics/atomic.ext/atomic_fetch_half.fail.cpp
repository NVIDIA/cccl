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
// UNSUPPORTED: nvcc-11, nvcc-12

// <cuda/atomic>

// TODO: Add support for new half

#include <cuda/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T, template <typename, typename> class Selector, cuda::thread_scope ThreadScope>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    // Fetch min
    {
      using A = cuda::atomic<T, ThreadScope>;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(-1);
      assert(t.fetch_min(T(-5)) == T(-1));
      printf("%i == %i\n", (int) t.load(), (int) T(-5));
      NV_IF_TARGET(NV_IS_HOST, (fflush(stdout);))
      assert(t.load() == T(-5));
    }
    {
      using A = cuda::atomic<T, ThreadScope>;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(-1);
      assert(t.fetch_min(T(-5)) == T(-1));
      assert(t.load() == T(-5));
    }
    // Test not lesser
    {
      using A = cuda::atomic<T, ThreadScope>;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(-1);
      assert(t.fetch_min(4) == T(-1));
      assert(t.load() == T(-1));
    }
    {
      using A = cuda::atomic<T, ThreadScope>;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(-1);
      assert(t.fetch_min(4) == T(-1));
      assert(t.load() == T(-1));
    }
    // Fetch max
    {
      using A = cuda::atomic<T>;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(1);
      assert(t.fetch_max(2) == T(1));
      assert(t.load() == T(2));
    }
    {
      using A = cuda::atomic<T>;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(1);
      assert(t.fetch_max(2) == T(1));
      assert(t.load() == T(2));
    }
    // Test not greater
    {
      using A = cuda::atomic<T>;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(3);
      assert(t.fetch_max(2) == T(3));
      assert(t.load() == T(3));
    }
    {
      using A = cuda::atomic<T>;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(3);
      assert(t.fetch_max(2) == T(3));
      assert(t.load() == T(3));
    }
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST,
                     (TestFn<__half, local_memory_selector, cuda::thread_scope::thread_scope_thread>()();),
                     NV_PROVIDES_SM_70,
                     (TestFn<__half, local_memory_selector, cuda::thread_scope::thread_scope_thread>()();))

  NV_IF_TARGET(NV_IS_DEVICE,
               (TestFn<__half, shared_memory_selector, cuda::thread_scope::thread_scope_thread>()();
                TestFn<__half, global_memory_selector, cuda::thread_scope::thread_scope_thread>()();))

  return 0;
}
