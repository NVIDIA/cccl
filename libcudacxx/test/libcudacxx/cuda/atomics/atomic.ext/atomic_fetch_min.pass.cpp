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

template <class T,
          template <typename, typename>
          class Selector,
          cuda::thread_scope ThreadScope,
          bool Signed = cuda::std::is_signed<T>::value>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    // Test lesser
    {
      typedef cuda::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(5);
      assert(t.fetch_min(4) == T(5));
      assert(t.load() == T(4));
    }
    {
      typedef cuda::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(5);
      assert(t.fetch_min(4) == T(5));
      assert(t.load() == T(4));
    }
    // Test not lesser
    {
      typedef cuda::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(3);
      assert(t.fetch_min(4) == T(3));
      assert(t.load() == T(3));
    }
    {
      typedef cuda::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(3);
      assert(t.fetch_min(4) == T(3));
      assert(t.load() == T(3));
    }
  }
};

template <class T, template <typename, typename> class Selector, cuda::thread_scope ThreadScope>
struct TestFn<T, Selector, ThreadScope, true>
{
  __host__ __device__ void operator()() const
  {
    // Call unsigned tests
    TestFn<T, Selector, ThreadScope, false>()();
    // Test lesser, but with signed math
    {
      typedef cuda::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(-1);
      assert(t.fetch_min(-5) == T(-1));
      assert(t.load() == T(-5));
    }
    {
      typedef cuda::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(-1);
      assert(t.fetch_min(-5) == T(-1));
      assert(t.load() == T(-5));
    }
    // Test not lesser
    {
      typedef cuda::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(-1);
      assert(t.fetch_min(4) == T(-1));
      assert(t.load() == T(-1));
    }
    {
      typedef cuda::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(-1);
      assert(t.fetch_min(4) == T(-1));
      assert(t.load() == T(-1));
    }
  }
};

template <class T, template <typename, typename> class Selector, cuda::thread_scope ThreadScope>
struct TestFnDispatch
{
  __host__ __device__ void operator()() const
  {
    TestFn<T, Selector, ThreadScope>()();
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (TestEachIntegralType<TestFnDispatch, local_memory_selector>()();
     TestEachFloatingPointType<TestFnDispatch, local_memory_selector>()();),
    NV_PROVIDES_SM_70,
    (TestEachIntegralType<TestFnDispatch, local_memory_selector>()();
     TestEachFloatingPointType<TestFnDispatch, local_memory_selector>()();))

  NV_IF_TARGET(NV_IS_DEVICE,
               (TestEachIntegralType<TestFnDispatch, shared_memory_selector>()();
                TestEachFloatingPointType<TestFnDispatch, shared_memory_selector>()();
                TestEachIntegralType<TestFnDispatch, global_memory_selector>()();
                TestEachFloatingPointType<TestFnDispatch, global_memory_selector>()();))

  return 0;
}
