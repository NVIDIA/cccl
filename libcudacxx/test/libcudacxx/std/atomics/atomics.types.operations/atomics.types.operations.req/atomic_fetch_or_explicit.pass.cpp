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

// <cuda/std/atomic>

// template <class Integral>
//     Integral
//     atomic_fetch_or_explicit(volatile atomic<Integral>* obj, Integral op);
//
// template <class Integral>
//     Integral
//     atomic_fetch_or_explicit(atomic<Integral>* obj, Integral op);

#include <cuda/std/atomic>
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
      typedef cuda::std::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      cuda::std::atomic_init(&t, T(1));
      assert(cuda::std::atomic_fetch_or_explicit(&t, T(2), cuda::std::memory_order_seq_cst) == T(1));
      assert(t == T(3));
    }
    {
      typedef cuda::std::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      cuda::std::atomic_init(&t, T(3));
      assert(cuda::std::atomic_fetch_or_explicit(&t, T(2), cuda::std::memory_order_seq_cst) == T(3));
      assert(t == T(3));
    }
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST,
                     (TestEachIntegralType<TestFn, local_memory_selector>()();),
                     NV_PROVIDES_SM_70,
                     (TestEachIntegralType<TestFn, local_memory_selector>()();))

  NV_IF_TARGET(NV_IS_DEVICE,
               (TestEachIntegralType<TestFn, shared_memory_selector>()();
                TestEachIntegralType<TestFn, global_memory_selector>()();))

  return 0;
}
