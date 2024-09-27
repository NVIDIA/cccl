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
//  ... assertion fails line 38

// <cuda/std/atomic>

// template <class T>
//     bool
//     atomic_compare_exchange_weak_explicit(volatile atomic<T>* obj, T* expc,
//                                           T desr,
//                                           memory_order s, memory_order f);
//
// template <class T>
//     bool
//     atomic_compare_exchange_weak_explicit(atomic<T>* obj, T* expc, T desr,
//                                           memory_order s, memory_order f);

#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"
#include <cmpxchg_loop.h>

template <class T, template <typename, typename> class Selector, cuda::thread_scope>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    {
      typedef cuda::std::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& a = *sel.construct();
      T t(T(1));
      cuda::std::atomic_init(&a, t);
      assert(c_cmpxchg_weak_loop(&a, &t, T(2), cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst)
             == true);
      assert(a == T(2));
      assert(t == T(1));
      assert(cuda::std::atomic_compare_exchange_weak_explicit(
               &a, &t, T(3), cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst)
             == false);
      assert(a == T(2));
      assert(t == T(2));
    }
    {
      typedef cuda::std::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& a = *sel.construct();
      T t(T(1));
      cuda::std::atomic_init(&a, t);
      assert(c_cmpxchg_weak_loop(&a, &t, T(2), cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst)
             == true);
      assert(a == T(2));
      assert(t == T(1));
      assert(cuda::std::atomic_compare_exchange_weak_explicit(
               &a, &t, T(3), cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst)
             == false);
      assert(a == T(2));
      assert(t == T(2));
    }
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST,
                     (TestEachAtomicType<TestFn, local_memory_selector>()();),
                     NV_PROVIDES_SM_70,
                     (TestEachAtomicType<TestFn, local_memory_selector>()();))

  NV_IF_TARGET(
    NV_IS_DEVICE,
    (TestEachAtomicType<TestFn, shared_memory_selector>()(); TestEachAtomicType<TestFn, global_memory_selector>()();))

  return 0;
}
