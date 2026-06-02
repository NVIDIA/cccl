//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: asm statement is unsupported in tile code

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70
//  ... test crashes clang

// <cuda/std/atomic>

// template <class Integral>
//     Integral
//     atomic_fetch_max_explicit(volatile atomic<Integral>* obj, Integral op, memory_order m);
//
// template <class Integral>
//     Integral
//     atomic_fetch_max_explicit(atomic<Integral>* obj, Integral op, memory_order m);
//
// template <class T>
//     T*
//     atomic_fetch_max_explicit(volatile atomic<T*>* obj, T* op, memory_order m);
//
// template <class T>
//     T*
//     atomic_fetch_max_explicit(atomic<T*>* obj, T* op, memory_order m);

#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T, template <typename, typename> class Selector, cuda::thread_scope>
struct TestFn
{
  TEST_FUNC void operator()() const
  {
    // op greater than the stored value: replaced, old value returned
    {
      using A = cuda::std::atomic<T>;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      cuda::std::atomic_init(&t, T(1));
      assert(cuda::std::atomic_fetch_max_explicit(&t, T(3), cuda::std::memory_order_seq_cst) == T(1));
      assert(t == T(3));
    }
    {
      using A = cuda::std::atomic<T>;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      cuda::std::atomic_init(&t, T(1));
      assert(cuda::std::atomic_fetch_max_explicit(&t, T(3), cuda::std::memory_order_seq_cst) == T(1));
      assert(t == T(3));
    }
    // op not greater than the stored value: unchanged, old value returned
    {
      using A = cuda::std::atomic<T>;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      cuda::std::atomic_init(&t, T(3));
      assert(cuda::std::atomic_fetch_max_explicit(&t, T(1), cuda::std::memory_order_relaxed) == T(3));
      assert(t == T(3));
    }
    {
      using A = cuda::std::atomic<T>;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      cuda::std::atomic_init(&t, T(3));
      assert(cuda::std::atomic_fetch_max_explicit(&t, T(1), cuda::std::memory_order_relaxed) == T(3));
      assert(t == T(3));
    }
  }
};

template <class T, template <typename, typename> class Selector>
TEST_FUNC void testp()
{
  using X = typename cuda::std::remove_pointer<T>::type;
  // Pointers into the same array have a well-defined ordering.
  X arr[8] = {};
  {
    using A = cuda::std::atomic<T>;
    Selector<A, constructor_initializer> sel;
    A& t = *sel.construct();
    cuda::std::atomic_init(&t, arr + 2);
    assert(cuda::std::atomic_fetch_max_explicit(&t, arr + 5, cuda::std::memory_order_seq_cst) == arr + 2);
    assert(t == arr + 5);
  }
  {
    using A = cuda::std::atomic<T>;
    Selector<volatile A, constructor_initializer> sel;
    volatile A& t = *sel.construct();
    cuda::std::atomic_init(&t, arr + 5);
    assert(cuda::std::atomic_fetch_max_explicit(&t, arr + 2, cuda::std::memory_order_seq_cst) == arr + 5);
    assert(t == arr + 5);
  }
}

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (TestEachIntegralType<TestFn, local_memory_selector>()(); testp<int*, local_memory_selector>();
     testp<const int*, local_memory_selector>();),
    NV_PROVIDES_SM_70,
    (TestEachIntegralType<TestFn, local_memory_selector>()(); testp<int*, local_memory_selector>();
     testp<const int*, local_memory_selector>();))

  NV_IF_TARGET(
    NV_IS_DEVICE,
    (TestEachIntegralType<TestFn, shared_memory_selector>()(); testp<int*, shared_memory_selector>();
     testp<const int*, shared_memory_selector>();
     TestEachIntegralType<TestFn, global_memory_selector>()();
     testp<int*, global_memory_selector>();
     testp<const int*, global_memory_selector>();))

  return 0;
}
