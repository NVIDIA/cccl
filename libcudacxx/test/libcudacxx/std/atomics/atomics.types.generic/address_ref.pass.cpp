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
//  ... test case crashes clang.

// <cuda/std/atomic>

// template <class T>
// struct atomic_ref<T*>
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(T* desr, memory_order m = memory_order_seq_cst) volatile;
//     void store(T* desr, memory_order m = memory_order_seq_cst);
//     T* load(memory_order m = memory_order_seq_cst) const volatile;
//     T* load(memory_order m = memory_order_seq_cst) const;
//     operator T*() const volatile;
//     operator T*() const;
//     T* exchange(T* desr, memory_order m = memory_order_seq_cst) volatile;
//     T* exchange(T* desr, memory_order m = memory_order_seq_cst);
//     bool compare_exchange_weak(T*& expc, T* desr,
//                                memory_order s, memory_order f) volatile;
//     bool compare_exchange_weak(T*& expc, T* desr,
//                                memory_order s, memory_order f);
//     bool compare_exchange_strong(T*& expc, T* desr,
//                                  memory_order s, memory_order f) volatile;
//     bool compare_exchange_strong(T*& expc, T* desr,
//                                  memory_order s, memory_order f);
//     bool compare_exchange_weak(T*& expc, T* desr,
//                                memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(T*& expc, T* desr,
//                                memory_order m = memory_order_seq_cst);
//     bool compare_exchange_strong(T*& expc, T* desr,
//                                 memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(T*& expc, T* desr,
//                                  memory_order m = memory_order_seq_cst);
//     T* fetch_add(ptrdiff_t op, memory_order m = memory_order_seq_cst) volatile;
//     T* fetch_add(ptrdiff_t op, memory_order m = memory_order_seq_cst);
//     T* fetch_sub(ptrdiff_t op, memory_order m = memory_order_seq_cst) volatile;
//     T* fetch_sub(ptrdiff_t op, memory_order m = memory_order_seq_cst);
//
//     atomic() = default;
//     constexpr atomic(T* desr);
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
//
//     T* operator=(T*) volatile;
//     T* operator=(T*);
//     T* operator++(int) volatile;
//     T* operator++(int);
//     T* operator--(int) volatile;
//     T* operator--(int);
//     T* operator++() volatile;
//     T* operator++();
//     T* operator--() volatile;
//     T* operator--();
//     T* operator+=(ptrdiff_t op) volatile;
//     T* operator+=(ptrdiff_t op);
//     T* operator-=(ptrdiff_t op) volatile;
//     T* operator-=(ptrdiff_t op);
// };

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include <cmpxchg_loop.h>
#if !defined(TEST_COMPILER_MSVC)
#  include "placement_new.h"
#endif
#include "cuda_space_selector.h"

template <class A, class T, template <typename, typename> class Selector>
__host__ __device__ void do_test()
{
  typedef typename cuda::std::remove_pointer<T>::type X;
  Selector<T, constructor_initializer> sel;
  T& val = *sel.construct(T(0));
  A obj(val);
  bool b0 = obj.is_lock_free();
  ((void) b0); // mark as unused
  assert(obj == T(0));
  obj.store(T(0));
  assert(obj == T(0));
  obj.store(T(1), cuda::std::memory_order_release);
  assert(obj == T(1));
  assert(obj.load() == T(1));
  assert(obj.load(cuda::std::memory_order_acquire) == T(1));
  assert(obj.exchange(T(2)) == T(1));
  assert(obj == T(2));
  assert(obj.exchange(T(3), cuda::std::memory_order_relaxed) == T(2));
  assert(obj == T(3));
  T x = obj;
  assert(cmpxchg_weak_loop(obj, x, T(2)) == true);
  assert(obj == T(2));
  assert(x == T(3));
  assert(obj.compare_exchange_weak(x, T(1)) == false);
  assert(obj == T(2));
  assert(x == T(2));
  x = T(2);
  assert(obj.compare_exchange_strong(x, T(1)) == true);
  assert(obj == T(1));
  assert(x == T(2));
  assert(obj.compare_exchange_strong(x, T(0)) == false);
  assert(obj == T(1));
  assert(x == T(1));
  assert((obj = T(0)) == T(0));
  assert(obj == T(0));
  obj = T(2 * sizeof(X));
  assert((obj += cuda::std::ptrdiff_t(3)) == T(5 * sizeof(X)));
  assert(obj == T(5 * sizeof(X)));
  assert((obj -= cuda::std::ptrdiff_t(3)) == T(2 * sizeof(X)));
  assert(obj == T(2 * sizeof(X)));
}

template <class A, class T, template <typename, typename> class Selector>
__host__ __device__ void test()
{
  do_test<A, T, Selector>();
}

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (test<cuda::std::atomic_ref<int*>, int*, local_memory_selector>();
     test<cuda::atomic_ref<int*, cuda::thread_scope_system>, int*, local_memory_selector>();
     test<cuda::atomic_ref<int*, cuda::thread_scope_device>, int*, local_memory_selector>();
     test<cuda::atomic_ref<int*, cuda::thread_scope_block>, int*, local_memory_selector>();),
    NV_PROVIDES_SM_70,
    (test<cuda::std::atomic_ref<int*>, int*, local_memory_selector>();
     test<cuda::atomic_ref<int*, cuda::thread_scope_system>, int*, local_memory_selector>();
     test<cuda::atomic_ref<int*, cuda::thread_scope_device>, int*, local_memory_selector>();
     test<cuda::atomic_ref<int*, cuda::thread_scope_block>, int*, local_memory_selector>();))

  NV_IF_TARGET(
    NV_IS_DEVICE,
    (test<cuda::std::atomic_ref<int*>, int*, shared_memory_selector>();
     test<cuda::atomic_ref<int*, cuda::thread_scope_system>, int*, shared_memory_selector>();
     test<cuda::atomic_ref<int*, cuda::thread_scope_device>, int*, shared_memory_selector>();
     test<cuda::atomic_ref<int*, cuda::thread_scope_block>, int*, shared_memory_selector>();

     test<cuda::std::atomic_ref<int*>, int*, global_memory_selector>();
     test<cuda::atomic_ref<int*, cuda::thread_scope_system>, int*, global_memory_selector>();
     test<cuda::atomic_ref<int*, cuda::thread_scope_device>, int*, global_memory_selector>();
     test<cuda::atomic_ref<int*, cuda::thread_scope_block>, int*, global_memory_selector>();))

  return 0;
}
