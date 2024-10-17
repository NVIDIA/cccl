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

// template <>
// struct atomic<floating_point>
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(floating_point desr, memory_order m = memory_order_seq_cst) volatile;
//     void store(floating_point desr, memory_order m = memory_order_seq_cst);
//     floating_point load(memory_order m = memory_order_seq_cst) const volatile;
//     floating_point load(memory_order m = memory_order_seq_cst) const;
//     operator floating_point() const volatile;
//     operator floating_point() const;
//     floating_point exchange(floating_point desr,
//                       memory_order m = memory_order_seq_cst) volatile;
//     floating_point exchange(floating_point desr, memory_order m = memory_order_seq_cst);
//     bool compare_exchange_weak(floating_point& expc, floating_point desr,
//                                memory_order s, memory_order f) volatile;
//     bool compare_exchange_weak(floating_point& expc, floating_point desr,
//                                memory_order s, memory_order f);
//     bool compare_exchange_strong(floating_point& expc, floating_point desr,
//                                  memory_order s, memory_order f) volatile;
//     bool compare_exchange_strong(floating_point& expc, floating_point desr,
//                                  memory_order s, memory_order f);
//     bool compare_exchange_weak(floating_point& expc, floating_point desr,
//                                memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(floating_point& expc, floating_point desr,
//                                memory_order m = memory_order_seq_cst);
//     bool compare_exchange_strong(floating_point& expc, floating_point desr,
//                                 memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(floating_point& expc, floating_point desr,
//                                  memory_order m = memory_order_seq_cst);
//
//     floating_point
//         fetch_add(floating_point op, memory_order m = memory_order_seq_cst) volatile;
//     floating_point fetch_add(floating_point op, memory_order m = memory_order_seq_cst);
//     floating_point
//         fetch_sub(floating_point op, memory_order m = memory_order_seq_cst) volatile;
//     floating_point fetch_sub(floating_point op, memory_order m = memory_order_seq_cst);
//     floating_point
//
//     atomic() = default;
//     constexpr atomic(floating_point desr);
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
//     floating_point operator=(floating_point desr) volatile;
//     floating_point operator=(floating_point desr);
//
//     floating_point operator++(int) volatile;
//     floating_point operator++(int);
//     floating_point operator--(int) volatile;
//     floating_point operator--(int);
//     floating_point operator++() volatile;
//     floating_point operator++();
//     floating_point operator--() volatile;
//     floating_point operator--();
//     floating_point operator+=(floating_point op) volatile;
//     floating_point operator+=(floating_point op);
//     floating_point operator-=(floating_point op) volatile;
//     floating_point operator-=(floating_point op);
// };

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"
#include <cmpxchg_loop.h>
#if !defined(TEST_COMPILER_MSVC)
#  include "placement_new.h"
#endif
#include "cuda_space_selector.h"

template <class A, class T, template <typename, typename> class Selector>
__host__ __device__ __noinline__ void do_test()
{
  Selector<A, constructor_initializer> sel;
  A& obj  = *sel.construct(T(0));
  bool b0 = obj.is_lock_free();
  ((void) b0); // mark as unused
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
  assert(obj++ == T(0));
  assert(obj == T(1));
  assert(++obj == T(2));
  assert(obj == T(2));
  assert(--obj == T(1));
  assert(obj == T(1));
  assert(obj-- == T(1));
  assert(obj == T(0));
  obj = T(2);
  assert((obj += T(3)) == T(5));
  assert(obj == T(5));
  assert((obj -= T(3)) == T(2));
  assert(obj == T(2));

#if TEST_STD_VER > 2017
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {23}; A& zero = *new (storage) A(); assert(zero == 0); zero.~A();),
    NV_PROVIDES_SM_70,
    (TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {23}; A& zero = *new (storage) A(); assert(zero == 0); zero.~A();))
#endif // TEST_STD_VER > 2017
}

template <class A, class T, template <typename, typename> class Selector>
__host__ __device__ __noinline__ void test()
{
  do_test<A, T, Selector>();
  do_test<volatile A, T, Selector>();
}

template <template <typename, cuda::thread_scope> class Atomic,
          cuda::thread_scope Scope,
          template <typename, typename>
          class Selector>
__host__ __device__ void test_for_all_types()
{
  test<Atomic<float, Scope>, float, Selector>();
  test<Atomic<double, Scope>, double, Selector>();
}

template <typename T, cuda::thread_scope Scope>
using cuda_std_atomic = cuda::std::atomic<T>;

template <typename T, cuda::thread_scope Scope>
using cuda_atomic = cuda::atomic<T, Scope>;

int main(int, char**)
{
  // this test would instantiate more cases than just the ones below
  // but ptxas already consumes 5 GB of RAM while translating these
  // so in the interest of not eating all memory, it's limited to the current set
  //
  // the per-function tests *should* cover the other codegen aspects of the
  // code, and the cross between scopes and memory locations below should provide
  // a *reasonable* subset of all the possible combinations to provide enough
  // confidence that this all actually works

  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (test_for_all_types<cuda_std_atomic, cuda::thread_scope_system, local_memory_selector>();
     test_for_all_types<cuda_atomic, cuda::thread_scope_system, local_memory_selector>();),
    NV_PROVIDES_SM_70,
    (test_for_all_types<cuda_std_atomic, cuda::thread_scope_system, local_memory_selector>();
     test_for_all_types<cuda_atomic, cuda::thread_scope_system, local_memory_selector>();))

  NV_IF_TARGET(NV_IS_DEVICE,
               (test_for_all_types<cuda_std_atomic, cuda::thread_scope_system, shared_memory_selector>();
                test_for_all_types<cuda_atomic, cuda::thread_scope_block, shared_memory_selector>();

                test_for_all_types<cuda_std_atomic, cuda::thread_scope_system, global_memory_selector>();
                test_for_all_types<cuda_atomic, cuda::thread_scope_device, global_memory_selector>();))

  return 0;
}
