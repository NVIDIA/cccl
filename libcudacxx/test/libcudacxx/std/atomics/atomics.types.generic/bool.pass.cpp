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

// template <class T>
// struct atomic
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(T desr, memory_order m = memory_order_seq_cst) volatile;
//     void store(T desr, memory_order m = memory_order_seq_cst);
//     T load(memory_order m = memory_order_seq_cst) const volatile;
//     T load(memory_order m = memory_order_seq_cst) const;
//     operator T() const volatile;
//     operator T() const;
//     T exchange(T desr, memory_order m = memory_order_seq_cst) volatile;
//     T exchange(T desr, memory_order m = memory_order_seq_cst);
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order s, memory_order f) volatile;
//     bool compare_exchange_weak(T& expc, T desr, memory_order s, memory_order f);
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f) volatile;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f);
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst);
//     bool compare_exchange_strong(T& expc, T desr,
//                                 memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order m = memory_order_seq_cst);
//
//     atomic() = default;
//     constexpr atomic(T desr);
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
//     T operator=(T) volatile;
//     T operator=(T);
// };
//
// typedef atomic<bool> atomic_bool;

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"
#include <cmpxchg_loop.h>
#if !defined(TEST_COMPILER_MSVC)
#  include "placement_new.h"
#endif
#include "cuda_space_selector.h"

template <template <cuda::thread_scope> class Atomic,
          cuda::thread_scope Scope,
          template <typename, typename>
          class Selector>
__host__ __device__ __noinline__ void do_test()
{
  {
    Selector<volatile Atomic<Scope>, constructor_initializer> sel;
    volatile Atomic<Scope>& obj = *sel.construct(true);
    assert(obj == true);
    bool b0 = obj.is_lock_free();
    (void) b0; // to placate scan-build
    obj.store(false);
    assert(obj == false);
    obj.store(true, cuda::std::memory_order_release);
    assert(obj == true);
    assert(obj.load() == true);
    assert(obj.load(cuda::std::memory_order_acquire) == true);
    assert(obj.exchange(false) == true);
    assert(obj == false);
    assert(obj.exchange(true, cuda::std::memory_order_relaxed) == false);
    assert(obj == true);
    bool x = obj;
    assert(cmpxchg_weak_loop(obj, x, false) == true);
    assert(obj == false);
    assert(x == true);
    assert(obj.compare_exchange_weak(x, true, cuda::std::memory_order_seq_cst) == false);
    assert(obj == false);
    assert(x == false);
    obj.store(true);
    x = true;
    assert(cmpxchg_weak_loop(obj, x, false, cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst) == true);
    assert(obj == false);
    assert(x == true);
    x = true;
    obj.store(true);
    assert(obj.compare_exchange_strong(x, false) == true);
    assert(obj == false);
    assert(x == true);
    assert(obj.compare_exchange_strong(x, true, cuda::std::memory_order_seq_cst) == false);
    assert(obj == false);
    assert(x == false);
    x = true;
    obj.store(true);
    assert(obj.compare_exchange_strong(x, false, cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst)
           == true);
    assert(obj == false);
    assert(x == true);
    assert((obj = false) == false);
    assert(obj == false);
    assert((obj = true) == true);
    assert(obj == true);
  }
  {
    Selector<Atomic<Scope>, constructor_initializer> sel;
    Atomic<Scope>& obj = *sel.construct(true);
    assert(obj == true);
    bool b0 = obj.is_lock_free();
    (void) b0; // to placate scan-build
    obj.store(false);
    assert(obj == false);
    obj.store(true, cuda::std::memory_order_release);
    assert(obj == true);
    assert(obj.load() == true);
    assert(obj.load(cuda::std::memory_order_acquire) == true);
    assert(obj.exchange(false) == true);
    assert(obj == false);
    assert(obj.exchange(true, cuda::std::memory_order_relaxed) == false);
    assert(obj == true);
    bool x = obj;
    assert(cmpxchg_weak_loop(obj, x, false) == true);
    assert(obj == false);
    assert(x == true);
    assert(obj.compare_exchange_weak(x, true, cuda::std::memory_order_seq_cst) == false);
    assert(obj == false);
    assert(x == false);
    obj.store(true);
    x = true;
    assert(cmpxchg_weak_loop(obj, x, false, cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst) == true);
    assert(obj == false);
    assert(x == true);
    x = true;
    obj.store(true);
    assert(obj.compare_exchange_strong(x, false) == true);
    assert(obj == false);
    assert(x == true);
    assert(obj.compare_exchange_strong(x, true, cuda::std::memory_order_seq_cst) == false);
    assert(obj == false);
    assert(x == false);
    x = true;
    obj.store(true);
    assert(obj.compare_exchange_strong(x, false, cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst)
           == true);
    assert(obj == false);
    assert(x == true);
    assert((obj = false) == false);
    assert(obj == false);
    assert((obj = true) == true);
    assert(obj == true);
  }
  {
    Selector<Atomic<Scope>, constructor_initializer> sel;
    Atomic<Scope>& obj = *sel.construct(true);
    assert(obj == true);
    bool b0 = obj.is_lock_free();
    (void) b0; // to placate scan-build
    obj.store(false);
    assert(obj == false);
    obj.store(true, cuda::std::memory_order_release);
    assert(obj == true);
    assert(obj.load() == true);
    assert(obj.load(cuda::std::memory_order_acquire) == true);
    assert(obj.exchange(false) == true);
    assert(obj == false);
    assert(obj.exchange(true, cuda::std::memory_order_relaxed) == false);
    assert(obj == true);
    bool x = obj;
    assert(cmpxchg_weak_loop(obj, x, false) == true);
    assert(obj == false);
    assert(x == true);
    assert(obj.compare_exchange_weak(x, true, cuda::std::memory_order_seq_cst) == false);
    assert(obj == false);
    assert(x == false);
    obj.store(true);
    x = true;
    assert(cmpxchg_weak_loop(obj, x, false, cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst) == true);
    assert(obj == false);
    assert(x == true);
    x = true;
    obj.store(true);
    assert(obj.compare_exchange_strong(x, false) == true);
    assert(obj == false);
    assert(x == true);
    assert(obj.compare_exchange_strong(x, true, cuda::std::memory_order_seq_cst) == false);
    assert(obj == false);
    assert(x == false);
    x = true;
    obj.store(true);
    assert(obj.compare_exchange_strong(x, false, cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst)
           == true);
    assert(obj == false);
    assert(x == true);
    assert((obj = false) == false);
    assert(obj == false);
    assert((obj = true) == true);
    assert(obj == true);
  }

#if TEST_STD_VER > 2017
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (typedef Atomic<Scope> A; TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {1}; A& zero = *new (storage) A();
     assert(zero == false);
     zero.~A();),
    NV_PROVIDES_SM_70,
    (typedef Atomic<Scope> A; TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {1}; A& zero = *new (storage) A();
     assert(zero == false);
     zero.~A();))
#endif // TEST_STD_VER > 2017
}

template <cuda::thread_scope Scope>
using cuda_std_atomic = cuda::std::atomic<bool>;

template <cuda::thread_scope Scope>
using cuda_atomic = cuda::atomic<bool, Scope>;

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (do_test<cuda_std_atomic, cuda::thread_scope_system, local_memory_selector>();
     do_test<cuda_atomic, cuda::thread_scope_system, local_memory_selector>();
     do_test<cuda_atomic, cuda::thread_scope_device, local_memory_selector>();
     do_test<cuda_atomic, cuda::thread_scope_block, local_memory_selector>();),
    NV_PROVIDES_SM_70,
    (do_test<cuda_std_atomic, cuda::thread_scope_system, local_memory_selector>();
     do_test<cuda_atomic, cuda::thread_scope_system, local_memory_selector>();
     do_test<cuda_atomic, cuda::thread_scope_device, local_memory_selector>();
     do_test<cuda_atomic, cuda::thread_scope_block, local_memory_selector>();))

  NV_IF_TARGET(
    NV_IS_DEVICE,
    (do_test<cuda_std_atomic, cuda::thread_scope_system, shared_memory_selector>();
     do_test<cuda_atomic, cuda::thread_scope_system, shared_memory_selector>();
     do_test<cuda_atomic, cuda::thread_scope_device, shared_memory_selector>();
     do_test<cuda_atomic, cuda::thread_scope_block, shared_memory_selector>();

     do_test<cuda_std_atomic, cuda::thread_scope_system, global_memory_selector>();
     do_test<cuda_atomic, cuda::thread_scope_system, global_memory_selector>();
     do_test<cuda_atomic, cuda::thread_scope_device, global_memory_selector>();
     do_test<cuda_atomic, cuda::thread_scope_block, global_memory_selector>();))

  return 0;
}
