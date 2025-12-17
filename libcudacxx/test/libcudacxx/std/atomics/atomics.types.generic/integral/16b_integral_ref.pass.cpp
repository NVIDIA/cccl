//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: pre-sm-90
// UNSUPPORTED: windows
// UNSUPPORTED: aarch64-unknown-linux-gnu

// <cuda/std/atomic>

// template <>
// struct atomic_ref<integral>
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(integral desr, memory_order m = memory_order_seq_cst) volatile;
//     void store(integral desr, memory_order m = memory_order_seq_cst);
//     integral load(memory_order m = memory_order_seq_cst) const volatile;
//     integral load(memory_order m = memory_order_seq_cst) const;
//     operator integral() const volatile;
//     operator integral() const;
//     integral exchange(integral desr,
//                       memory_order m = memory_order_seq_cst) volatile;
//     integral exchange(integral desr, memory_order m = memory_order_seq_cst);
//     bool compare_exchange_weak(integral& expc, integral desr,
//                                memory_order s, memory_order f) volatile;
//     bool compare_exchange_weak(integral& expc, integral desr,
//                                memory_order s, memory_order f);
//     bool compare_exchange_strong(integral& expc, integral desr,
//                                  memory_order s, memory_order f) volatile;
//     bool compare_exchange_strong(integral& expc, integral desr,
//                                  memory_order s, memory_order f);
//     bool compare_exchange_weak(integral& expc, integral desr,
//                                memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(integral& expc, integral desr,
//                                memory_order m = memory_order_seq_cst);
//     bool compare_exchange_strong(integral& expc, integral desr,
//                                 memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(integral& expc, integral desr,
//                                  memory_order m = memory_order_seq_cst);
//
//     integral
//         fetch_add(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_add(integral op, memory_order m = memory_order_seq_cst);
//     integral
//         fetch_sub(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_sub(integral op, memory_order m = memory_order_seq_cst);
//     integral
//         fetch_and(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_and(integral op, memory_order m = memory_order_seq_cst);
//     integral
//         fetch_or(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_or(integral op, memory_order m = memory_order_seq_cst);
//     integral
//         fetch_xor(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_xor(integral op, memory_order m = memory_order_seq_cst);
//
//     atomic_ref() = delete;
//     constexpr atomic_ref(integral& desr);
//     atomic_ref(const atomic_ref&) = default;
//     atomic_ref& operator=(const atomic_ref&) = delete;
//     atomic_ref& operator=(const atomic_ref&) volatile = delete;
//     integral operator=(integral desr) volatile;
//     integral operator=(integral desr);
//
//     integral operator++(int) volatile;
//     integral operator++(int);
//     integral operator--(int) volatile;
//     integral operator--(int);
//     integral operator++() volatile;
//     integral operator++();
//     integral operator--() volatile;
//     integral operator--();
//     integral operator+=(integral op) volatile;
//     integral operator+=(integral op);
//     integral operator-=(integral op) volatile;
//     integral operator-=(integral op);
//     integral operator&=(integral op) volatile;
//     integral operator&=(integral op);
//     integral operator|=(integral op) volatile;
//     integral operator|=(integral op);
//     integral operator^=(integral op) volatile;
//     integral operator^=(integral op);
// };

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include <cmpxchg_loop.h>

#include "test_macros.h"
#if !defined(TEST_COMPILER_MSVC)
#  include "placement_new.h"
#endif
#include "cuda_space_selector.h"

template <class A, class T, template <typename, typename> class Selector>
__host__ __device__ __noinline__ void do_test()
{
  Selector<T, constructor_initializer> sel;
  T& val = *sel.construct(T(0));
  assert(&val);
  assert(val == T(0));
  A obj(val);
  assert(obj.load() == T(0));
  bool b0 = obj.is_lock_free();
  ((void) b0); // mark as unused
  obj.store(T(0));
  assert(obj.load() == T(0));
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
  assert((obj |= T(5)) == T(7));
  assert(obj == T(7));
  assert((obj &= T(0xF)) == T(7));
  assert(obj == T(7));
  assert((obj ^= T(0xF)) == T(8));
  assert(obj == T(8));
}

template <class A, class T, template <typename, typename> class Selector>
__host__ __device__ __noinline__ void test()
{
  do_test<A, T, Selector>();
}

template <template <typename, cuda::thread_scope> class Atomic,
          cuda::thread_scope Scope,
          template <typename, typename> class Selector>
__host__ __device__ void test_for_all_types()
{
  test<Atomic<__int128_t, Scope>, __int128_t, Selector>();
  test<Atomic<__uint128_t, Scope>, __uint128_t, Selector>();
}

template <typename T, cuda::thread_scope Scope>
using cuda_std_atomic_ref = cuda::std::atomic_ref<T>;

template <typename T, cuda::thread_scope Scope>
using cuda_atomic_ref = cuda::atomic_ref<T, Scope>;

int main(int, char**)
{
  // Skip tests if PTX is insufficient
#if __cccl_ptx_isa >= 840
  NV_DISPATCH_TARGET(
    /* TODO: Enable when lit is capable of parsing flags
    NV_IS_HOST,
    (test_for_all_types<cuda_std_atomic_ref, cuda::thread_scope_system, local_memory_selector>();
    test_for_all_types<cuda_atomic_ref, cuda::thread_scope_system, local_memory_selector>();),
    */
    NV_PROVIDES_SM_90,
    (test_for_all_types<cuda_std_atomic_ref, cuda::thread_scope_system, local_memory_selector>();
     test_for_all_types<cuda_atomic_ref, cuda::thread_scope_system, local_memory_selector>();

     test_for_all_types<cuda_std_atomic_ref, cuda::thread_scope_block, shared_memory_selector>();
     test_for_all_types<cuda_atomic_ref, cuda::thread_scope_block, shared_memory_selector>();

     test_for_all_types<cuda_std_atomic_ref, cuda::thread_scope_device, global_memory_selector>();
     test_for_all_types<cuda_atomic_ref, cuda::thread_scope_device, global_memory_selector>();))
#endif

  return 0;
}
