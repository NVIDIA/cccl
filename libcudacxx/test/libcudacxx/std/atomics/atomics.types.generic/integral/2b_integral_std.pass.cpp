//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// template <>
// struct atomic<integral>
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
//     atomic() = default;
//     constexpr atomic(integral desr);
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
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

#include "common.h"

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
  test<Atomic<short, Scope>, short, Selector>();
  test<Atomic<unsigned short, Scope>, unsigned short, Selector>();

#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<Atomic<char16_t, Scope>, char16_t, Selector>();
#endif // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<Atomic<wchar_t, Scope>, wchar_t, Selector>();

  test<Atomic<int16_t, Scope>, int16_t, Selector>();
  test<Atomic<uint16_t, Scope>, uint16_t, Selector>();
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

  NV_DISPATCH_TARGET(NV_IS_HOST,
                     (test_for_all_types<cuda_std_atomic, cuda::thread_scope_system, local_memory_selector>();),
                     NV_PROVIDES_SM_70,
                     (test_for_all_types<cuda_std_atomic, cuda::thread_scope_system, local_memory_selector>();))

  NV_IF_TARGET(NV_IS_DEVICE,
               (test_for_all_types<cuda_std_atomic, cuda::thread_scope_system, shared_memory_selector>();),
               (test_for_all_types<cuda_std_atomic, cuda::thread_scope_system, global_memory_selector>();))

  return 0;
}
