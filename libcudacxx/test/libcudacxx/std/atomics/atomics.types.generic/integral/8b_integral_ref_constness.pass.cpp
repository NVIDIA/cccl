//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// UNSUPPORTED: force-tile
// error: asm statement is unsupported in tile code

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

#include "common_ref.h"

template <template <typename, cuda::thread_scope> class Atomic,
          cuda::thread_scope Scope,
          template <typename, typename> class Selector>
TEST_HOST_DEVICE_FUNC void test_for_all_types()
{
  test<Atomic<long, Scope>, long, Selector>();
  test<Atomic<unsigned long, Scope>, unsigned long, Selector>();
  test<Atomic<long long, Scope>, long long, Selector>();
  test<Atomic<unsigned long long, Scope>, unsigned long long, Selector>();

  test<Atomic<int64_t, Scope>, int64_t, Selector>();
  test<Atomic<uint64_t, Scope>, uint64_t, Selector>();
}

int main(int, char**)
{
  // Split by integer width so ptxas stays within host memory limits,
  // especially when compiling with --enable-tile.

  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (test_for_all_types<cuda_std_atomic_ref_const, cuda::thread_scope_system, local_memory_selector>();
     test_for_all_types<cuda_atomic_ref_const, cuda::thread_scope_system, local_memory_selector>();),
    NV_PROVIDES_SM_70,
    (test_for_all_types<cuda_std_atomic_ref_const, cuda::thread_scope_system, local_memory_selector>();
     test_for_all_types<cuda_atomic_ref_const, cuda::thread_scope_system, local_memory_selector>();))

  NV_IF_TARGET(NV_IS_DEVICE,
               (test_for_all_types<cuda_std_atomic_ref_const, cuda::thread_scope_block, shared_memory_selector>();
                test_for_all_types<cuda_atomic_ref_const, cuda::thread_scope_block, shared_memory_selector>();

                test_for_all_types<cuda_std_atomic_ref_const, cuda::thread_scope_device, global_memory_selector>();
                test_for_all_types<cuda_atomic_ref_const, cuda::thread_scope_device, global_memory_selector>();))

  return 0;
}
