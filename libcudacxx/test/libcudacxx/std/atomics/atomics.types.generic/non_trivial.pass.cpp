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

// NOTE: atomic<> of a TriviallyCopyable class is wrongly rejected by older
// clang versions. It was fixed right before the llvm 3.5 release. See PR18097.
// XFAIL: apple-clang-6.0, clang-3.4, clang-3.3

// <cuda/std/atomic>

// template <class T>
// struct atomic
// {
//     bool is_lock_free() const volatile noexcept;
//     bool is_lock_free() const noexcept;
//     void store(T desr, memory_order m = memory_order_seq_cst) volatile noexcept;
//     void store(T desr, memory_order m = memory_order_seq_cst) noexcept;
//     T load(memory_order m = memory_order_seq_cst) const volatile noexcept;
//     T load(memory_order m = memory_order_seq_cst) const noexcept;
//     operator T() const volatile noexcept;
//     operator T() const noexcept;
//     T exchange(T desr, memory_order m = memory_order_seq_cst) volatile noexcept;
//     T exchange(T desr, memory_order m = memory_order_seq_cst) noexcept;
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order s, memory_order f) volatile noexcept;
//     bool compare_exchange_weak(T& expc, T desr, memory_order s, memory_order f) noexcept;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f) volatile noexcept;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f) noexcept;
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst) volatile noexcept;
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst) noexcept;
//     bool compare_exchange_strong(T& expc, T desr,
//                                 memory_order m = memory_order_seq_cst) volatile noexcept;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order m = memory_order_seq_cst) noexcept;
//
//     atomic() noexcept = default;
//     constexpr atomic(T desr) noexcept;
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
//     T operator=(T) volatile noexcept;
//     T operator=(T) noexcept;
// };

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

struct aggregate
{
  double a;
  double b;
};

struct small_aggregate
{
  uint8_t a;
  uint8_t b;
};

template <class T>
__host__ __device__ void test()
{
  cuda::atomic<T> a({42, 137});
  cuda::std::atomic<T> b({42, 137});
  unused(a);
  unused(b);
}

int main(int, char**)
{
  test<double2>();
  test<cuda::std::pair<double, double>>();
  test<cuda::std::tuple<double, double>>();
  test<aggregate>();
  test<small_aggregate>();

  return 0;
}
