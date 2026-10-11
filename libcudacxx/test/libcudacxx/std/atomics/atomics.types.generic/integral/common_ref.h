//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_ATOMICS_ATOMICS_TYPES_GENERIC_INTEGRAL_COMMON_REF_H
#define TEST_STD_ATOMICS_ATOMICS_TYPES_GENERIC_INTEGRAL_COMMON_REF_H

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include <cmpxchg_loop.h>

#include "test_macros.h"
#if !TEST_COMPILER(MSVC)
#  include "placement_new.h"
#endif // !TEST_COMPILER(MSVC)
#include "cuda_space_selector.h"

template <class A, class T, template <typename, typename> class Selector>
TEST_HOST_DEVICE_FUNC __noinline__ void do_test()
{
  Selector<T, constructor_initializer> sel;
  T& val = *sel.construct(T(0));
  assert(&val);
  assert(val == T(0));
  A obj(val);
  assert(obj.load() == T(0));
  [[maybe_unused]] bool b0 = obj.is_lock_free();
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
TEST_HOST_DEVICE_FUNC __noinline__ void test()
{
  do_test<A, T, Selector>();
}

template <typename T, cuda::thread_scope Scope>
using cuda_std_atomic_ref = cuda::std::atomic_ref<T>;

template <typename T, cuda::thread_scope Scope>
using cuda_atomic_ref = cuda::atomic_ref<T, Scope>;

template <typename T, cuda::thread_scope Scope>
using cuda_std_atomic_ref_const = const cuda::std::atomic_ref<T>;

template <typename T, cuda::thread_scope Scope>
using cuda_atomic_ref_const = const cuda::atomic_ref<T, Scope>;

#endif // TEST_STD_ATOMICS_ATOMICS_TYPES_GENERIC_INTEGRAL_COMMON_REF_H
