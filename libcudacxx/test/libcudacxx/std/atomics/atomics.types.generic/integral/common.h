//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

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
  assert((obj |= T(5)) == T(7));
  assert(obj == T(7));
  assert((obj &= T(0xF)) == T(7));
  assert(obj == T(7));
  assert((obj ^= T(0xF)) == T(8));
  assert(obj == T(8));

#if TEST_STD_VER > 2017
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {23}; A& zero = *new (storage) A(); assert(zero == 0); zero.~A();),
    NV_PROVIDES_SM_70,
    (TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {23}; A& zero = *new (storage) A(); assert(zero == 0); zero.~A();))
#endif // TEST_STD_VER > 2017
}
