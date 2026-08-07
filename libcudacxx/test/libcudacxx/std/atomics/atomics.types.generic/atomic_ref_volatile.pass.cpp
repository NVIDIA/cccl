//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// UNSUPPORTED: force-tile
// error: asm statement is unsupported in tile code

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class AtomicRef>
TEST_HOST_DEVICE_FUNC void test()
{
  static_assert(cuda::std::is_same_v<typename AtomicRef::value_type, int>);
  static_assert(AtomicRef::is_always_lock_free);

  volatile int value = 0;
  AtomicRef atom(value);

  atom.store(1, cuda::std::memory_order_release);
  assert(atom.load(cuda::std::memory_order_acquire) == 1);
  assert(atom.exchange(2, cuda::std::memory_order_acq_rel) == 1);

  int expected = 2;
  assert(atom.compare_exchange_strong(expected, 3, cuda::std::memory_order_acq_rel, cuda::std::memory_order_acquire));
  assert(expected == 2);

  expected = 2;
  assert(!atom.compare_exchange_strong(expected, 4, cuda::std::memory_order_seq_cst));
  assert(expected == 3);

  expected = 3;
  while (!atom.compare_exchange_weak(expected, 4, cuda::std::memory_order_relaxed))
  {
  }
  assert(atom.load(cuda::std::memory_order_relaxed) == 4);
}

int main(int, char**)
{
  test<cuda::std::atomic_ref<volatile int>>();
  test<cuda::atomic_ref<volatile int, cuda::thread_scope_system>>();
  test<cuda::atomic_ref<volatile int, cuda::thread_scope_device>>();
  test<cuda::atomic_ref<volatile int, cuda::thread_scope_block>>();

  return 0;
}
