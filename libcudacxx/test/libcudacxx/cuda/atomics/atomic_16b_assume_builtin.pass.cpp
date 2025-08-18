//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// XFAIL: gcc
// UNSUPPORTED: pre-sm-90
// UNSUPPORTED: windows
// UNSUPPORTED: aarch64-unknown-linux-gnu

// <cuda/atomic>

#include <cuda/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

// This test specifically checks that GCC triggers a static assertion when detecting non-builtin use of 128b sized
// atomics.
template <class T>
__host__ __device__ void do_test()
{
  T v(0);
  cuda::atomic_ref<T> a(v);
  a.store(1);
  assert(a++ == 1);
  assert(a.load() == 2);
}

int main(int, char**)
{
  do_test<__int128_t>();
  do_test<__uint128_t>();
  return 0;
}
