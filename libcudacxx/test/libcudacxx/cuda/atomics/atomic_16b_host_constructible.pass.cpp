//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: windows

// <cuda/atomic>

#include <cuda/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

// Check that atomics on host may be constructed
template <class T>
__host__ __device__ void do_test()
{
  T v(0);
  cuda::atomic_ref<T> a(v);
}

int main(int, char**)
{
  do_test<__int128_t>();
  do_test<__uint128_t>();
  return 0;
}
