//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: windows
// ADDITIONAL_COMPILE_OPTIONS_HOST: -mcx16
// UNSUPPORTED: aarch64-unknown-linux-gnu
//
// <cuda/atomic>

#define LIBCUDACXX_IGNORE_MISSING_BUILTIN_128_ATOMICS
#include <cuda/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

// This test specifically checks load/store operations are supported with i128.
template <class T>
__host__ __device__ void do_test()
{
  T v(0);
  cuda::atomic_ref<T> a(v);
  a.store(1);
  assert(a.load() == 1);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (do_test<__int128_t>(); do_test<__uint128_t>();))
  return 0;
}
