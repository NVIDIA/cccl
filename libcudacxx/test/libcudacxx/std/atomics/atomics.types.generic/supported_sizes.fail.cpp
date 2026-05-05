//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: asm statement is unsupported in tile code

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

struct TooLarge
{
  int32_t v[6];
};

template <typename T>
TEST_FUNC void check_supported_type(T v)
{
  cuda::std::atomic<T> atom(v);
  cuda::std::atomic_ref<T> ref(v);
}

int main(int, char**)
{
  check_supported_type(TooLarge{});

  return 0;
}
