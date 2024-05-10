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
// XFAIL: c++98, c++03

// <cuda/std/atomic>

// #define LIBCUDACXX_ATOMIC_VAR_INIT(value)

#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  cuda::std::atomic<int> v = LIBCUDACXX_ATOMIC_VAR_INIT(5);
  assert(v == 5);

  return 0;
}
