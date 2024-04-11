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

// <cuda/std/atomic>

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"
#if !defined(TEST_COMPILER_MSVC)
#  include "placement_new.h"
#endif
#include "cuda_space_selector.h"

struct non_arithmetic
{
  int a;
};

int main(int, char**)
{
  cuda::std::atomic<non_arithmetic> a;
  a.fetch_add(non_arithmetic{0});
  a.fetch_sub(non_arithmetic{0});

  return 0;
}
