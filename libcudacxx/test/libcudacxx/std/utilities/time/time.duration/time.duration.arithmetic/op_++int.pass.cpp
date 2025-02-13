//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// constexpr duration operator++(int);  // constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

__host__ __device__ constexpr bool test_constexpr()
{
  cuda::std::chrono::hours h1(3);
  cuda::std::chrono::hours h2 = h1++;
  return h1.count() == 4 && h2.count() == 3;
}

int main(int, char**)
{
  {
    cuda::std::chrono::hours h1(3);
    cuda::std::chrono::hours h2 = h1++;
    assert(h1.count() == 4);
    assert(h2.count() == 3);
  }

  static_assert(test_constexpr(), "");

  return 0;
}
