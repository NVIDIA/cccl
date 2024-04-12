//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// constexpr duration operator--(int);   // constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

#if TEST_STD_VER > 2014
__host__ __device__ constexpr bool test_constexpr()
{
  cuda::std::chrono::hours h1(3);
  cuda::std::chrono::hours h2 = h1--;
  return h1.count() == 2 && h2.count() == 3;
}
#endif

int main(int, char**)
{
  {
    cuda::std::chrono::hours h1(3);
    cuda::std::chrono::hours h2 = h1--;
    assert(h1.count() == 2);
    assert(h2.count() == 3);
  }

#if TEST_STD_VER > 2014
  static_assert(test_constexpr(), "");
#endif

  return 0;
}
