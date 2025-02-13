//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// duration& operator%=(const rep& rhs)

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

__host__ __device__ constexpr bool test_constexpr()
{
  cuda::std::chrono::seconds s(11);
  s %= 3;
  return s.count() == 2;
}

int main(int, char**)
{
  {
    cuda::std::chrono::microseconds us(11);
    us %= 3;
    assert(us.count() == 2);
  }

  static_assert(test_constexpr(), "");

  return 0;
}
