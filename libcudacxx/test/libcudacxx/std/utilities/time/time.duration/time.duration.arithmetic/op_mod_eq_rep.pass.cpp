//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// duration& operator%=(const rep& rhs)

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

class NotARep
{};

using Duration = cuda::std::chrono::seconds;
__host__ __device__ Duration operator%=(Duration d, NotARep)
{
  return d;
}

__host__ __device__ constexpr bool test()
{
  cuda::std::chrono::microseconds us(11);
  us %= 3;
  assert(us.count() == 2);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  { // This is PR#41130
    Duration d(5);
    NotARep n;
    d %= n;
    assert(d.count() == 5);
  }

  return 0;
}
