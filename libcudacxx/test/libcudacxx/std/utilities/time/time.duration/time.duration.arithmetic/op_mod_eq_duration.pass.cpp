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

// duration& operator%=(const duration& rhs)

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "../../rep.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  cuda::std::chrono::microseconds us1(11);
  cuda::std::chrono::microseconds us2(3);
  us1 %= us2;
  assert(us1.count() == 2);

  if (!cuda::std::is_constant_evaluated())
  {
    us1 %= cuda::std::chrono::milliseconds(3);
    assert(us1.count() == 2);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  { // This is PR#41130
    cuda::std::chrono::nanoseconds d(5);
    NotARep n;
    d %= n;
    assert(d.count() == 5);
  }

  return 0;
}
