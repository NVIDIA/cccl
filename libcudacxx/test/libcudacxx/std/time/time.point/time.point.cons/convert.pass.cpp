//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// template <cuda/std/class Duration2>
//   time_point(const time_point<cuda/std/clock, Duration2>& t);

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using Clock     = cuda::std::chrono::system_clock;
  using Duration1 = cuda::std::chrono::microseconds;
  using Duration2 = cuda::std::chrono::milliseconds;

  cuda::std::chrono::time_point<Clock, Duration2> t2(Duration2(3));
  cuda::std::chrono::time_point<Clock, Duration1> t1 = t2;
  assert(t1.time_since_epoch() == Duration1(3000));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
