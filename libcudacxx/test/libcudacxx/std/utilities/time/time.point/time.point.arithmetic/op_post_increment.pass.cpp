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

// constexpr time_point operator++(int);

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using Clock    = cuda::std::chrono::system_clock;
  using Duration = cuda::std::chrono::milliseconds;
  cuda::std::chrono::time_point<Clock, Duration> t1{Duration{3}};
  cuda::std::chrono::time_point<Clock, Duration> t2{t1++};
  assert(t1.time_since_epoch() == Duration{4});
  assert(t2.time_since_epoch() == Duration{3});
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
