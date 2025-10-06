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

// template <cuda/std/class Clock, class Duration1, class Rep2, class Period2>
//   time_point<cuda/std/Clock, typename common_type<Duration1, duration<Rep2, Period2>>::type>
//   operator+(const time_point<cuda/std/Clock, Duration1>& lhs, const duration<Rep2, Period2>& rhs);

// template <cuda/std/class Rep1, class Period1, class Clock, class Duration2>
//   time_point<cuda/std/Clock, typename common_type<duration<Rep1, Period1>, Duration2>::type>
//   operator+(const duration<cuda/std/Rep1, Period1>& lhs, const time_point<Clock, Duration2>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using Clock     = cuda::std::chrono::system_clock;
  using Duration1 = cuda::std::chrono::milliseconds;
  using Duration2 = cuda::std::chrono::microseconds;

  cuda::std::chrono::time_point<Clock, Duration1> t1(Duration1(3));
  cuda::std::chrono::time_point<Clock, Duration2> t2 = t1 + Duration2(5);
  assert(t2.time_since_epoch() == Duration2(3005));
  t2 = Duration2(6) + t1;
  assert(t2.time_since_epoch() == Duration2(3006));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
