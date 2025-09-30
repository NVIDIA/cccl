//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// ceil

// template <cuda/std/class ToDuration, class Clock, class Duration>
//   time_point<cuda/std/Clock, ToDuration>
//   ceil(const time_point<cuda/std/Clock, Duration>& t);

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class FromDuration, class ToDuration>
__host__ __device__ constexpr void test(const FromDuration& df, const ToDuration& d)
{
  using Clock         = cuda::std::chrono::system_clock;
  using FromTimePoint = cuda::std::chrono::time_point<Clock, FromDuration>;
  using ToTimePoint   = cuda::std::chrono::time_point<Clock, ToDuration>;

  FromTimePoint f(df);
  ToTimePoint t(d);
  using R = decltype(cuda::std::chrono::ceil<ToDuration>(f));
  static_assert(cuda::std::is_same_v<R, ToTimePoint>);
  assert(cuda::std::chrono::ceil<ToDuration>(f) == t);
}

__host__ __device__ constexpr bool test()
{
  //  7290000ms is 2 hours, 1 minute, and 30 seconds
  test(cuda::std::chrono::milliseconds(7290000), cuda::std::chrono::hours(3));
  test(cuda::std::chrono::milliseconds(-7290000), cuda::std::chrono::hours(-2));
  test(cuda::std::chrono::milliseconds(7290000), cuda::std::chrono::minutes(122));
  test(cuda::std::chrono::milliseconds(-7290000), cuda::std::chrono::minutes(-121));

  //  9000000ms is 2 hours and 30 minutes
  test(cuda::std::chrono::milliseconds(9000000), cuda::std::chrono::hours(3));
  test(cuda::std::chrono::milliseconds(-9000000), cuda::std::chrono::hours(-2));
  test(cuda::std::chrono::milliseconds(9000001), cuda::std::chrono::minutes(151));
  test(cuda::std::chrono::milliseconds(-9000001), cuda::std::chrono::minutes(-150));

  test(cuda::std::chrono::milliseconds(9000000), cuda::std::chrono::seconds(9000));
  test(cuda::std::chrono::milliseconds(-9000000), cuda::std::chrono::seconds(-9000));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
