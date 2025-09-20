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

// template <cuda/std/class ToDuration, class Clock, class Duration>
//   time_point<cuda/std/Clock, ToDuration>
//   time_point_cast(const time_point<cuda/std/Clock, Duration>& t);

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/ratio>
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
  using R = decltype(cuda::std::chrono::time_point_cast<ToDuration>(f));
  static_assert(cuda::std::is_same_v<R, ToTimePoint>);
  assert(cuda::std::chrono::time_point_cast<ToDuration>(f) == t);
}

__host__ __device__ constexpr bool test()
{
  test(cuda::std::chrono::milliseconds(7265000), cuda::std::chrono::hours(2));
  test(cuda::std::chrono::milliseconds(7265000), cuda::std::chrono::minutes(121));
  test(cuda::std::chrono::milliseconds(7265000), cuda::std::chrono::seconds(7265));
  test(cuda::std::chrono::milliseconds(7265000), cuda::std::chrono::milliseconds(7265000));
  test(cuda::std::chrono::milliseconds(7265000), cuda::std::chrono::microseconds(7265000000LL));
  test(cuda::std::chrono::milliseconds(7265000), cuda::std::chrono::nanoseconds(7265000000000LL));
  test(cuda::std::chrono::milliseconds(7265000),
       cuda::std::chrono::duration<double, cuda::std::ratio<3600>>(7265. / 3600));
  test(cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>>(9),
       cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>>(10));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
