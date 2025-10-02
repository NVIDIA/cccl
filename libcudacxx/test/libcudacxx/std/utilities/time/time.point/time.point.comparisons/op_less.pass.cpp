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

// template <cuda/std/class Clock, class Duration1, class Duration2>
//   bool
//   operator<cuda/std/ (const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <cuda/std/class Clock, class Duration1, class Duration2>
//   bool
//   operator> (const time_point<cuda/std/Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <cuda/std/class Clock, class Duration1, class Duration2>
//   bool
//   operator<cuda/std/=(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <cuda/std/class Clock, class Duration1, class Duration2>
//   bool
//   operator>=(const time_point<cuda/std/Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using Clock     = cuda::std::chrono::system_clock;
  using Duration1 = cuda::std::chrono::milliseconds;
  using Duration2 = cuda::std::chrono::microseconds;
  using T1        = cuda::std::chrono::time_point<Clock, Duration1>;
  using T2        = cuda::std::chrono::time_point<Clock, Duration2>;

  {
    T1 t1(Duration1(3));
    T1 t2(Duration1(3));
    assert(!(t1 < t2));
    assert(!(t1 > t2));
    assert((t1 <= t2));
    assert((t1 >= t2));
  }
  {
    T1 t1(Duration1(3));
    T1 t2(Duration1(4));
    assert((t1 < t2));
    assert(!(t1 > t2));
    assert((t1 <= t2));
    assert(!(t1 >= t2));
  }
  {
    T1 t1(Duration1(3));
    T2 t2(Duration2(3000));
    assert(!(t1 < t2));
    assert(!(t1 > t2));
    assert((t1 <= t2));
    assert((t1 >= t2));
  }
  {
    T1 t1(Duration1(3));
    T2 t2(Duration2(3001));
    assert((t1 < t2));
    assert(!(t1 > t2));
    assert((t1 <= t2));
    assert(!(t1 >= t2));
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
