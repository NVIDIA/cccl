//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: true

// <cuda/std/chrono>

// time_point

// template<class Clock, class Duration1,
//          three_way_comparable_with<cuda/std/Duration1> Duration2>
//   constexpr auto operator<cuda/std/=>(const time_point<Clock, Duration1>& lhs,
//                              const time_point<cuda/std/Clock, Duration2>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/ratio>

#include "test_comparisons.h"

__host__ __device__ constexpr void test_with_integral_ticks_value()
{
  using Clock = cuda::std::chrono::system_clock;

  using Duration1 = cuda::std::chrono::milliseconds;
  using Duration2 = cuda::std::chrono::microseconds;
  using T1        = cuda::std::chrono::time_point<Clock, Duration1>;
  using T2        = cuda::std::chrono::time_point<Clock, Duration2>;

  {
    T1 t1(Duration1(3));
    T1 t2(Duration1(3));
    assert((t1 <=> t2) == cuda::std::strong_ordering::equal);
    assert(testOrder(t1, t2, cuda::std::strong_ordering::equal));
  }
  {
    T1 t1(Duration1(3));
    T1 t2(Duration1(4));
    assert((t1 <=> t2) == cuda::std::strong_ordering::less);
    assert(testOrder(t1, t2, cuda::std::strong_ordering::less));
  }
  {
    T1 t1(Duration1(3));
    T2 t2(Duration2(3000));
    assert((t1 <=> t2) == cuda::std::strong_ordering::equal);
    assert(testOrder(t1, t2, cuda::std::strong_ordering::equal));
  }
  {
    T1 t1(Duration1(3));
    T2 t2(Duration2(3001));
    assert((t1 <=> t2) == cuda::std::strong_ordering::less);
    assert(testOrder(t1, t2, cuda::std::strong_ordering::less));
    assert((t2 <=> t1) == cuda::std::strong_ordering::greater);
    assert(testOrder(t2, t1, cuda::std::strong_ordering::greater));
  }
}

__host__ __device__ constexpr void test_with_integral_ticks_value_and_custom_period_value()
{
  using Clock = cuda::std::chrono::system_clock;

  using DInt30Hz = cuda::std::chrono::duration<int, cuda::std::ratio<1, 30>>;
  using DInt60Hz = cuda::std::chrono::duration<int, cuda::std::ratio<1, 60>>;

  using TIntR1 = cuda::std::chrono::time_point<Clock, DInt30Hz>;
  using TIntR2 = cuda::std::chrono::time_point<Clock, DInt60Hz>;

  {
    TIntR1 t1(DInt30Hz(10));
    TIntR2 t2(DInt60Hz(20));
    assert((t1 <=> t2) == cuda::std::strong_ordering::equal);
    assert(testOrder(t1, t2, cuda::std::strong_ordering::equal));
  }
  {
    TIntR1 t1(DInt30Hz(10));
    TIntR2 t2(DInt60Hz(21));
    assert((t1 <=> t2) == cuda::std::strong_ordering::less);
    assert(testOrder(t1, t2, cuda::std::strong_ordering::less));
  }
  {
    TIntR1 t1(DInt30Hz(11));
    TIntR2 t2(DInt60Hz(20));
    assert((t1 <=> t2) == cuda::std::strong_ordering::greater);
    assert(testOrder(t1, t2, cuda::std::strong_ordering::greater));
  }
}

__host__ __device__ constexpr void test_with_floating_point_ticks_value()
{
  using Clock = cuda::std::chrono::system_clock;

  using DF30Hz = cuda::std::chrono::duration<double, cuda::std::ratio<1, 30>>;
  using DF60Hz = cuda::std::chrono::duration<double, cuda::std::ratio<1, 60>>;
  using F1     = cuda::std::chrono::time_point<Clock, DF30Hz>;
  using F2     = cuda::std::chrono::time_point<Clock, DF60Hz>;

  // No equality comparison test for floating point values.

  {
    F1 t1(DF30Hz(3.5));
    F2 t2(DF60Hz(7.1));
    assert((t1 <=> t2) == cuda::std::weak_ordering::less);
    assert(testOrder(t1, t2, cuda::std::weak_ordering::less));
  }
  {
    F1 t1(DF30Hz(3.6));
    F2 t2(DF60Hz(7.0));
    assert((t1 <=> t2) == cuda::std::weak_ordering::greater);
    assert(testOrder(t1, t2, cuda::std::weak_ordering::greater));
  }
}

__host__ __device__ constexpr bool test()
{
  test_with_integral_ticks_value();
  test_with_integral_ticks_value_and_custom_period_value();
  test_with_floating_point_ticks_value();

  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
