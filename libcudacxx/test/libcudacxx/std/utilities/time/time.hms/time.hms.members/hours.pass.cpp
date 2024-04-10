//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11
// <chrono>

// template <class Duration>
// class hh_mm_ss
//
// constexpr chrono::hours hours() const noexcept;

// Test values
// duration     hours   minutes seconds fractional
// 5000s            1       23      20      0
// 5000 minutes     83      20      0       0
// 123456789ms      34      17      36      789ms
// 123456789us      0       2       3       456789us
// 123456789ns      0       0       0       123456789ns
// 1000mfn          0       20      9       0.6 (6000/10000)
// 10000mfn         3       21      36      0

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

template <typename Duration>
__host__ __device__ constexpr long check_hours(Duration d)
{
  using HMS = cuda::std::chrono::hh_mm_ss<Duration>;
  ASSERT_SAME_TYPE(cuda::std::chrono::hours, decltype(cuda::std::declval<HMS>().hours()));
  ASSERT_NOEXCEPT(cuda::std::declval<HMS>().hours());
  return HMS(d).hours().count();
}

int main(int, char**)
{
  using microfortnights = cuda::std::chrono::duration<int, cuda::std::ratio<756, 625>>;

  static_assert(check_hours(cuda::std::chrono::minutes(1)) == 0, "");
  static_assert(check_hours(cuda::std::chrono::minutes(-1)) == 0, "");

  assert(check_hours(cuda::std::chrono::seconds(5000)) == 1);
  assert(check_hours(cuda::std::chrono::seconds(-5000)) == 1);
  assert(check_hours(cuda::std::chrono::minutes(5000)) == 83);
  assert(check_hours(cuda::std::chrono::minutes(-5000)) == 83);
  assert(check_hours(cuda::std::chrono::hours(11)) == 11);
  assert(check_hours(cuda::std::chrono::hours(-11)) == 11);

  assert(check_hours(cuda::std::chrono::milliseconds(123456789LL)) == 34);
  assert(check_hours(cuda::std::chrono::milliseconds(-123456789LL)) == 34);
  assert(check_hours(cuda::std::chrono::microseconds(123456789LL)) == 0);
  assert(check_hours(cuda::std::chrono::microseconds(-123456789LL)) == 0);
  assert(check_hours(cuda::std::chrono::nanoseconds(123456789LL)) == 0);
  assert(check_hours(cuda::std::chrono::nanoseconds(-123456789LL)) == 0);

  assert(check_hours(microfortnights(1000)) == 0);
  assert(check_hours(microfortnights(-1000)) == 0);
  assert(check_hours(microfortnights(10000)) == 3);
  assert(check_hours(microfortnights(-10000)) == 3);

  return 0;
}
