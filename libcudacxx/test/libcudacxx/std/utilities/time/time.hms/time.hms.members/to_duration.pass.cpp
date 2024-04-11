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
// constexpr precision to_duration() const noexcept;
//
// See the table in hours.pass.cpp for correspondence between the magic values used below

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

template <typename Duration>
__host__ __device__ constexpr long long check_duration(Duration d)
{
  using HMS = cuda::std::chrono::hh_mm_ss<Duration>;
  ASSERT_SAME_TYPE(typename HMS::precision, decltype(cuda::std::declval<HMS>().to_duration()));
  ASSERT_NOEXCEPT(cuda::std::declval<HMS>().to_duration());

  return HMS(d).to_duration().count();
}

int main(int, char**)
{
  using microfortnights = cuda::std::chrono::duration<int, cuda::std::ratio<756, 625>>;

  static_assert(check_duration(cuda::std::chrono::minutes(1)) == 60, "");
  static_assert(check_duration(cuda::std::chrono::minutes(-1)) == -60, "");

  assert(check_duration(cuda::std::chrono::seconds(5000)) == 5000LL);
  assert(check_duration(cuda::std::chrono::seconds(-5000)) == -5000LL);
  assert(check_duration(cuda::std::chrono::minutes(5000)) == 300000LL);
  assert(check_duration(cuda::std::chrono::minutes(-5000)) == -300000LL);
  assert(check_duration(cuda::std::chrono::hours(11)) == 39600LL);
  assert(check_duration(cuda::std::chrono::hours(-11)) == -39600LL);

  assert(check_duration(cuda::std::chrono::milliseconds(123456789LL)) == 123456789LL);
  assert(check_duration(cuda::std::chrono::milliseconds(-123456789LL)) == -123456789LL);
  assert(check_duration(cuda::std::chrono::microseconds(123456789LL)) == 123456789LL);
  assert(check_duration(cuda::std::chrono::microseconds(-123456789LL)) == -123456789LL);
  assert(check_duration(cuda::std::chrono::nanoseconds(123456789LL)) == 123456789LL);
  assert(check_duration(cuda::std::chrono::nanoseconds(-123456789LL)) == -123456789LL);

  assert(check_duration(microfortnights(1000)) == 12096000);
  assert(check_duration(microfortnights(-1000)) == -12096000);
  assert(check_duration(microfortnights(10000)) == 120960000);
  assert(check_duration(microfortnights(-10000)) == -120960000);

  return 0;
}
