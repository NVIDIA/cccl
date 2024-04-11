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
// constexpr chrono::minutes minutes() const noexcept;
//
// See the table in hours.pass.cpp for correspondence between the magic values used below

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

template <typename Duration>
__host__ __device__ constexpr long check_minutes(Duration d)
{
  using HMS = cuda::std::chrono::hh_mm_ss<Duration>;
  ASSERT_SAME_TYPE(cuda::std::chrono::minutes, decltype(cuda::std::declval<HMS>().minutes()));
  ASSERT_NOEXCEPT(cuda::std::declval<HMS>().minutes());
  return HMS(d).minutes().count();
}

int main(int, char**)
{
  using microfortnights = cuda::std::chrono::duration<int, cuda::std::ratio<756, 625>>;

  static_assert(check_minutes(cuda::std::chrono::minutes(1)) == 1, "");
  static_assert(check_minutes(cuda::std::chrono::minutes(-1)) == 1, "");

  assert(check_minutes(cuda::std::chrono::seconds(5000)) == 23);
  assert(check_minutes(cuda::std::chrono::seconds(-5000)) == 23);
  assert(check_minutes(cuda::std::chrono::minutes(5000)) == 20);
  assert(check_minutes(cuda::std::chrono::minutes(-5000)) == 20);
  assert(check_minutes(cuda::std::chrono::hours(11)) == 0);
  assert(check_minutes(cuda::std::chrono::hours(-11)) == 0);

  assert(check_minutes(cuda::std::chrono::milliseconds(123456789LL)) == 17);
  assert(check_minutes(cuda::std::chrono::milliseconds(-123456789LL)) == 17);
  assert(check_minutes(cuda::std::chrono::microseconds(123456789LL)) == 2);
  assert(check_minutes(cuda::std::chrono::microseconds(-123456789LL)) == 2);
  assert(check_minutes(cuda::std::chrono::nanoseconds(123456789LL)) == 0);
  assert(check_minutes(cuda::std::chrono::nanoseconds(-123456789LL)) == 0);

  assert(check_minutes(microfortnights(1000)) == 20);
  assert(check_minutes(microfortnights(-1000)) == 20);
  assert(check_minutes(microfortnights(10000)) == 21);
  assert(check_minutes(microfortnights(-10000)) == 21);

  return 0;
}
