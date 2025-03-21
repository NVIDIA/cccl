//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// template <class Duration>
// class hh_mm_ss
//
// constexpr bool is_negative() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

template <typename Duration>
__host__ __device__ constexpr bool check_neg(Duration d)
{
  static_assert(
    cuda::std::is_same_v<bool, decltype(cuda::std::declval<cuda::std::chrono::hh_mm_ss<Duration>>().is_negative())>);
  static_assert(noexcept(cuda::std::declval<cuda::std::chrono::hh_mm_ss<Duration>>().is_negative()));
  return cuda::std::chrono::hh_mm_ss<Duration>(d).is_negative();
}

int main(int, char**)
{
  using microfortnights = cuda::std::chrono::duration<int, cuda::std::ratio<756, 625>>;

  static_assert(!check_neg(cuda::std::chrono::minutes(1)), "");
  static_assert(check_neg(cuda::std::chrono::minutes(-1)), "");

  assert(!check_neg(cuda::std::chrono::seconds(5000)));
  assert(check_neg(cuda::std::chrono::seconds(-5000)));
  assert(!check_neg(cuda::std::chrono::minutes(5000)));
  assert(check_neg(cuda::std::chrono::minutes(-5000)));
  assert(!check_neg(cuda::std::chrono::hours(11)));
  assert(check_neg(cuda::std::chrono::hours(-11)));

  assert(!check_neg(cuda::std::chrono::milliseconds(123456789LL)));
  assert(check_neg(cuda::std::chrono::milliseconds(-123456789LL)));
  assert(!check_neg(cuda::std::chrono::microseconds(123456789LL)));
  assert(check_neg(cuda::std::chrono::microseconds(-123456789LL)));
  assert(!check_neg(cuda::std::chrono::nanoseconds(123456789LL)));
  assert(check_neg(cuda::std::chrono::nanoseconds(-123456789LL)));

  assert(!check_neg(microfortnights(10000)));
  assert(check_neg(microfortnights(-10000)));

  return 0;
}
