//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class month_day;

// constexpr bool ok() const noexcept;
//  Returns: true if m_.ok() is true, 1d <= d_, and d_ is less than or equal to the
//    number of days in month m_; otherwise returns false.
//  When m_ == February, the number of days is considered to be 29.

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using day       = cuda::std::chrono::day;
  using month     = cuda::std::chrono::month;
  using month_day = cuda::std::chrono::month_day;

  ASSERT_NOEXCEPT(cuda::std::declval<const month_day>().ok());
  ASSERT_SAME_TYPE(bool, decltype(cuda::std::declval<const month_day>().ok()));

  static_assert(!month_day{}.ok(), "");
  static_assert(month_day{cuda::std::chrono::May, day{2}}.ok(), "");

  assert(!(month_day(cuda::std::chrono::April, day{0}).ok()));

  assert((month_day{cuda::std::chrono::March, day{1}}.ok()));
  for (unsigned i = 1; i <= 12; ++i)
  {
    const bool is31 = i == 1 || i == 3 || i == 5 || i == 7 || i == 8 || i == 10 || i == 12;
    assert(!(month_day{month{i}, day{0}}.ok()));
    assert((month_day{month{i}, day{1}}.ok()));
    assert((month_day{month{i}, day{10}}.ok()));
    assert((month_day{month{i}, day{29}}.ok()));
    assert((month_day{month{i}, day{30}}.ok()) == (i != 2));
    assert((month_day{month{i}, day{31}}.ok()) == is31);
    assert(!(month_day{month{i}, day{32}}.ok()));
  }

  //  If the month is not ok, all the days are bad
  for (unsigned i = 1; i <= 35; ++i)
  {
    assert(!(month_day{month{13}, day{i}}.ok()));
  }

  return 0;
}
