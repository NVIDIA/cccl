//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class month_weekday;

// constexpr bool ok() const noexcept;
//  Returns: m_.ok() && wdi_.ok().

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month_weekday   = cuda::std::chrono::month_weekday;
  using month           = cuda::std::chrono::month;
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  constexpr weekday Sunday = cuda::std::chrono::Sunday;

  ASSERT_NOEXCEPT(cuda::std::declval<const month_weekday>().ok());
  ASSERT_SAME_TYPE(bool, decltype(cuda::std::declval<const month_weekday>().ok()));

  static_assert(!month_weekday{month{}, weekday_indexed{}}.ok(), "");
  static_assert(month_weekday{cuda::std::chrono::May, weekday_indexed{Sunday, 2}}.ok(), "");

  assert(!(month_weekday(cuda::std::chrono::April, weekday_indexed{Sunday, 0}).ok()));
  assert((month_weekday{cuda::std::chrono::March, weekday_indexed{Sunday, 1}}.ok()));

  for (unsigned i = 1; i <= 12; ++i)
  {
    for (unsigned j = 0; j <= 6; ++j)
    {
      month_weekday mwd{month{i}, weekday_indexed{Sunday, j}};
      assert(mwd.ok() == (j >= 1 && j <= 5));
    }
  }

  //  If the month is not ok, all the weekday_indexed are bad
  for (unsigned i = 1; i <= 10; ++i)
  {
    assert(!(month_weekday{month{13}, weekday_indexed{Sunday, i}}.ok()));
  }

  return 0;
}
