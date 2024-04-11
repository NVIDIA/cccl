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

// constexpr bool operator==(const month_weekday& x, const month_weekday& y) noexcept;
//   Returns: x.month() == y.month() && x.day() == y.day().
//

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using month_weekday   = cuda::std::chrono::month_weekday;
  using month           = cuda::std::chrono::month;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;
  using weekday         = cuda::std::chrono::weekday;

  constexpr weekday Sunday = cuda::std::chrono::Sunday;
  constexpr weekday Monday = cuda::std::chrono::Monday;

  AssertEqualityAreNoexcept<month_weekday>();
  AssertEqualityReturnBool<month_weekday>();

  static_assert(testEquality(month_weekday{cuda::std::chrono::January, weekday_indexed{Sunday, 1}},
                             month_weekday{cuda::std::chrono::January, weekday_indexed{Sunday, 1}},
                             true),
                "");

  static_assert(testEquality(month_weekday{cuda::std::chrono::January, weekday_indexed{Sunday, 1}},
                             month_weekday{cuda::std::chrono::January, weekday_indexed{Sunday, 2}},
                             false),
                "");

  static_assert(testEquality(month_weekday{cuda::std::chrono::January, weekday_indexed{Sunday, 1}},
                             month_weekday{cuda::std::chrono::February, weekday_indexed{Sunday, 1}},
                             false),
                "");

  static_assert(testEquality(month_weekday{cuda::std::chrono::January, weekday_indexed{Monday, 1}},
                             month_weekday{cuda::std::chrono::January, weekday_indexed{Sunday, 2}},
                             false),
                "");

  static_assert(testEquality(month_weekday{cuda::std::chrono::January, weekday_indexed{Monday, 1}},
                             month_weekday{cuda::std::chrono::February, weekday_indexed{Sunday, 1}},
                             false),
                "");

  //  same day, different months
  for (unsigned i = 1; i < 12; ++i)
  {
    for (unsigned j = 1; j < 12; ++j)
    {
      assert((testEquality(month_weekday{month{i}, weekday_indexed{Sunday, 1}},
                           month_weekday{month{j}, weekday_indexed{Sunday, 1}},
                           i == j)));
    }
  }

  //  same month, different weeks
  for (unsigned i = 1; i < 5; ++i)
  {
    for (unsigned j = 1; j < 5; ++j)
    {
      assert((testEquality(month_weekday{month{2}, weekday_indexed{Sunday, i}},
                           month_weekday{month{2}, weekday_indexed{Sunday, j}},
                           i == j)));
    }
  }

  //  same month, different days
  for (unsigned i = 0; i < 6; ++i)
  {
    for (unsigned j = 0; j < 6; ++j)
    {
      assert((testEquality(month_weekday{month{2}, weekday_indexed{weekday{i}, 2}},
                           month_weekday{month{2}, weekday_indexed{weekday{j}, 2}},
                           i == j)));
    }
  }

  return 0;
}
