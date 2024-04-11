//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year_month;

// constexpr year_month operator/(const year& y, const month& m) noexcept;
//   Returns: {y, m}.
//
// constexpr year_month operator/(const year& y, int m) noexcept;
//   Returns: y / month(m).

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using month      = cuda::std::chrono::month;
  using year       = cuda::std::chrono::year;
  using year_month = cuda::std::chrono::year_month;

  constexpr month February = cuda::std::chrono::February;

  { // operator/(const year& y, const month& m)
    ASSERT_NOEXCEPT(year{2018} / February);
    ASSERT_SAME_TYPE(year_month, decltype(year{2018} / February));

    static_assert((year{2018} / February).year() == year{2018}, "");
    static_assert((year{2018} / February).month() == month{2}, "");
    for (int i = 1000; i <= 1030; ++i)
    {
      for (unsigned j = 1; j <= 12; ++j)
      {
        year_month ym = year{i} / month{j};
        assert(static_cast<int>(ym.year()) == i);
        assert(static_cast<unsigned>(ym.month()) == j);
      }
    }
  }

  { // operator/(const year& y, const int m)
    ASSERT_NOEXCEPT(year{2018} / 4);
    ASSERT_SAME_TYPE(year_month, decltype(year{2018} / 4));

    static_assert((year{2018} / 2).year() == year{2018}, "");
    static_assert((year{2018} / 2).month() == month{2}, "");

    for (int i = 1000; i <= 1030; ++i)
    {
      for (unsigned j = 1; j <= 12; ++j)
      {
        year_month ym = year{i} / j;
        assert(static_cast<int>(ym.year()) == i);
        assert(static_cast<unsigned>(ym.month()) == j);
      }
    }
  }

  return 0;
}
