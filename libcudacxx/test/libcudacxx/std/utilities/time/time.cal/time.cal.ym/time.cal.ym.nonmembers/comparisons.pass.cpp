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

// constexpr bool operator==(const year_month& x, const year_month& y) noexcept;
//   Returns: x.year() == y.year() && x.month() == y.month().
//
// constexpr bool operator< (const year_month& x, const year_month& y) noexcept;
//   Returns:
//      If x.year() < y.year() returns true.
//      Otherwise, if x.year() > y.year() returns false.
//      Otherwise, returns x.month() < y.month().

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using year       = cuda::std::chrono::year;
  using month      = cuda::std::chrono::month;
  using year_month = cuda::std::chrono::year_month;

  AssertComparisonsAreNoexcept<year_month>();
  AssertComparisonsReturnBool<year_month>();

  auto constexpr January  = cuda::std::chrono::January;
  auto constexpr February = cuda::std::chrono::February;

  static_assert(testComparisons(year_month{year{1234}, January}, year_month{year{1234}, January}, true, false), "");

  static_assert(testComparisons(year_month{year{1234}, January}, year_month{year{1234}, February}, false, true), "");

  static_assert(testComparisons(year_month{year{1234}, January}, year_month{year{1235}, January}, false, true), "");

  //  same year, different months
  for (unsigned i = 1; i < 12; ++i)
  {
    for (unsigned j = 1; j < 12; ++j)
    {
      assert((testComparisons(year_month{year{1234}, month{i}}, year_month{year{1234}, month{j}}, i == j, i < j)));
    }
  }

  //  same month, different years
  for (int i = 1000; i < 20; ++i)
  {
    for (int j = 1000; j < 20; ++j)
    {
      assert((testComparisons(year_month{year{i}, January}, year_month{year{j}, January}, i == j, i < j)));
    }
  }

  return 0;
}
