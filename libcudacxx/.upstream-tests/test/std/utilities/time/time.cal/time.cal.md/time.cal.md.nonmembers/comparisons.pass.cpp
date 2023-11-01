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

// constexpr bool operator==(const month_day& x, const month_day& y) noexcept;
//   Returns: x.month() == y.month() && x.day() == y.day().
//
// constexpr bool operator< (const month_day& x, const month_day& y) noexcept;
//   Returns:
//      If x.month() < y.month() returns true.
//      Otherwise, if x.month() > y.month() returns false.
//      Otherwise, returns x.day() < y.day().

#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using day       = cuda::std::chrono::day;
    using month     = cuda::std::chrono::month;
    using month_day = cuda::std::chrono::month_day;

    AssertComparisons6AreNoexcept<month_day>();
    AssertComparisons6ReturnBool<month_day>();

    static_assert( testComparisons6(
        month_day{cuda::std::chrono::January, day{1}},
        month_day{cuda::std::chrono::January, day{1}},
        true, false), "");

    static_assert( testComparisons6(
        month_day{cuda::std::chrono::January, day{1}},
        month_day{cuda::std::chrono::January, day{2}},
        false, true), "");

    static_assert( testComparisons6(
        month_day{cuda::std::chrono::January,  day{1}},
        month_day{cuda::std::chrono::February, day{1}},
        false, true), "");

//  same day, different months
    for (unsigned i = 1; i < 12; ++i)
        for (unsigned j = 1; j < 12; ++j)
            assert((testComparisons6(
                month_day{month{i}, day{1}},
                month_day{month{j}, day{1}},
                i == j, i < j )));

//  same month, different days
    for (unsigned i = 1; i < 31; ++i)
        for (unsigned j = 1; j < 31; ++j)
            assert((testComparisons6(
                month_day{month{2}, day{i}},
                month_day{month{2}, day{j}},
                i == j, i < j )));


  return 0;
}
