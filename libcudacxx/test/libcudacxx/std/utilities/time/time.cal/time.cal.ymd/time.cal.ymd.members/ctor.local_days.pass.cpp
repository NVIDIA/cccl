//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year_month_day;

//  explicit constexpr year_month_day(const local_days& dp) noexcept;
//
//
//  Effects:  Constructs an object of type year_month_day that corresponds
//                to the date represented by dp
//
//  Remarks: Equivalent to constructing with sys_days{dp.time_since_epoch()}.
//
//  constexpr chrono::year   year() const noexcept;
//  constexpr chrono::month month() const noexcept;
//  constexpr chrono::day     day() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

// MSVC warns about unsigned/signed comparisons and addition/subtraction
// Silence these warnings, but not the ones within the header itself.
#if defined(_MSC_VER)
#  pragma warning(disable : 4307)
#  pragma warning(disable : 4308)
#endif

int main(int, char**)
{
  using year           = cuda::std::chrono::year;
  using day            = cuda::std::chrono::day;
  using local_days     = cuda::std::chrono::local_days;
  using days           = cuda::std::chrono::days;
  using year_month_day = cuda::std::chrono::year_month_day;

  ASSERT_NOEXCEPT(year_month_day{cuda::std::declval<local_days>()});

  auto constexpr January = cuda::std::chrono::January;

  {
    constexpr local_days sd{};
    constexpr year_month_day ymd{sd};

    static_assert(ymd.ok(), "");
    static_assert(ymd.year() == year{1970}, "");
    static_assert(ymd.month() == January, "");
    static_assert(ymd.day() == day{1}, "");
  }

  {
    constexpr local_days sd{days{10957 + 32}};
    constexpr year_month_day ymd{sd};

    auto constexpr February = cuda::std::chrono::February;

    static_assert(ymd.ok(), "");
    static_assert(ymd.year() == year{2000}, "");
    static_assert(ymd.month() == February, "");
    static_assert(ymd.day() == day{2}, "");
  }

  //  There's one more leap day between 1/1/40 and 1/1/70
  //  when compared to 1/1/70 -> 1/1/2000
  {
    constexpr local_days sd{days{-10957}};
    constexpr year_month_day ymd{sd};

    static_assert(ymd.ok(), "");
    static_assert(ymd.year() == year{1940}, "");
    static_assert(ymd.month() == January, "");
    static_assert(ymd.day() == day{2}, "");
  }

  {
    local_days sd{days{-(10957 + 34)}};
    year_month_day ymd{sd};
    auto constexpr November = cuda::std::chrono::November;

    assert(ymd.ok());
    assert(ymd.year() == year{1939});
    assert(ymd.month() == November);
    assert(ymd.day() == day{29});
  }

  return 0;
}
