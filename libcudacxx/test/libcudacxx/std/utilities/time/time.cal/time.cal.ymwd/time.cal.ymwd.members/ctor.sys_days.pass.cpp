//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year_month_weekday;

//  constexpr year_month_weekday(const sys_days& dp) noexcept;
//
//  Effects:  Constructs an object of type year_month_weekday that corresponds
//                to the date represented by dp
//
//  Remarks: For any value ymd of type year_month_weekday for which ymd.ok() is true,
//                ymd == year_month_weekday{sys_days{ymd}} is true.
//
//  constexpr chrono::year   year() const noexcept;
//  constexpr chrono::month month() const noexcept;
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
  using year               = cuda::std::chrono::year;
  using days               = cuda::std::chrono::days;
  using sys_days           = cuda::std::chrono::sys_days;
  using weekday_indexed    = cuda::std::chrono::weekday_indexed;
  using year_month_weekday = cuda::std::chrono::year_month_weekday;

  ASSERT_NOEXCEPT(year_month_weekday{cuda::std::declval<const sys_days>()});

  {
    constexpr sys_days sd{}; // 1-Jan-1970 was a Thursday
    constexpr year_month_weekday ymwd{sd};
    auto constexpr January  = cuda::std::chrono::January;
    auto constexpr Thursday = cuda::std::chrono::Thursday;

    static_assert(ymwd.ok(), "");
    static_assert(ymwd.year() == year{1970}, "");
    static_assert(ymwd.month() == January, "");
    static_assert(ymwd.weekday() == Thursday, "");
    static_assert(ymwd.index() == 1, "");
    static_assert(ymwd.weekday_indexed() == weekday_indexed{Thursday, 1}, "");
    static_assert(ymwd == year_month_weekday{sys_days{ymwd}}, ""); // round trip
  }

  {
    constexpr sys_days sd{days{10957 + 32}}; // 2-Feb-2000 was a Wednesday
    constexpr year_month_weekday ymwd{sd};
    auto constexpr February  = cuda::std::chrono::February;
    auto constexpr Wednesday = cuda::std::chrono::Wednesday;

    static_assert(ymwd.ok(), "");
    static_assert(ymwd.year() == year{2000}, "");
    static_assert(ymwd.month() == February, "");
    static_assert(ymwd.weekday() == Wednesday, "");
    static_assert(ymwd.index() == 1, "");
    static_assert(ymwd.weekday_indexed() == weekday_indexed{Wednesday, 1}, "");
    static_assert(ymwd == year_month_weekday{sys_days{ymwd}}, ""); // round trip
  }

  {
    constexpr sys_days sd{days{-10957}}; // 2-Jan-1940 was a Tuesday
    constexpr year_month_weekday ymwd{sd};
    auto constexpr January = cuda::std::chrono::January;
    auto constexpr Tuesday = cuda::std::chrono::Tuesday;

    static_assert(ymwd.ok(), "");
    static_assert(ymwd.year() == year{1940}, "");
    static_assert(ymwd.month() == January, "");
    static_assert(ymwd.weekday() == Tuesday, "");
    static_assert(ymwd.index() == 1, "");
    static_assert(ymwd.weekday_indexed() == weekday_indexed{Tuesday, 1}, "");
    static_assert(ymwd == year_month_weekday{sys_days{ymwd}}, ""); // round trip
  }

  {
    sys_days sd{days{-(10957 + 34)}}; // 29-Nov-1939 was a Wednesday
    year_month_weekday ymwd{sd};
    auto constexpr November  = cuda::std::chrono::November;
    auto constexpr Wednesday = cuda::std::chrono::Wednesday;

    assert(ymwd.ok());
    assert(ymwd.year() == year{1939});
    assert(ymwd.month() == November);
    assert(ymwd.weekday() == Wednesday);
    assert(ymwd.index() == 5);
    assert((ymwd.weekday_indexed() == weekday_indexed{Wednesday, 5}));
    assert(ymwd == year_month_weekday{sys_days{ymwd}}); // round trip
  }

  return 0;
}
