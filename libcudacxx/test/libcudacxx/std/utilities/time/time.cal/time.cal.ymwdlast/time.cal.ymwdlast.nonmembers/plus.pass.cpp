//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year_month_weekday_last;

// constexpr year_month_weekday_last operator+(const year_month_weekday_last& ymwdl, const months& dm) noexcept;
//   Returns: (ymwdl.year() / ymwdl.month() + dm) / ymwdl.weekday_last().
//
// constexpr year_month_weekday_last operator+(const months& dm, const year_month_weekday_last& ymwdl) noexcept;
//   Returns: ymwdl + dm.
//
// constexpr year_month_weekday_last operator+(const year_month_weekday_last& ymwdl, const years& dy) noexcept;
//   Returns: {ymwdl.year()+dy, ymwdl.month(), ymwdl.weekday_last()}.
//
// constexpr year_month_weekday_last operator+(const years& dy, const year_month_weekday_last& ymwdl) noexcept;
//   Returns: ymwdl + dy.

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool testConstexprYears(cuda::std::chrono::year_month_weekday_last ym)
{
  cuda::std::chrono::years offset{23};
  if (static_cast<int>((ym).year()) != 1)
  {
    return false;
  }
  if (static_cast<int>((ym + offset).year()) != 24)
  {
    return false;
  }
  if (static_cast<int>((offset + ym).year()) != 24)
  {
    return false;
  }
  return true;
}

__host__ __device__ constexpr bool testConstexprMonths(cuda::std::chrono::year_month_weekday_last ym)
{
  cuda::std::chrono::months offset{6};
  if (static_cast<unsigned>((ym).month()) != 1)
  {
    return false;
  }
  if (static_cast<unsigned>((ym + offset).month()) != 7)
  {
    return false;
  }
  if (static_cast<unsigned>((offset + ym).month()) != 7)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year                    = cuda::std::chrono::year;
  using month                   = cuda::std::chrono::month;
  using weekday_last            = cuda::std::chrono::weekday_last;
  using year_month_weekday_last = cuda::std::chrono::year_month_weekday_last;
  using years                   = cuda::std::chrono::years;
  using months                  = cuda::std::chrono::months;

  auto constexpr Tuesday = cuda::std::chrono::Tuesday;
  auto constexpr January = cuda::std::chrono::January;

  { // year_month_weekday_last + months
    ASSERT_NOEXCEPT(cuda::std::declval<year_month_weekday_last>() + cuda::std::declval<months>());
    ASSERT_NOEXCEPT(cuda::std::declval<months>() + cuda::std::declval<year_month_weekday_last>());

    ASSERT_SAME_TYPE(year_month_weekday_last,
                     decltype(cuda::std::declval<year_month_weekday_last>() + cuda::std::declval<months>()));
    ASSERT_SAME_TYPE(year_month_weekday_last,
                     decltype(cuda::std::declval<months>() + cuda::std::declval<year_month_weekday_last>()));

    static_assert(testConstexprMonths(year_month_weekday_last{year{1}, January, weekday_last{Tuesday}}), "");

    year_month_weekday_last ym{year{1234}, January, weekday_last{Tuesday}};
    for (int i = 0; i <= 10; ++i) // TODO test wrap-around
    {
      year_month_weekday_last ym1 = ym + months{i};
      year_month_weekday_last ym2 = months{i} + ym;
      assert(ym1.year() == year(1234));
      assert(ym2.year() == year(1234));
      assert(ym1.month() == month(1 + i));
      assert(ym2.month() == month(1 + i));
      assert(ym1.weekday() == Tuesday);
      assert(ym2.weekday() == Tuesday);
      assert(ym1.weekday_last() == weekday_last{Tuesday});
      assert(ym2.weekday_last() == weekday_last{Tuesday});
      assert(ym1 == ym2);
    }
  }

  { // year_month_weekday_last + years
    ASSERT_NOEXCEPT(cuda::std::declval<year_month_weekday_last>() + cuda::std::declval<years>());
    ASSERT_NOEXCEPT(cuda::std::declval<years>() + cuda::std::declval<year_month_weekday_last>());

    ASSERT_SAME_TYPE(year_month_weekday_last,
                     decltype(cuda::std::declval<year_month_weekday_last>() + cuda::std::declval<years>()));
    ASSERT_SAME_TYPE(year_month_weekday_last,
                     decltype(cuda::std::declval<years>() + cuda::std::declval<year_month_weekday_last>()));

    static_assert(testConstexprYears(year_month_weekday_last{year{1}, January, weekday_last{Tuesday}}), "");

    year_month_weekday_last ym{year{1234}, cuda::std::chrono::January, weekday_last{Tuesday}};
    for (int i = 0; i <= 10; ++i)
    {
      year_month_weekday_last ym1 = ym + years{i};
      year_month_weekday_last ym2 = years{i} + ym;
      assert(ym1.year() == year(1234 + i));
      assert(ym2.year() == year(1234 + i));
      assert(ym1.month() == January);
      assert(ym2.month() == January);
      assert(ym1.weekday() == Tuesday);
      assert(ym2.weekday() == Tuesday);
      assert(ym1.weekday_last() == weekday_last{Tuesday});
      assert(ym2.weekday_last() == weekday_last{Tuesday});
      assert(ym1 == ym2);
    }
  }

  return 0;
}
