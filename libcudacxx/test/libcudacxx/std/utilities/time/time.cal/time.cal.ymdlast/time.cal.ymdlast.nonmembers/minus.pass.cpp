//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year_month_day_last;

// constexpr year_month_day_last
//   operator-(const year_month_day_last& ymdl, const months& dm) noexcept;
//
//   Returns: ymdl + (-dm).
//
// constexpr year_month_day_last
//   operator-(const year_month_day_last& ymdl, const years& dy) noexcept;
//
//   Returns: ymdl + (-dy).

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool testConstexprYears(cuda::std::chrono::year_month_day_last ymdl)
{
  cuda::std::chrono::year_month_day_last ym1 = ymdl - cuda::std::chrono::years{10};
  return ym1.year() == cuda::std::chrono::year{static_cast<int>(ymdl.year()) - 10} && ym1.month() == ymdl.month();
}

__host__ __device__ constexpr bool testConstexprMonths(cuda::std::chrono::year_month_day_last ymdl)
{
  cuda::std::chrono::year_month_day_last ym1 = ymdl - cuda::std::chrono::months{6};
  return ym1.year() == ymdl.year() && ym1.month() == cuda::std::chrono::month{static_cast<unsigned>(ymdl.month()) - 6};
}

int main(int, char**)
{
  using year                = cuda::std::chrono::year;
  using month               = cuda::std::chrono::month;
  using month_day_last      = cuda::std::chrono::month_day_last;
  using year_month_day_last = cuda::std::chrono::year_month_day_last;
  using months              = cuda::std::chrono::months;
  using years               = cuda::std::chrono::years;

  constexpr month December = cuda::std::chrono::December;

  { // year_month_day_last - years
    ASSERT_NOEXCEPT(cuda::std::declval<year_month_day_last>() - cuda::std::declval<years>());
    ASSERT_SAME_TYPE(year_month_day_last,
                     decltype(cuda::std::declval<year_month_day_last>() - cuda::std::declval<years>()));

    static_assert(testConstexprYears(year_month_day_last{year{1234}, month_day_last{December}}), "");
    year_month_day_last ym{year{1234}, month_day_last{December}};
    for (int i = 0; i <= 10; ++i)
    {
      year_month_day_last ym1 = ym - years{i};
      assert(static_cast<int>(ym1.year()) == 1234 - i);
      assert(ym1.month() == December);
    }
  }

  { // year_month_day_last - months
    ASSERT_NOEXCEPT(cuda::std::declval<year_month_day_last>() - cuda::std::declval<months>());
    ASSERT_SAME_TYPE(year_month_day_last,
                     decltype(cuda::std::declval<year_month_day_last>() - cuda::std::declval<months>()));

    static_assert(testConstexprMonths(year_month_day_last{year{1234}, month_day_last{December}}), "");
    //  TODO test wrapping
    year_month_day_last ym{year{1234}, month_day_last{December}};
    for (unsigned i = 0; i <= 10; ++i)
    {
      year_month_day_last ym1 = ym - months{i};
      assert(static_cast<int>(ym1.year()) == 1234);
      assert(static_cast<unsigned>(ym1.month()) == 12U - i);
    }
  }

  return 0;
}
