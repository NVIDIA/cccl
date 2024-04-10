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

// constexpr year_month_day operator-(const year_month_day& ymd, const years& dy) noexcept;
//    Returns: ymd + (-dy)

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test_constexpr()
{
  cuda::std::chrono::year_month_day ym0{
    cuda::std::chrono::year{1234}, cuda::std::chrono::January, cuda::std::chrono::day{12}};
  cuda::std::chrono::year_month_day ym1 = ym0 - cuda::std::chrono::years{10};
  return ym1.year() == cuda::std::chrono::year{1234 - 10} && ym1.month() == cuda::std::chrono::January
      && ym1.day() == cuda::std::chrono::day{12};
}

int main(int, char**)
{
  using year           = cuda::std::chrono::year;
  using month          = cuda::std::chrono::month;
  using day            = cuda::std::chrono::day;
  using year_month_day = cuda::std::chrono::year_month_day;
  using years          = cuda::std::chrono::years;

  ASSERT_NOEXCEPT(cuda::std::declval<year_month_day>() - cuda::std::declval<years>());
  ASSERT_SAME_TYPE(year_month_day, decltype(cuda::std::declval<year_month_day>() - cuda::std::declval<years>()));

  constexpr month January = cuda::std::chrono::January;

  static_assert(test_constexpr(), "");

  year_month_day ym{year{1234}, January, day{10}};
  for (int i = 0; i <= 10; ++i)
  {
    year_month_day ym1 = ym - years{i};
    assert(static_cast<int>(ym1.year()) == 1234 - i);
    assert(ym1.month() == January);
    assert(ym1.day() == day{10});
  }

  return 0;
}
