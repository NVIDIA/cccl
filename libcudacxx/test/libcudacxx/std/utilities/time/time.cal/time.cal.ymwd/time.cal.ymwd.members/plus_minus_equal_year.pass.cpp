//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11
// XFAIL: gcc-4.8, gcc-5, gcc-6
// gcc before gcc-7 fails with an internal compiler error

// <chrono>
// class year_month_weekday;

// constexpr year_month_weekday& operator+=(const years& d) noexcept;
// constexpr year_month_weekday& operator-=(const years& d) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename D, typename Ds>
__host__ __device__ constexpr bool testConstexpr(D d1)
{
  if (static_cast<int>((d1).year()) != 1)
  {
    return false;
  }
  if (static_cast<int>((d1 += Ds{1}).year()) != 2)
  {
    return false;
  }
  if (static_cast<int>((d1 += Ds{2}).year()) != 4)
  {
    return false;
  }
  if (static_cast<int>((d1 += Ds{12}).year()) != 16)
  {
    return false;
  }
  if (static_cast<int>((d1 -= Ds{1}).year()) != 15)
  {
    return false;
  }
  if (static_cast<int>((d1 -= Ds{2}).year()) != 13)
  {
    return false;
  }
  if (static_cast<int>((d1 -= Ds{12}).year()) != 1)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year               = cuda::std::chrono::year;
  using weekday_indexed    = cuda::std::chrono::weekday_indexed;
  using year_month_weekday = cuda::std::chrono::year_month_weekday;
  using years              = cuda::std::chrono::years;

  ASSERT_NOEXCEPT(cuda::std::declval<year_month_weekday&>() += cuda::std::declval<years>());
  ASSERT_SAME_TYPE(year_month_weekday&,
                   decltype(cuda::std::declval<year_month_weekday&>() += cuda::std::declval<years>()));

  ASSERT_NOEXCEPT(cuda::std::declval<year_month_weekday&>() -= cuda::std::declval<years>());
  ASSERT_SAME_TYPE(year_month_weekday&,
                   decltype(cuda::std::declval<year_month_weekday&>() -= cuda::std::declval<years>()));

  auto constexpr Tuesday = cuda::std::chrono::Tuesday;
  auto constexpr January = cuda::std::chrono::January;

  static_assert(
    testConstexpr<year_month_weekday, years>(year_month_weekday{year{1}, January, weekday_indexed{Tuesday, 2}}), "");

  for (int i = 1000; i <= 1010; ++i)
  {
    year_month_weekday ymwd(year{i}, January, weekday_indexed{Tuesday, 2});

    assert(static_cast<int>((ymwd += years{2}).year()) == i + 2);
    assert(ymwd.month() == January);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);

    assert(static_cast<int>((ymwd).year()) == i + 2);
    assert(ymwd.month() == January);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);

    assert(static_cast<int>((ymwd -= years{1}).year()) == i + 1);
    assert(ymwd.month() == January);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);

    assert(static_cast<int>((ymwd).year()) == i + 1);
    assert(ymwd.month() == January);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);
  }

  return 0;
}
