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

// constexpr year_month_weekday& operator+=(const months& m) noexcept;
// constexpr year_month_weekday& operator-=(const months& m) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename D, typename Ds>
__host__ __device__ constexpr bool testConstexpr(D d1)
{
  if (static_cast<unsigned>((d1).month()) != 1)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 += Ds{1}).month()) != 2)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 += Ds{2}).month()) != 4)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 += Ds{12}).month()) != 4)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 -= Ds{1}).month()) != 3)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 -= Ds{2}).month()) != 1)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 -= Ds{12}).month()) != 1)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year               = cuda::std::chrono::year;
  using month              = cuda::std::chrono::month;
  using weekday_indexed    = cuda::std::chrono::weekday_indexed;
  using year_month_weekday = cuda::std::chrono::year_month_weekday;
  using months             = cuda::std::chrono::months;

  ASSERT_NOEXCEPT(cuda::std::declval<year_month_weekday&>() += cuda::std::declval<months>());
  ASSERT_SAME_TYPE(year_month_weekday&,
                   decltype(cuda::std::declval<year_month_weekday&>() += cuda::std::declval<months>()));

  ASSERT_NOEXCEPT(cuda::std::declval<year_month_weekday&>() -= cuda::std::declval<months>());
  ASSERT_SAME_TYPE(year_month_weekday&,
                   decltype(cuda::std::declval<year_month_weekday&>() -= cuda::std::declval<months>()));

  auto constexpr Tuesday = cuda::std::chrono::Tuesday;
  static_assert(
    testConstexpr<year_month_weekday, months>(year_month_weekday{year{1234}, month{1}, weekday_indexed{Tuesday, 2}}),
    "");

  for (unsigned i = 0; i <= 10; ++i)
  {
    year y{1234};
    year_month_weekday ymwd(y, month{i}, weekday_indexed{Tuesday, 2});

    assert(static_cast<unsigned>((ymwd += months{2}).month()) == i + 2);
    assert(ymwd.year() == y);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);

    assert(static_cast<unsigned>((ymwd).month()) == i + 2);
    assert(ymwd.year() == y);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);

    assert(static_cast<unsigned>((ymwd -= months{1}).month()) == i + 1);
    assert(ymwd.year() == y);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);

    assert(static_cast<unsigned>((ymwd).month()) == i + 1);
    assert(ymwd.year() == y);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);
  }

  return 0;
}
