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
// class year_month_day_last;

// constexpr year_month_day_last& operator+=(const months& m) noexcept;
// constexpr year_month_day_last& operator-=(const months& m) noexcept;

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
  using year                = cuda::std::chrono::year;
  using month               = cuda::std::chrono::month;
  using month_day_last      = cuda::std::chrono::month_day_last;
  using year_month_day_last = cuda::std::chrono::year_month_day_last;
  using months              = cuda::std::chrono::months;

  ASSERT_NOEXCEPT(cuda::std::declval<year_month_day_last&>() += cuda::std::declval<months>());
  ASSERT_NOEXCEPT(cuda::std::declval<year_month_day_last&>() -= cuda::std::declval<months>());

  ASSERT_SAME_TYPE(year_month_day_last&,
                   decltype(cuda::std::declval<year_month_day_last&>() += cuda::std::declval<months>()));
  ASSERT_SAME_TYPE(year_month_day_last&,
                   decltype(cuda::std::declval<year_month_day_last&>() -= cuda::std::declval<months>()));

  static_assert(testConstexpr<year_month_day_last, months>(year_month_day_last{year{1234}, month_day_last{month{1}}}),
                "");

  for (unsigned i = 0; i <= 10; ++i)
  {
    year y{1234};
    month_day_last mdl{month{i}};
    year_month_day_last ym(y, mdl);
    assert(static_cast<unsigned>((ym += months{2}).month()) == i + 2);
    assert(ym.year() == y);
    assert(static_cast<unsigned>((ym).month()) == i + 2);
    assert(ym.year() == y);
    assert(static_cast<unsigned>((ym -= months{1}).month()) == i + 1);
    assert(ym.year() == y);
    assert(static_cast<unsigned>((ym).month()) == i + 1);
    assert(ym.year() == y);
  }

  return 0;
}
