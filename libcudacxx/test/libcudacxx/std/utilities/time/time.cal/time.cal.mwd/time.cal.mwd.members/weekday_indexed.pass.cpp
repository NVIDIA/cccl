//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class month_weekday;

// constexpr chrono::weekday_indexed weekday_indexed() const noexcept;
//  Returns: wdi_

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month_weekday   = cuda::std::chrono::month_weekday;
  using month           = cuda::std::chrono::month;
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  constexpr weekday Sunday = cuda::std::chrono::Sunday;

  ASSERT_NOEXCEPT(cuda::std::declval<const month_weekday>().weekday_indexed());
  ASSERT_SAME_TYPE(weekday_indexed, decltype(cuda::std::declval<const month_weekday>().weekday_indexed()));

  static_assert(month_weekday{month{}, weekday_indexed{}}.weekday_indexed() == weekday_indexed{}, "");

  for (unsigned i = 1; i <= 10; ++i)
  {
    constexpr month March = cuda::std::chrono::March;
    month_weekday md(March, weekday_indexed{Sunday, i});
    assert(static_cast<unsigned>(md.weekday_indexed().weekday() == Sunday));
    assert(static_cast<unsigned>(md.weekday_indexed().index() == i));
  }

  return 0;
}
