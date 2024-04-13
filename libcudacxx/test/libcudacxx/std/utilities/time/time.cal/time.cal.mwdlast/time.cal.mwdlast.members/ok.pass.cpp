//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class month_weekday_last;

// constexpr bool ok() const noexcept;
//  Returns: m_.ok() && wdl_.ok().

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month              = cuda::std::chrono::month;
  using weekday            = cuda::std::chrono::weekday;
  using weekday_last       = cuda::std::chrono::weekday_last;
  using month_weekday_last = cuda::std::chrono::month_weekday_last;

  constexpr month January            = cuda::std::chrono::January;
  constexpr weekday Tuesday          = cuda::std::chrono::Tuesday;
  constexpr weekday_last lastTuesday = weekday_last{Tuesday};

  ASSERT_NOEXCEPT(cuda::std::declval<const month_weekday_last>().ok());
  ASSERT_SAME_TYPE(bool, decltype(cuda::std::declval<const month_weekday_last>().ok()));

  static_assert(!month_weekday_last{month{}, lastTuesday}.ok(), ""); // Bad month
  static_assert(!month_weekday_last{January, weekday_last{weekday{12}}}.ok(), ""); // Bad month
  static_assert(month_weekday_last{January, lastTuesday}.ok(), ""); // Both OK

  for (unsigned i = 0; i <= 50; ++i)
  {
    month_weekday_last mwdl{month{i}, lastTuesday};
    assert(mwdl.ok() == month{i}.ok());
  }

  for (unsigned i = 0; i <= 50; ++i)
  {
    month_weekday_last mwdl{January, weekday_last{weekday{i}}};
    assert(mwdl.ok() == weekday_last{weekday{i}}.ok());
  }

  return 0;
}
