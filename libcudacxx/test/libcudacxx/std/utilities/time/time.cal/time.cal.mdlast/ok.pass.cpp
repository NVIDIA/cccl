//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class month_day_last;

// constexpr bool ok() const noexcept;
//  Returns: m_.ok()

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month          = cuda::std::chrono::month;
  using month_day_last = cuda::std::chrono::month_day_last;

  ASSERT_NOEXCEPT(cuda::std::declval<const month_day_last>().ok());
  ASSERT_SAME_TYPE(bool, decltype(cuda::std::declval<const month_day_last>().ok()));

  static_assert(!month_day_last{month{}}.ok(), "");
  static_assert(month_day_last{cuda::std::chrono::May}.ok(), "");

  for (unsigned i = 1; i <= 12; ++i)
  {
    month_day_last mdl{month{i}};
    assert(mdl.ok());
  }

  //  If the month is not ok, all the days are bad
  for (unsigned i = 13; i <= 50; ++i)
  {
    month_day_last mdl{month{i}};
    assert(!mdl.ok());
  }

  return 0;
}
