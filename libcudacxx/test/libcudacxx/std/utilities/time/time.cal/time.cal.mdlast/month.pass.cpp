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

// constexpr chrono::month month() const noexcept;
//  Returns: m_

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month          = cuda::std::chrono::month;
  using month_day_last = cuda::std::chrono::month_day_last;

  ASSERT_NOEXCEPT(cuda::std::declval<const month_day_last>().month());
  ASSERT_SAME_TYPE(month, decltype(cuda::std::declval<const month_day_last>().month()));

  static_assert(month_day_last{month{}}.month() == month{}, "");

  for (unsigned i = 1; i <= 50; ++i)
  {
    month_day_last mdl(month{i});
    assert(static_cast<unsigned>(mdl.month()) == i);
  }

  return 0;
}
