//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class month_weekday_last;

// constexpr chrono::month month() const noexcept;
//  Returns: m_

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

  constexpr weekday Tuesday = cuda::std::chrono::Tuesday;

  static_assert(noexcept(cuda::std::declval<const month_weekday_last>().month()));
  static_assert(cuda::std::is_same_v<month, decltype(cuda::std::declval<const month_weekday_last>().month())>);

  static_assert(month_weekday_last{month{}, weekday_last{Tuesday}}.month() == month{}, "");

  for (unsigned i = 1; i <= 50; ++i)
  {
    month_weekday_last mdl(month{i}, weekday_last{Tuesday});
    assert(static_cast<unsigned>(mdl.month()) == i);
  }

  return 0;
}
