//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_weekday;

// constexpr unsigned index() const noexcept;
//  Returns: wdi_.index()

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year               = cuda::std::chrono::year;
  using month              = cuda::std::chrono::month;
  using weekday            = cuda::std::chrono::weekday;
  using weekday_indexed    = cuda::std::chrono::weekday_indexed;
  using year_month_weekday = cuda::std::chrono::year_month_weekday;

  static_assert(noexcept(cuda::std::declval<const year_month_weekday>().index()));
  static_assert(cuda::std::is_same_v<unsigned, decltype(cuda::std::declval<const year_month_weekday>().index())>);

  static_assert(year_month_weekday{}.index() == 0, "");

  for (unsigned i = 1; i <= 50; ++i)
  {
    year_month_weekday ymwd0(year{1234}, month{2}, weekday_indexed{weekday{2}, i});
    assert(ymwd0.index() == i);
  }

  return 0;
}
