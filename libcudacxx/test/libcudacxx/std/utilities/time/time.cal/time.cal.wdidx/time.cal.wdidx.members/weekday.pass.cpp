//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class weekday_indexed;

// constexpr chrono::weekday weekday() const noexcept;
//  Returns: wd_

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  ASSERT_NOEXCEPT(cuda::std::declval<const weekday_indexed>().weekday());
  ASSERT_SAME_TYPE(cuda::std::chrono::weekday, decltype(cuda::std::declval<const weekday_indexed>().weekday()));

  static_assert(weekday_indexed{}.weekday() == weekday{}, "");
  static_assert(weekday_indexed{cuda::std::chrono::Tuesday, 0}.weekday() == cuda::std::chrono::Tuesday, "");

  for (unsigned i = 0; i <= 6; ++i)
  {
    weekday_indexed wdi(weekday{i}, 2);
    assert(wdi.weekday().c_encoding() == i);
  }

  return 0;
}
