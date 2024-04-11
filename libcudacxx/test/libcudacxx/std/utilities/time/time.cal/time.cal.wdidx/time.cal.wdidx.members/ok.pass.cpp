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

// constexpr bool ok() const noexcept;
//  Returns: wd_.ok() && 1 <= index_ && index_ <= 5

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  ASSERT_NOEXCEPT(cuda::std::declval<const weekday_indexed>().ok());
  ASSERT_SAME_TYPE(bool, decltype(cuda::std::declval<const weekday_indexed>().ok()));

  static_assert(!weekday_indexed{}.ok(), "");
  static_assert(weekday_indexed{cuda::std::chrono::Sunday, 2}.ok(), "");

  assert(!(weekday_indexed(cuda::std::chrono::Tuesday, 0).ok()));
  auto constexpr Tuesday = cuda::std::chrono::Tuesday;
  for (unsigned i = 1; i <= 5; ++i)
  {
    weekday_indexed wdi(Tuesday, i);
    assert(wdi.ok());
  }

  for (unsigned i = 6; i <= 20; ++i)
  {
    weekday_indexed wdi(Tuesday, i);
    assert(!wdi.ok());
  }

  //  Not a valid weekday
  assert(!(weekday_indexed(weekday{9U}, 1).ok()));

  return 0;
}
