//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class weekday_last;

//  constexpr bool ok() const noexcept;
//  Returns: wd_.ok()

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using weekday      = cuda::std::chrono::weekday;
  using weekday_last = cuda::std::chrono::weekday_last;

  ASSERT_NOEXCEPT(cuda::std::declval<const weekday_last>().ok());
  ASSERT_SAME_TYPE(bool, decltype(cuda::std::declval<const weekday_last>().ok()));

  static_assert(weekday_last{weekday{0}}.ok(), "");
  static_assert(weekday_last{weekday{1}}.ok(), "");
  static_assert(!weekday_last{weekday{8}}.ok(), "");

  for (unsigned i = 0; i <= 255; ++i)
  {
    assert(weekday_last{weekday{i}}.ok() == weekday{i}.ok());
  }

  return 0;
}
