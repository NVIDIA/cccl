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

// constexpr bool operator==(const weekday& x, const weekday& y) noexcept;
// constexpr bool operator!=(const weekday& x, const weekday& y) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using weekday      = cuda::std::chrono::weekday;
  using weekday_last = cuda::std::chrono::weekday_last;

  AssertEqualityAreNoexcept<weekday_last>();
  AssertEqualityReturnBool<weekday_last>();

  static_assert(testEqualityValues<weekday_last>(weekday{0}, weekday{0}), "");
  static_assert(testEqualityValues<weekday_last>(weekday{0}, weekday{1}), "");

  //  Some 'ok' values as well
  static_assert(testEqualityValues<weekday_last>(weekday{2}, weekday{2}), "");
  static_assert(testEqualityValues<weekday_last>(weekday{2}, weekday{3}), "");

  for (unsigned i = 0; i < 6; ++i)
  {
    for (unsigned j = 0; j < 6; ++j)
    {
      assert(testEqualityValues<weekday_last>(weekday{i}, weekday{j}));
    }
  }

  return 0;
}
