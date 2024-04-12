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

// constexpr bool operator==(const weekday_indexed& x, const weekday_indexed& y) noexcept;
//   Returns: x.weekday() == y.weekday() && x.index() == y.index().
// constexpr bool operator!=(const weekday_indexed& x, const weekday_indexed& y) noexcept;
//   Returns: !(x == y)

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  AssertEqualityAreNoexcept<weekday_indexed>();
  AssertEqualityReturnBool<weekday_indexed>();

  static_assert((weekday_indexed{} == weekday_indexed{}), "");
  static_assert(!(weekday_indexed{} != weekday_indexed{}), "");

  static_assert(!(weekday_indexed{} == weekday_indexed{cuda::std::chrono::Tuesday, 1}), "");
  static_assert((weekday_indexed{} != weekday_indexed{cuda::std::chrono::Tuesday, 1}), "");

  //  Some 'ok' values as well
  static_assert((weekday_indexed{weekday{1}, 2} == weekday_indexed{weekday{1}, 2}), "");
  static_assert(!(weekday_indexed{weekday{1}, 2} != weekday_indexed{weekday{1}, 2}), "");

  static_assert(!(weekday_indexed{weekday{1}, 2} == weekday_indexed{weekday{1}, 1}), "");
  static_assert((weekday_indexed{weekday{1}, 2} != weekday_indexed{weekday{1}, 1}), "");
  static_assert(!(weekday_indexed{weekday{1}, 2} == weekday_indexed{weekday{2}, 2}), "");
  static_assert((weekday_indexed{weekday{1}, 2} != weekday_indexed{weekday{2}, 2}), "");

  return 0;
}
