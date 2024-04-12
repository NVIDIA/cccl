//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class day;

// constexpr bool operator==(const day& x, const day& y) noexcept;
//   Returns: unsigned{x} == unsigned{y}.
// constexpr bool operator<(const day& x, const day& y) noexcept;
//   Returns: unsigned{x} < unsigned{y}.

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using day = cuda::std::chrono::day;

  AssertComparisonsAreNoexcept<day>();
  AssertComparisonsReturnBool<day>();

  static_assert(testComparisonsValues<day>(0U, 0U), "");
  static_assert(testComparisonsValues<day>(0U, 1U), "");

  //  Some 'ok' values as well
  static_assert(testComparisonsValues<day>(5U, 5U), "");
  static_assert(testComparisonsValues<day>(5U, 10U), "");

  for (unsigned i = 1; i < 10; ++i)
  {
    for (unsigned j = 1; j < 10; ++j)
    {
      assert(testComparisonsValues<day>(i, j));
    }
  }

  return 0;
}
