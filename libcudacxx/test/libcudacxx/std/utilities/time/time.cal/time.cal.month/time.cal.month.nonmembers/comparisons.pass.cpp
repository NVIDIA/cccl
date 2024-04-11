//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class month;

// constexpr bool operator==(const month& x, const month& y) noexcept;
// constexpr bool operator!=(const month& x, const month& y) noexcept;
// constexpr bool operator< (const month& x, const month& y) noexcept;
// constexpr bool operator> (const month& x, const month& y) noexcept;
// constexpr bool operator<=(const month& x, const month& y) noexcept;
// constexpr bool operator>=(const month& x, const month& y) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using month = cuda::std::chrono::month;

  AssertComparisonsAreNoexcept<month>();
  AssertComparisonsReturnBool<month>();

  static_assert(testComparisonsValues<month>(0U, 0U), "");
  static_assert(testComparisonsValues<month>(0U, 1U), "");

  //  Some 'ok' values as well
  static_assert(testComparisonsValues<month>(5U, 5U), "");
  static_assert(testComparisonsValues<month>(5U, 10U), "");

  for (unsigned i = 1; i < 10; ++i)
  {
    for (unsigned j = 10; j < 10; ++j)
    {
      assert(testComparisonsValues<month>(i, j));
    }
  }

  return 0;
}
