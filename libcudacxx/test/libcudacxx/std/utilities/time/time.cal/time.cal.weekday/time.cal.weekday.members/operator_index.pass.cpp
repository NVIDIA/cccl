//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class weekday;

//   constexpr weekday_indexed operator[](unsigned index) const noexcept;
//   constexpr weekday_last    operator[](last_spec)      const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "../../euclidian.h"
#include "test_macros.h"

int main(int, char**)
{
  using weekday         = cuda::std::chrono::weekday;
  using weekday_last    = cuda::std::chrono::weekday_last;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  constexpr weekday Sunday = cuda::std::chrono::Sunday;

  ASSERT_NOEXCEPT(cuda::std::declval<weekday>()[1U]);
  ASSERT_SAME_TYPE(weekday_indexed, decltype(cuda::std::declval<weekday>()[1U]));

  ASSERT_NOEXCEPT(cuda::std::declval<weekday>()[cuda::std::chrono::last]);
  ASSERT_SAME_TYPE(weekday_last, decltype(cuda::std::declval<weekday>()[cuda::std::chrono::last]));

  static_assert(Sunday[2].weekday() == Sunday, "");
  static_assert(Sunday[2].index() == 2, "");

  for (unsigned i = 0; i <= 6; ++i)
  {
    weekday wd(i);
    weekday_last wdl = wd[cuda::std::chrono::last];
    assert(wdl.weekday() == wd);
    assert(wdl.ok());
  }

  for (unsigned i = 0; i <= 6; ++i)
  {
    for (unsigned j = 1; j <= 5; ++j)
    {
      weekday wd(i);
      weekday_indexed wdi = wd[j];
      assert(wdi.weekday() == wd);
      assert(wdi.index() == j);
      assert(wdi.ok());
    }
  }

  return 0;
}
