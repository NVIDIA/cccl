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

// constexpr weekday operator-(const weekday& x, const days& y) noexcept;
//   Returns: x + -y.
//
// constexpr days operator-(const weekday& x, const weekday& y) noexcept;
// Returns: If x.ok() == true and y.ok() == true, returns a value d in the range
//    [days{0}, days{6}] satisfying y + d == x.
// Otherwise the value returned is unspecified.
// [Example: Sunday - Monday == days{6}. —end example]

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "../../euclidian.h"
#include "test_macros.h"

template <typename WD, typename Ds>
__host__ __device__ constexpr bool testConstexpr()
{
  {
    WD wd{5};
    Ds offset{3};
    if (wd - offset != WD{2})
    {
      return false;
    }
    if (wd - WD{2} != offset)
    {
      return false;
    }
  }

  //  Check the example
  if (WD{0} - WD{1} != Ds{6})
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using weekday = cuda::std::chrono::weekday;
  using days    = cuda::std::chrono::days;

  ASSERT_NOEXCEPT(cuda::std::declval<weekday>() - cuda::std::declval<days>());
  ASSERT_SAME_TYPE(weekday, decltype(cuda::std::declval<weekday>() - cuda::std::declval<days>()));

  ASSERT_NOEXCEPT(cuda::std::declval<weekday>() - cuda::std::declval<weekday>());
  ASSERT_SAME_TYPE(days, decltype(cuda::std::declval<weekday>() - cuda::std::declval<weekday>()));

  static_assert(testConstexpr<weekday, days>(), "");

  for (unsigned i = 0; i <= 6; ++i)
  {
    for (unsigned j = 0; j <= 6; ++j)
    {
      weekday wd = weekday{i} - days{j};
      assert(wd + days{j} == weekday{i});
#ifndef TEST_COMPILER_ICC
      assert((wd.c_encoding() == euclidian_subtraction<unsigned, 0, 6>(i, j)));
#endif // TEST_COMPILER_ICC
    }
  }

  for (unsigned i = 0; i <= 6; ++i)
  {
    for (unsigned j = 0; j <= 6; ++j)
    {
      days d = weekday{j} - weekday{i};
      assert(weekday{i} + d == weekday{j});
    }
  }

  return 0;
}
