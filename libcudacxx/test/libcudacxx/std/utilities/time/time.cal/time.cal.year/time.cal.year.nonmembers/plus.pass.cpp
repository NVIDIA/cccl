//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year;

// constexpr year operator+(const year& x, const years& y) noexcept;
//   Returns: year(int{x} + y.count()).
//
// constexpr year operator+(const years& x, const year& y) noexcept;
//   Returns: y + x

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename Y, typename Ys>
__host__ __device__ constexpr bool testConstexpr()
{
  Y y{1001};
  Ys offset{23};
  if (y + offset != Y{1024})
  {
    return false;
  }
  if (offset + y != Y{1024})
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year  = cuda::std::chrono::year;
  using years = cuda::std::chrono::years;

  ASSERT_NOEXCEPT(cuda::std::declval<year>() + cuda::std::declval<years>());
  ASSERT_SAME_TYPE(year, decltype(cuda::std::declval<year>() + cuda::std::declval<years>()));

  ASSERT_NOEXCEPT(cuda::std::declval<years>() + cuda::std::declval<year>());
  ASSERT_SAME_TYPE(year, decltype(cuda::std::declval<years>() + cuda::std::declval<year>()));

  static_assert(testConstexpr<year, years>(), "");

  year y{1223};
  for (int i = 1100; i <= 1110; ++i)
  {
    year y1 = y + years{i};
    year y2 = years{i} + y;
    assert(y1 == y2);
    assert(static_cast<int>(y1) == i + 1223);
    assert(static_cast<int>(y2) == i + 1223);
  }

  return 0;
}
