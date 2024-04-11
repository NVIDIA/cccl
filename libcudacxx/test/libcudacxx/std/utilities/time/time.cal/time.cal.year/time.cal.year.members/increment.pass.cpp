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

//  constexpr year& operator++() noexcept;
//  constexpr year operator++(int) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename Y>
__host__ __device__ constexpr bool testConstexpr()
{
  Y y1{10};
  if (static_cast<int>(++y1) != 11)
  {
    return false;
  }
  if (static_cast<int>(y1++) != 11)
  {
    return false;
  }
  if (static_cast<int>(y1) != 12)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year = cuda::std::chrono::year;
  ASSERT_NOEXCEPT(++(cuda::std::declval<year&>()));
  ASSERT_NOEXCEPT((cuda::std::declval<year&>())++);

  ASSERT_SAME_TYPE(year, decltype(cuda::std::declval<year&>()++));
  ASSERT_SAME_TYPE(year&, decltype(++cuda::std::declval<year&>()));

  static_assert(testConstexpr<year>(), "");

  for (int i = 11000; i <= 11020; ++i)
  {
    year year(i);
    assert(static_cast<int>(++year) == i + 1);
    assert(static_cast<int>(year++) == i + 1);
    assert(static_cast<int>(year) == i + 2);
  }

  return 0;
}
