//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year;

// constexpr year operator+() const noexcept;
// constexpr year operator-() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename Y>
__host__ __device__ constexpr bool testConstexpr()
{
  Y y1{1};
  if (static_cast<int>(+y1) != 1)
  {
    return false;
  }
  if (static_cast<int>(-y1) != -1)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year = cuda::std::chrono::year;

  static_assert(noexcept(+cuda::std::declval<year>()));
  static_assert(noexcept(-cuda::std::declval<year>()));

  static_assert(cuda::std::is_same_v<year, decltype(+cuda::std::declval<year>())>);
  static_assert(cuda::std::is_same_v<year, decltype(-cuda::std::declval<year>())>);

  static_assert(testConstexpr<year>(), "");

  for (int i = 10000; i <= 10020; ++i)
  {
    year year(i);
    assert(static_cast<int>(+year) == i);
    assert(static_cast<int>(-year) == -i);
  }

  return 0;
}
