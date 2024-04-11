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

//  constexpr unsigned c_encoding() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename WD>
__host__ __device__ constexpr bool testConstexpr()
{
  WD wd{5};
  return wd.c_encoding() == 5;
}

int main(int, char**)
{
  using weekday = cuda::std::chrono::weekday;

  ASSERT_NOEXCEPT(cuda::std::declval<weekday&>().c_encoding());
  ASSERT_SAME_TYPE(unsigned, decltype(cuda::std::declval<weekday&>().c_encoding()));

  static_assert(testConstexpr<weekday>(), "");

  for (unsigned i = 0; i <= 10; ++i)
  {
    weekday wd(i);
    assert(wd.c_encoding() == (i == 7 ? 0 : i));
  }

  return 0;
}
