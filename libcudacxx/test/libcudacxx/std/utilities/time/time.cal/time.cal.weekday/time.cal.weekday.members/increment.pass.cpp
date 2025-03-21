//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class weekday;

//  constexpr weekday& operator++() noexcept;
//  constexpr weekday operator++(int) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "../../euclidian.h"
#include "test_macros.h"

template <typename WD>
__host__ __device__ constexpr bool testConstexpr()
{
  WD wd{5};
  if ((++wd).c_encoding() != 6)
  {
    return false;
  }
  if ((wd++).c_encoding() != 6)
  {
    return false;
  }
  if ((wd).c_encoding() != 0)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using weekday = cuda::std::chrono::weekday;
  static_assert(noexcept(++(cuda::std::declval<weekday&>())));
  static_assert(noexcept((cuda::std::declval<weekday&>())++));

  static_assert(cuda::std::is_same_v<weekday, decltype(cuda::std::declval<weekday&>()++)>);
  static_assert(cuda::std::is_same_v<weekday&, decltype(++cuda::std::declval<weekday&>())>);

  static_assert(testConstexpr<weekday>(), "");

  for (unsigned i = 0; i <= 6; ++i)
  {
    weekday wd(i);
    assert(((++wd).c_encoding() == euclidian_addition<unsigned, 0, 6>(i, 1)));
    assert(((wd++).c_encoding() == euclidian_addition<unsigned, 0, 6>(i, 1)));
    assert(((wd).c_encoding() == euclidian_addition<unsigned, 0, 6>(i, 2)));
  }

  return 0;
}
