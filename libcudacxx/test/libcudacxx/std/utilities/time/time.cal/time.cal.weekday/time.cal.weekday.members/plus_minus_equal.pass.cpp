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

// constexpr weekday& operator+=(const days& d) noexcept;
// constexpr weekday& operator-=(const days& d) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "../../euclidian.h"
#include "test_macros.h"

template <typename M, typename Ms>
__host__ __device__ constexpr bool testConstexpr()
{
  M m1{1};
  if ((m1 += Ms{1}).c_encoding() != 2)
  {
    return false;
  }
  if ((m1 += Ms{2}).c_encoding() != 4)
  {
    return false;
  }
  if ((m1 += Ms{4}).c_encoding() != 1)
  {
    return false;
  }
  if ((m1 -= Ms{1}).c_encoding() != 0)
  {
    return false;
  }
  if ((m1 -= Ms{2}).c_encoding() != 5)
  {
    return false;
  }
  if ((m1 -= Ms{4}).c_encoding() != 1)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using weekday = cuda::std::chrono::weekday;
  using days    = cuda::std::chrono::days;

  ASSERT_NOEXCEPT(cuda::std::declval<weekday&>() += cuda::std::declval<days&>());
  ASSERT_SAME_TYPE(weekday&, decltype(cuda::std::declval<weekday&>() += cuda::std::declval<days&>()));

  ASSERT_NOEXCEPT(cuda::std::declval<weekday&>() -= cuda::std::declval<days&>());
  ASSERT_SAME_TYPE(weekday&, decltype(cuda::std::declval<weekday&>() -= cuda::std::declval<days&>()));

  static_assert(testConstexpr<weekday, days>(), "");

  for (unsigned i = 0; i <= 6; ++i)
  {
    weekday wd(i);
    assert(((wd += days{3}).c_encoding() == euclidian_addition<unsigned, 0, 6>(i, 3)));
    assert(((wd).c_encoding() == euclidian_addition<unsigned, 0, 6>(i, 3)));
  }

#ifndef TEST_COMPILER_ICC
  for (unsigned i = 0; i <= 6; ++i)
  {
    weekday wd(i);
    assert(((wd -= days{4}).c_encoding() == euclidian_subtraction<unsigned, 0, 6>(i, 4)));
    assert(((wd).c_encoding() == euclidian_subtraction<unsigned, 0, 6>(i, 4)));
  }
#endif // TEST_COMPILER_ICC

  return 0;
}
