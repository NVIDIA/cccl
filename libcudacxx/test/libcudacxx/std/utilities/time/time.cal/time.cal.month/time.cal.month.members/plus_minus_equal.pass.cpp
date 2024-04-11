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

// constexpr month& operator+=(const month& d) noexcept;
// constexpr month& operator-=(const month& d) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename M, typename Ms>
__host__ __device__ constexpr bool testConstexpr()
{
  M m1{1};
  if (static_cast<unsigned>(m1 += Ms{1}) != 2)
  {
    return false;
  }
  if (static_cast<unsigned>(m1 += Ms{2}) != 4)
  {
    return false;
  }
  if (static_cast<unsigned>(m1 += Ms{8}) != 12)
  {
    return false;
  }
  if (static_cast<unsigned>(m1 -= Ms{1}) != 11)
  {
    return false;
  }
  if (static_cast<unsigned>(m1 -= Ms{2}) != 9)
  {
    return false;
  }
  if (static_cast<unsigned>(m1 -= Ms{8}) != 1)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using month  = cuda::std::chrono::month;
  using months = cuda::std::chrono::months;

  ASSERT_NOEXCEPT(cuda::std::declval<month&>() += cuda::std::declval<months&>());
  ASSERT_NOEXCEPT(cuda::std::declval<month&>() -= cuda::std::declval<months&>());
  ASSERT_SAME_TYPE(month&, decltype(cuda::std::declval<month&>() += cuda::std::declval<months&>()));
  ASSERT_SAME_TYPE(month&, decltype(cuda::std::declval<month&>() -= cuda::std::declval<months&>()));

  static_assert(testConstexpr<month, months>(), "");

  for (unsigned i = 1; i <= 10; ++i)
  {
    month month(i);
    int exp = i + 10;
    while (exp > 12)
    {
      exp -= 12;
    }
    assert(static_cast<unsigned>(month += months{10}) == static_cast<unsigned>(exp));
    assert(static_cast<unsigned>(month) == static_cast<unsigned>(exp));
  }

  for (unsigned i = 1; i <= 10; ++i)
  {
    month month(i);
    int exp = i - 9;
    while (exp < 1)
    {
      exp += 12;
    }
    assert(static_cast<unsigned>(month -= months{9}) == static_cast<unsigned>(exp));
    assert(static_cast<unsigned>(month) == static_cast<unsigned>(exp));
  }

  return 0;
}
