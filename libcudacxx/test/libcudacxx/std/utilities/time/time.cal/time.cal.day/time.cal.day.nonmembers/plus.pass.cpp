//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class day;

// constexpr day operator+(const day& x, const days& y) noexcept;
//   Returns: day(unsigned{x} + y.count()).
//
// constexpr day operator+(const days& x, const day& y) noexcept;
//   Returns: y + x.

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename D, typename Ds>
__host__ __device__ constexpr bool testConstexpr()
{
  D d{1};
  Ds offset{23};
  if (d + offset != D{24})
  {
    return false;
  }
  if (offset + d != D{24})
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using day  = cuda::std::chrono::day;
  using days = cuda::std::chrono::days;

  ASSERT_NOEXCEPT(cuda::std::declval<day>() + cuda::std::declval<days>());
  ASSERT_NOEXCEPT(cuda::std::declval<days>() + cuda::std::declval<day>());

  ASSERT_SAME_TYPE(day, decltype(cuda::std::declval<day>() + cuda::std::declval<days>()));
  ASSERT_SAME_TYPE(day, decltype(cuda::std::declval<days>() + cuda::std::declval<day>()));

  static_assert(testConstexpr<day, days>(), "");

  day dy{12};
  for (unsigned i = 0; i <= 10; ++i)
  {
    day d1 = dy + days{i};
    day d2 = days{i} + dy;
    assert(d1 == d2);
    assert(static_cast<unsigned>(d1) == i + 12);
    assert(static_cast<unsigned>(d2) == i + 12);
  }

  return 0;
}
