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

//  constexpr unsigned iso_encoding() const noexcept;
//  Returns the underlying weekday, _except_ that returns '7' for Sunday (zero)
//    See [time.cal.wd.members]

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

  ASSERT_NOEXCEPT(cuda::std::declval<weekday&>().iso_encoding());
  ASSERT_SAME_TYPE(unsigned, decltype(cuda::std::declval<weekday&>().iso_encoding()));

  static_assert(testConstexpr<weekday>(), "");

  //  This is different than all the other tests, because the '7' gets converted to
  //  a zero in the constructor, but then back to '7' by iso_encoding().
  for (unsigned i = 0; i <= 10; ++i)
  {
    weekday wd(i);
    assert(wd.iso_encoding() == (i == 0 ? 7 : i));
  }

  return 0;
}
