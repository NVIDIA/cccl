//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class weekday_indexed;

// constexpr bool ok() const noexcept;
//  Returns: wd_.ok() && 1 <= index_ && index_ <= 5

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  {
    weekday_indexed defaulted{};
    assert(!defaulted.ok());
  }
  {
    weekday_indexed from_day_index{cuda::std::chrono::Sunday, 2};
    assert(from_day_index.ok());
  }
  {
    weekday_indexed from_invalid{cuda::std::chrono::Tuesday, 0};
    assert(!from_invalid.ok());
  }

  constexpr auto Tuesday = cuda::std::chrono::Tuesday;
  for (unsigned i = 1; i <= 5; ++i)
  {
    const weekday_indexed wdi(Tuesday, i);
    assert(wdi.ok());
    static_assert(noexcept(wdi.ok()));
    static_assert(cuda::std::is_same_v<bool, decltype(wdi.ok())>);
  }

  for (unsigned i = 6; i <= 20; ++i)
  {
    weekday_indexed wdi(Tuesday, i);
    assert(!wdi.ok());
  }

  //  Not a valid weekday
  assert(!(weekday_indexed(weekday{9U}, 1).ok()));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
