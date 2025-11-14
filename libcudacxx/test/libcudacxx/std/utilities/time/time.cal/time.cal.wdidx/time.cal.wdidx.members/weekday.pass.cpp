//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class weekday_indexed;

// constexpr chrono::weekday weekday() const noexcept;
//  Returns: wd_

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
    assert(defaulted.weekday() == weekday{});
  }

  {
    weekday_indexed from_day_index{cuda::std::chrono::Tuesday, 0};
    assert(from_day_index.weekday() == cuda::std::chrono::Tuesday);
  }

  for (unsigned i = 0; i <= 6; ++i)
  {
    const weekday_indexed wdi(weekday{i}, 2);
    assert(wdi.weekday().c_encoding() == i);
    static_assert(noexcept(wdi.weekday()));
    static_assert(cuda::std::is_same_v<cuda::std::chrono::weekday, decltype(wdi.weekday())>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
