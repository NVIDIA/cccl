//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class weekday_last;

//  constexpr chrono::weekday weekday() const noexcept;
//  Returns: wd_

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using weekday      = cuda::std::chrono::weekday;
  using weekday_last = cuda::std::chrono::weekday_last;

  for (unsigned i = 0; i <= 255; ++i)
  {
    const weekday_last wdl{weekday{i}};
    assert(wdl.weekday() == weekday{i});
    static_assert(noexcept(wdl.weekday()));
    static_assert(cuda::std::is_same_v<weekday, decltype(wdl.weekday())>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
