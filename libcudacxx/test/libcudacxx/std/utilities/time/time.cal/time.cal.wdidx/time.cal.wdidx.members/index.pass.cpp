//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class weekday_indexed;

// constexpr unsigned index() const noexcept;
//  Returns: index_

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  assert(weekday_indexed{}.index() == 0);

  for (unsigned i = 1; i <= 5; ++i)
  {
    const weekday_indexed wdi(weekday{2}, i);
    assert(static_cast<unsigned>(wdi.index()) == i);

    static_assert(noexcept(wdi.index()));
    static_assert(cuda::std::is_same_v<unsigned, decltype(wdi.index())>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
