//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class weekday_indexed;

// constexpr bool operator==(const weekday_indexed& x, const weekday_indexed& y) noexcept;
//   Returns: x.weekday() == y.weekday() && x.index() == y.index().
// constexpr bool operator!=(const weekday_indexed& x, const weekday_indexed& y) noexcept;
//   Returns: !(x == y)

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  AssertEqualityAreNoexcept<weekday_indexed>();
  AssertEqualityReturnBool<weekday_indexed>();

  assert((weekday_indexed{} == weekday_indexed{}));
  assert(!(weekday_indexed{} != weekday_indexed{}));

  assert(!(weekday_indexed{} == weekday_indexed{cuda::std::chrono::Tuesday, 1}));
  assert((weekday_indexed{} != weekday_indexed{cuda::std::chrono::Tuesday, 1}));

  //  Some 'ok' values as well
  assert((weekday_indexed{weekday{1}, 2} == weekday_indexed{weekday{1}, 2}));
  assert(!(weekday_indexed{weekday{1}, 2} != weekday_indexed{weekday{1}, 2}));

  assert(!(weekday_indexed{weekday{1}, 2} == weekday_indexed{weekday{1}, 1}));
  assert((weekday_indexed{weekday{1}, 2} != weekday_indexed{weekday{1}, 1}));
  assert(!(weekday_indexed{weekday{1}, 2} == weekday_indexed{weekday{2}, 2}));
  assert((weekday_indexed{weekday{1}, 2} != weekday_indexed{weekday{2}, 2}));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
