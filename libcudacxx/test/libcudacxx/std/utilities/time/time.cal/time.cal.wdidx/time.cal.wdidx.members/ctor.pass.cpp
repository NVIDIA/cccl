//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class weekday_indexed;

//                     weekday_indexed() = default;
//  constexpr weekday_indexed(const chrono::weekday& wd, unsigned index) noexcept;
//
//  Effects: Constructs an object of type weekday_indexed by initializing wd_ with wd and index_ with index.
//    The values held are unspecified if !wd.ok() or index is not in the range [1, 5].
//
//  constexpr chrono::weekday weekday() const noexcept;
//  constexpr unsigned        index()   const noexcept;
//  constexpr bool ok()                 const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  weekday_indexed wdi0{};
  assert(wdi0.weekday() == weekday{});
  assert(wdi0.index() == 0);
  assert(!wdi0.ok());
  static_assert(noexcept(weekday_indexed{}));

  weekday_indexed wdi1{cuda::std::chrono::Sunday, 2};
  assert(wdi1.weekday() == cuda::std::chrono::Sunday);
  assert(wdi1.index() == 2);
  assert(wdi1.ok());
  static_assert(noexcept(weekday_indexed(weekday{1}, 1)));

  constexpr auto Tuesday = cuda::std::chrono::Tuesday;
  for (unsigned i = 1; i <= 5; ++i)
  {
    weekday_indexed wdi(Tuesday, i);
    assert(wdi.weekday() == Tuesday);
    assert(wdi.index() == i);
    assert(wdi.ok());
  }

  for (unsigned i = 6; i <= 20; ++i)
  {
    weekday_indexed wdi(Tuesday, i);
    assert(!wdi.ok());
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
