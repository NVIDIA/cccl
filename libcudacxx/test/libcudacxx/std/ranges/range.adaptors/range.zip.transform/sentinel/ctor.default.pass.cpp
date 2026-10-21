//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// sentinel() = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"

struct PODSentinel
{
  bool b; // deliberately uninitialised

  TEST_FUNC friend constexpr bool operator==(int*, const PODSentinel& s)
  {
    return s.b;
  }
};

struct Fn
{
  TEST_FUNC int operator()(auto&&...) const
  {
    return 5;
  }
};

struct Range : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin() const;
  TEST_FUNC PODSentinel end();
};

TEST_FUNC constexpr bool test()
{
  {
    using R        = cuda::std::ranges::zip_transform_view<Fn, Range>;
    using Sentinel = cuda::std::ranges::sentinel_t<R>;
    static_assert(!cuda::std::is_same_v<Sentinel, cuda::std::ranges::iterator_t<R>>);

    cuda::std::ranges::iterator_t<R> it;

    Sentinel s1;
    assert(it != s1); // PODSentinel.b is initialised to false

    Sentinel s2 = {};
    assert(it != s2); // PODSentinel.b is initialised to false
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
