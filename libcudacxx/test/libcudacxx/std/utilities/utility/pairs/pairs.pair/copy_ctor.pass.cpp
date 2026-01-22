//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// pair(const pair&) = default;

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  {
    using P1 = cuda::std::pair<int, short>;
    P1 p1(3, static_cast<short>(4));
    P1 p2 = p1;
    assert(p2.first == 3);
    assert(p2.second == 4);
  }
  {
    using P1 = cuda::std::pair<int, short>;
    constexpr P1 p1(3, static_cast<short>(4));
    constexpr P1 p2 = p1;
    static_assert(p2.first == 3, "");
    static_assert(p2.second == 4, "");
  }

  return 0;
}
