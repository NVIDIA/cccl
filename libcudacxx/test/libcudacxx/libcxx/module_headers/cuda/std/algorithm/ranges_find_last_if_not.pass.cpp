//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.ranges.find_last_if_not.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct ranges_find_last_if_not_odd
{
  TEST_FUNC constexpr bool operator()(int x) const
  {
    return x % 2 != 0;
  }
};
struct ranges_find_last_if_not_ge3
{
  TEST_FUNC constexpr bool operator()(int x) const
  {
    return x >= 3;
  }
};

TEST_FUNC constexpr bool test()
{
  int a[]  = {1, 2, 3, 4, 5};
  auto ret = cuda::std::ranges::find_last_if_not(a, a + 5, ranges_find_last_if_not_odd{});
  assert(ret.begin() == a + 3);
  assert(ret.end() == a + 5);
  auto ret2 = cuda::std::ranges::find_last_if_not(a, ranges_find_last_if_not_ge3{});
  assert(ret2.begin() == a + 1);
  assert(ret2.end() == a + 5);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
