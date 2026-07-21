//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: a return statement inside a loop is not currently supported in a tile function

// Some basic examples of how take_while_view might be used in the wild. This is a general
// collection of sample algorithms and functions that try to mock general usage of
// this view.

// #include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

template <class Range, class Expected>
TEST_FUNC constexpr bool equal(Range&& range, Expected&& expected)
{
  auto irange    = range.begin();
  auto iexpected = cuda::std::begin(expected);
  for (; irange != range.end(); ++irange, ++iexpected)
  {
    if (*irange != *iexpected)
    {
      return false;
    }
  }
  return true;
}

int main(int, char**)
{
  {
    auto input = {0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0};
    auto small = [](const int x) noexcept {
      return x < 5;
    };
    auto small_ints = input | cuda::std::views::take_while(small);
    auto expected   = {0, 1, 2, 3, 4};
    assert(equal(small_ints, expected));
  }
  return 0;
}
