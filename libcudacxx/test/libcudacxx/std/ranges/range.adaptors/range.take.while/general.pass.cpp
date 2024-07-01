//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// Some basic examples of how take_while_view might be used in the wild. This is a general
// collection of sample algorithms and functions that try to mock general usage of
// this view.

// #include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/ranges>

template <class Range, class Expected>
__host__ __device__ constexpr bool equal(Range&& range, Expected&& expected)
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
