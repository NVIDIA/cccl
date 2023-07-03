//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// friend constexpr bool operator==(const iterator& x, const sentinel& y);

#include <cuda/std/ranges>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../types.h"

__host__ __device__ constexpr bool test() {
  {
    const cuda::std::ranges::iota_view<int, IntComparableWith<int>> io(0, IntComparableWith<int>(10));
    auto iter = io.begin();
    auto sent = io.end();
    assert(iter != sent);
    assert(iter + 10 == sent);
  }
  {
    cuda::std::ranges::iota_view<int, IntComparableWith<int>> io(0, IntComparableWith<int>(10));
    auto iter = io.begin();
    auto sent = io.end();
    assert(iter != sent);
    assert(iter + 10 == sent);
  }
  {
    const cuda::std::ranges::iota_view io(SomeInt(0), IntComparableWith(SomeInt(10)));
    auto iter = io.begin();
    auto sent = io.end();
    assert(iter != sent);
    assert(iter + 10 == sent);
  }
  {
    cuda::std::ranges::iota_view io(SomeInt(0), IntComparableWith(SomeInt(10)));
    auto iter = io.begin();
    auto sent = io.end();
    assert(iter != sent);
    assert(iter + 10 == sent);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
