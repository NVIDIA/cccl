//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.ranges.find_last.h>
#include <cuda/std/cassert>

#include "test_macros.h"

TEST_FUNC constexpr bool test()
{
  int a[]  = {1, 2, 3, 2};
  auto ret = cuda::std::ranges::find_last(a, a + 4, 2);
  assert(ret.begin() == a + 3);
  assert(ret.end() == a + 4);
  auto ret2 = cuda::std::ranges::find_last(a, 1);
  assert(ret2.begin() == a);
  assert(ret2.end() == a + 4);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
