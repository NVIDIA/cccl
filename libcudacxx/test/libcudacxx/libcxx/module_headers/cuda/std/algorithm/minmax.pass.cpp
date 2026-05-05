//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.minmax.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct greater
{
  __host__ __device__ constexpr bool operator()(int a, int b) const
  {
    return a > b;
  }
};

__host__ __device__ constexpr bool test()
{
  const int one{1};
  const int two{2};
  const int three{3};

  auto a = cuda::std::minmax(one, two);
  assert(a.first == 1 && a.second == 2);
  auto b = cuda::std::minmax(one, two, greater{});
  assert(b.first == 2 && b.second == 1);
  auto c = cuda::std::minmax({three, one, two});
  assert(c.first == 1 && c.second == 3);
  auto d = cuda::std::minmax({three, one, two}, greater{});
  assert(d.first == 3 && d.second == 1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
