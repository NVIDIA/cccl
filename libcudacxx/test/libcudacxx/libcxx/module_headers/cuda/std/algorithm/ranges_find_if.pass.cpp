//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.ranges.find_if.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct ranges_find_if_even
{
  __host__ __device__ constexpr bool operator()(int x) const
  {
    return x % 2 == 0;
  }
};
struct ranges_find_if_gt2
{
  __host__ __device__ constexpr bool operator()(int x) const
  {
    return x > 2;
  }
};

__host__ __device__ constexpr bool test()
{
  int a[] = {1, 2, 3};
  auto it = cuda::std::ranges::find_if(a, a + 3, ranges_find_if_even{});
  assert(*it == 2);
  auto it2 = cuda::std::ranges::find_if(a, ranges_find_if_gt2{});
  assert(*it2 == 3);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
