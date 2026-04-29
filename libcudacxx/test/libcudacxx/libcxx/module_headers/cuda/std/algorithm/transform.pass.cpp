//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.transform.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct transform_double
{
  __host__ __device__ constexpr int operator()(int x) const
  {
    return x * 2;
  }
};
struct transform_add
{
  __host__ __device__ constexpr int operator()(int x, int y) const
  {
    return x + y;
  }
};

__host__ __device__ constexpr bool test()
{
  constexpr int a[] = {1, 2, 3};
  int o[3]          = {};
  cuda::std::transform(a, a + 3, o, transform_double{});
  assert(o[1] == 4);
  constexpr int b[] = {10, 20, 30};
  int o2[3]         = {};
  cuda::std::transform(a, a + 3, b, o2, transform_add{});
  assert(o2[0] == 11 && o2[2] == 33);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
