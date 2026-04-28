//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.set_difference.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct less
{
  __host__ __device__ constexpr bool operator()(int a, int b) const
  {
    return a < b;
  }
};

__host__ __device__ constexpr bool test()
{
  constexpr int a[] = {1, 2, 3};
  constexpr int b[] = {2, 4};
  int o[3]          = {};
  auto r            = cuda::std::set_difference(a, a + 3, b, b + 2, o);
  assert(r == o + 2);
  int o2[3] = {};
  auto r2   = cuda::std::set_difference(a, a + 3, b, b + 2, o2, less{});
  assert(r2 == o2 + 2);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
