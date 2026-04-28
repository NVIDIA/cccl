//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.unique_copy.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct equal_to
{
  __host__ __device__ constexpr bool operator()(int a, int b) const
  {
    return a == b;
  }
};

__host__ __device__ constexpr bool test()
{
  constexpr int a[] = {1, 1, 2, 2, 3};
  int o[5]          = {};
  auto r            = cuda::std::unique_copy(a, a + 5, o);
  assert(r == o + 3 && o[0] == 1 && o[1] == 2 && o[2] == 3);
  auto r2 = cuda::std::unique_copy(a, a + 5, o, equal_to{});
  assert(r2 == o + 3);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
