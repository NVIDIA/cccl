//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.unique.h>
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
  int a[] = {1, 1, 2, 2, 3};
  auto r  = cuda::std::unique(a, a + 5);
  assert(r == a + 3 && a[0] == 1 && a[1] == 2 && a[2] == 3);
  auto r2 = cuda::std::unique(a, a + 3, equal_to{});
  assert(r2 == a + 3);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
