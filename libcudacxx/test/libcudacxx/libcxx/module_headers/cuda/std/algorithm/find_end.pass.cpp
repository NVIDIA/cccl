//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.find_end.h>
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
  constexpr int h[] = {1, 2, 1, 2, 3};
  constexpr int n[] = {1, 2};
  assert(cuda::std::find_end(h, h + 5, n, n + 2) == h + 2);
  assert(cuda::std::find_end(h, h + 5, n, n + 2, equal_to{}) == h + 2);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
