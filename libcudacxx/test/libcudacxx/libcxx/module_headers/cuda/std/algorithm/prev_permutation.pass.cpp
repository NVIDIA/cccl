//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.prev_permutation.h>
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
  int a[] = {3, 2, 1};
  assert(cuda::std::prev_permutation(a, a + 3));
  assert(cuda::std::prev_permutation(a, a + 3, greater{}));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
