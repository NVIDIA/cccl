//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.max.h>
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

  assert(cuda::std::max(one, two) == two);
  assert(cuda::std::max(one, two, greater{}) == one);
  assert(cuda::std::max({three, one, two}) == three);
  assert(cuda::std::max({three, one, two}, greater{}) == one);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
