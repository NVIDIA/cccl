//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.search_n.h>
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
  constexpr int h[] = {1, 2, 2, 2, 3};
  assert(cuda::std::search_n(h, h + 5, 3, 2) == h + 1);
  assert(cuda::std::search_n(h, h + 5, 3, 2, equal_to{}) == h + 1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
