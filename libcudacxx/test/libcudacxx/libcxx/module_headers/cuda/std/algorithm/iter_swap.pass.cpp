//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.iter_swap.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct iter_swap_iter
{
  int v;
};

__host__ __device__ constexpr void iter_swap(iter_swap_iter* a, iter_swap_iter* b)
{
  cuda::std::swap(a->v, b->v);
}

__host__ __device__ constexpr bool test()
{
  int x = 1, y = 2;
  cuda::std::iter_swap(&x, &y);
  assert(x == 2 && y == 1);
  iter_swap_iter p{3}, q{4};
  cuda::std::iter_swap(&p, &q);
  assert(p.v == 4 && q.v == 3);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
