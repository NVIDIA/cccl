//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.ranges.for_each_n.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct acc
{
  int* ps;
  __host__ __device__ void operator()(int x) const
  {
    *ps += x;
  }
};

__host__ __device__ bool test()
{
  int a[] = {1, 2, 3, 4};
  int s   = 0;

  auto res = cuda::std::ranges::for_each_n(a, 3, acc{&s});
  assert(res.in == a + 3 && s == 6);

  return true;
}

int main(int, char**)
{
  test();

  return 0;
}
