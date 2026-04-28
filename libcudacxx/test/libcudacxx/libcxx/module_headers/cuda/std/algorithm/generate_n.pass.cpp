//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.generate_n.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct gen
{
  int* pc;
  __host__ __device__ int operator()() const
  {
    return ++(*pc);
  }
};

__host__ __device__ bool test()
{
  int a[3] = {};
  int c    = 0;

  auto r = cuda::std::generate_n(a, 3, gen{&c});
  assert(r == a + 3 && a[2] == 3);

  return true;
}

int main(int, char**)
{
  test();

  return 0;
}
