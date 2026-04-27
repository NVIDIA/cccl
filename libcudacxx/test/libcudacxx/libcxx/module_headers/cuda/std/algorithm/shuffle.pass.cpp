//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.shuffle.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct urbg
{
  using result_type = unsigned;
  unsigned s        = 1;

  __host__ __device__ static constexpr result_type min()
  {
    return 0;
  }
  __host__ __device__ static constexpr result_type max()
  {
    return 0xFFFFFFFFu;
  }
  __host__ __device__ result_type operator()()
  {
    s = s * 48271u % 2147483647u;
    return s;
  }
};

__host__ __device__ bool test()
{
  int a[] = {1, 2, 3, 4};
  urbg g{};
  cuda::std::shuffle(a, a + 4, g);

  return true;
}

int main(int, char**)
{
  test();

  return 0;
}
