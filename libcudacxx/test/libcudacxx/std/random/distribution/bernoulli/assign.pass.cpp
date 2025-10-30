//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// <random>

// class bernoulli_distribution

// bernoulli_distribution& operator=(const bernoulli_distribution&);

#include <cuda/std/__random_>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using D = cuda::std::bernoulli_distribution;
  D d1(0.75);
  D d2;
  assert(d1 != d2);
  d2 = d1;
  assert(d1 == d2);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
