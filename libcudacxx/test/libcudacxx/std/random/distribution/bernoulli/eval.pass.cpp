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

// template<class _URng> result_type operator()(_URNG& g);

#include <cuda/std/__memory_>
#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstddef>
#include <cuda/std/numeric>
#include <cuda/std/span>

#include "random_utilities/test_discrete_distribution.h"
#include "test_macros.h"

__host__ __device__ void test()
{
  using D     = cuda::std::bernoulli_distribution;
  using G     = cuda::std::philox4x64;
  const int n = 1000;
  auto result = test_discrete_distribution<D, G, 2>(D::param_type{0.5}, {0.5, 0.5}, n);
  assert(result);
  result = test_discrete_distribution<D, G, 2>(D::param_type{0.2}, {0.8, 0.2}, n);
  assert(result);
  result = test_discrete_distribution<D, G, 2>(D::param_type{0.99}, {0.01, 0.99}, n);
  assert(result);
  result = test_discrete_distribution<D, G, 2>(D::param_type{0.0}, {1.0, 0.0}, n);
  assert(result);
}

int main(int, char**)
{
  test();
  return 0;
}
