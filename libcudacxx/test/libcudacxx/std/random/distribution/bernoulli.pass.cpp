//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: long_tests

// <random>

// class bernoulli_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>

#include "random_utilities/test_distribution.h"
#include "test_macros.h"

__host__ __device__ void test()
{
  const bool test_constexpr = false;
  using D                   = cuda::std::bernoulli_distribution;
  using P                   = D::param_type;
  using G                   = cuda::std::philox4x64;
  auto cdf                  = [] __host__ __device__(cuda::std::int64_t x, P p) {
    // CDF for Bernoulli distribution
    // F(x) = 0 if x < 0
    // F(x) = 1-p if 0 <= x < 1
    // F(x) = 1 if x >= 1
    if (x < 0)
    {
      return 0.0;
    }
    if (x < 1)
    {
      return 1.0 - p.p();
    }
    return 1.0;
  };
  cuda::std::array<P, 5> params = {P(0.5), P(0.1), P(0.9), P(0.25), P(0.75)};
  test_distribution<D, false, G, test_constexpr>(params, cdf);
}

int main(int, char**)
{
  test();
  return 0;
}
