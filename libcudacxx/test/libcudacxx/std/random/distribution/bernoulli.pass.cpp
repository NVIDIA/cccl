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

struct bernoulli_cdf
{
  using P = cuda::std::bernoulli_distribution::param_type;

  __host__ __device__ double operator()(cuda::std::int64_t x, const P& p) const
  {
    if (x < 0)
    {
      return 0.0;
    }
    if (x < 1)
    {
      return 1.0 - p.p();
    }
    return 1.0;
  }
};

__host__ __device__ void test()
{
  [[maybe_unused]] const bool test_constexpr = true; // Erroneous compiler warning about unused variable
  using D                                    = cuda::std::bernoulli_distribution;
  using P                                    = D::param_type;
  using G                                    = cuda::std::philox4x64;
  constexpr cuda::std::array<P, 5> params    = {P(0.5), P(0.1), P(0.9), P(0.25), P(0.75)};
  test_distribution<D, false, G, test_constexpr>(params, bernoulli_cdf{});
}

int main(int, char**)
{
  test();
  return 0;
}
