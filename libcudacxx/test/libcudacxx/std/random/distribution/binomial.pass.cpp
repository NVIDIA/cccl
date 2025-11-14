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

// class binomial_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>

#include "random_utilities/test_distribution.h"
#include "test_macros.h"

__host__ __device__ void test()
{
  const bool test_constexpr = false; // binomial_distribution is not yet constexpr
  using D                   = cuda::std::binomial_distribution<>;
  using P                   = D::param_type;
  using G                   = cuda::std::philox4x64;
  auto cdf                  = [] __host__ __device__(cuda::std::int64_t x, P p) {
    // CDF for Binomial distribution
    // F(x) = sum(i=0 to floor(x)) of C(t,i) * p^i * (1-p)^(t-i)
    // For simplicity in testing, we'll use a numerical approximation
    if (x < 0)
    {
      return 0.0;
    }
    if (x >= p.t())
    {
      return 1.0;
    }
    // Compute binomial CDF
    double sum  = 0.0;
    double prob = cuda::std::pow(1.0 - p.p(), static_cast<double>(p.t()));
    sum += prob;
    for (int i = 0; i <= x && i < p.t(); ++i)
    {
      if (i > 0)
      {
        prob *= (static_cast<double>(p.t() - i + 1) * p.p()) / (static_cast<double>(i) * (1.0 - p.p()));
        sum += prob;
      }
    }
    return sum;
  };
  constexpr cuda::std::array<P, 5> params = {P(10, 0.5), P(20, 0.3), P(15, 0.7), P(5, 0.25), P(30, 0.6)};
  test_distribution<D, false, G, test_constexpr>(params, cdf);
}

int main(int, char**)
{
  test();
  return 0;
}
