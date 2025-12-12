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

struct binomial_cdf
{
  using P = cuda::std::binomial_distribution<>::param_type;

  __host__ __device__ double operator()(cuda::std::int64_t x, P p) const
  {
    if (x < 0)
    {
      return 0.0;
    }
    if (x >= p.t())
    {
      return 1.0;
    }

    double sum  = 0.0;
    double q    = 1.0 - p.p();
    double prob = cuda::std::pow(q, static_cast<double>(p.t()));
    sum += prob;
    for (int i = 1; i <= x && i < p.t(); ++i)
    {
      prob *= (static_cast<double>(p.t() - i + 1) * p.p()) / (static_cast<double>(i) * q);
      sum += prob;
    }
    return sum;
  }
};

__host__ __device__ void test()
{
  [[maybe_unused]] const bool test_constexpr = false; // Math functions cuda::std::log, cuda::std::exp are not yet
                                                      // constexpr
  using D                       = cuda::std::binomial_distribution<>;
  using P                       = D::param_type;
  using G                       = cuda::std::philox4x64;
  cuda::std::array<P, 5> params = {P(10, 0.5), P(20, 0.3), P(15, 0.7), P(5, 0.25), P(30, 0.6)};
  test_distribution<D, false, G, test_constexpr>(params, binomial_cdf{});
}

int main(int, char**)
{
  test();
  return 0;
}
