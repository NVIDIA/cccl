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

// template<class IntType = int>
// class negative_binomial_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>

#include "random_utilities/stats_functions.h"
#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct negative_binomial_cdf
{
  using P = typename cuda::std::negative_binomial_distribution<T>::param_type;

  __host__ __device__ double operator()(double x, const P& p) const
  {
    // CDF: F(x; k, p) = I_p(k, x+1) where I_p is the regularized incomplete beta function
    // This represents P(X <= x) where X is the number of failures before k successes
    if (x < 0)
    {
      return 0.0;
    }
    double k    = static_cast<double>(p.k());
    double prob = p.p();
    return incomplete_beta(k, cuda::std::floor(x) + 1.0, prob);
  }
};

template <class T>
__host__ __device__ void test()
{
  // Cannot be constexpr due to gamma_distribution and log/exp
  [[maybe_unused]] const bool test_constexpr = false;
  using D                                    = cuda::std::negative_binomial_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;
  cuda::std::array<P, 5> params              = {P(5, 0.5), P(10, 0.3), P(3, 0.7), P(15, 0.4), P(20, 0.6)};
  test_distribution<D, false, G, test_constexpr>(params, negative_binomial_cdf<T>{});
}

int main(int, char**)
{
  test<int>();
  test<long>();
  return 0;
}
