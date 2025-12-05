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

// template<class RealType = double>
// class gamma_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "random_utilities/stats_functions.h"
#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct gamma_cdf
{
  using P = typename cuda::std::gamma_distribution<T>::param_type;

  __host__ __device__ double operator()(double x, const P& p) const
  {
    if (x <= 0.0)
    {
      return 0.0;
    }

    // Standardize: x' = x / beta
    double x_std = x / p.beta();
    double alpha = p.alpha();

    // CDF of Gamma distribution: F(x; alpha, beta) = P(alpha, x/beta) / Gamma(alpha)
    // where P is the regularized incomplete gamma function
    return incomplete_gamma(alpha, x_std);
  }
};

template <class T>
__host__ __device__ void test()
{
  [[maybe_unused]] const bool test_constexpr = false;
  using D                                    = cuda::std::gamma_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;
  cuda::std::array<P, 5> params              = {P(1, 1), P(2, 1), P(0.5, 2), P(5, 0.5), P(10, 2)};
  test_distribution<D, true, G, test_constexpr>(params, gamma_cdf<T>{});
}

int main(int, char**)
{
  test<double>();
  test<float>();
  return 0;
}
