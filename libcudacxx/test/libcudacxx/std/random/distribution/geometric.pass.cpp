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
// class geometric_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>

#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct geometric_cdf
{
  using P = typename cuda::std::geometric_distribution<T>::param_type;

  __host__ __device__ double operator()(double x, const P& p) const
  {
    // CDF: F(x; p) = 1 - (1-p)^(floor(x)+1) for x >= 0
    // This represents P(X <= x) where X is the number of failures before the first success
    if (x < 0)
    {
      return 0.0;
    }
    double prob = p.p();
    return 1.0 - cuda::std::pow(1.0 - prob, cuda::std::floor(x) + 1.0);
  }
};

template <class T>
__host__ __device__ void test()
{
  // Cannot be constexpr due to negative_binomial_distribution dependencies
  [[maybe_unused]] const bool test_constexpr = false;
  using D                                    = cuda::std::geometric_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;
  cuda::std::array<P, 5> params              = {P(0.5), P(0.3), P(0.7), P(0.1), P(0.9)};
  test_distribution<D, false, G, test_constexpr>(params, geometric_cdf<T>{});
}

int main(int, char**)
{
  test<int>();
  test<long>();
  return 0;
}
