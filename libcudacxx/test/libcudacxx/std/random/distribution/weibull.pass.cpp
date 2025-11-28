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
// class weibull_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct weibull_cdf
{
  using P = typename cuda::std::weibull_distribution<T>::param_type;

  __host__ __device__ double operator()(double x, const P& p) const
  {
    // CDF of Weibull distribution: F(x) = 1 - exp(-(x/b)^a)
    if (x < 0)
    {
      return 0.0;
    }
    double a = p.a();
    double b = p.b();
    return 1.0 - cuda::std::exp(-cuda::std::pow(x / b, a));
  }
};

template <class T>
__host__ __device__ void test()
{
  // Can be true if/when cuda::std::exp and cuda::std::pow are constexpr
  [[maybe_unused]] const bool test_constexpr = false;
  using D                                    = cuda::std::weibull_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;
  cuda::std::array<P, 5> params              = {P(1, 1), P(2, 1), P(1, 2), P(2, 3), P(5, 0.5)};
  test_distribution<D, true, G, test_constexpr>(params, weibull_cdf<T>{});
}

int main(int, char**)
{
  test<double>();
  test<float>();
  return 0;
}
