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
// class lognormal_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct lognormal_cdf
{
  using P = typename cuda::std::lognormal_distribution<T>::param_type;

  __host__ __device__ double operator()(double x, const P& p) const
  {
    // CDF: F(x; m, s) = 0.5 * (1 + erf((ln(x) - m) / (s * sqrt(2))))
    if (x <= 0.0)
    {
      return 0.0;
    }
    double m = static_cast<double>(p.m());
    double s = static_cast<double>(p.s());
    return 0.5 * (1.0 + cuda::std::erf((cuda::std::log(x) - m) / (s * cuda::std::sqrt(2.0))));
  }
};

template <class T>
__host__ __device__ void test()
{
  // Cannot be constexpr due to log/exp functions
  [[maybe_unused]] const bool test_constexpr = false;
  using D                                    = cuda::std::lognormal_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;
  cuda::std::array<P, 5> params              = {P(0, 1), P(1, 0.5), P(-1, 2), P(2, 0.25), P(0.5, 1.5)};
  test_distribution<D, true, G, test_constexpr>(params, lognormal_cdf<T>{});
}

int main(int, char**)
{
  test<double>();
  test<float>();
  return 0;
}
