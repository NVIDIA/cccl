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
// class student_t_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "random_utilities/stats_functions.h"
#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct student_t_cdf
{
  using P = typename cuda::std::student_t_distribution<T>::param_type;

  __host__ __device__ double operator()(double x, const P& p) const
  {
    // CDF of Student's t-distribution: F(x) = 0.5 + 0.5 * sgn(x) * I_{t²/(n+t²)}(0.5, n/2)
    // where I is the regularized incomplete beta function and t = x
    double t2       = x * x;
    double n        = p.n();
    double ratio    = t2 / (n + t2);
    double beta_val = incomplete_beta(0.5, n / 2.0, ratio);

    if (x >= 0)
    {
      return 0.5 + 0.5 * beta_val;
    }
    else
    {
      return 0.5 - 0.5 * beta_val;
    }
  }
};

template <class T>
__host__ __device__ void test()
{
  [[maybe_unused]] const bool test_constexpr = false;
  using D                                    = cuda::std::student_t_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;
  cuda::std::array<P, 5> params              = {P(1), P(2), P(5), P(10), P(30)};
  test_distribution<D, true, G, test_constexpr>(params, student_t_cdf<T>{});
}

int main(int, char**)
{
  test<double>();
  test<float>();
  return 0;
}
