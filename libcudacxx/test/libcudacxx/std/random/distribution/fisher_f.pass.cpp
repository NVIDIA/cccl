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
// class fisher_f_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "random_utilities/stats_functions.h"
#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct fisher_f_cdf
{
  using P = typename cuda::std::fisher_f_distribution<T>::param_type;

  __host__ __device__ double operator()(double x, const P& p) const
  {
    // CDF: F(x; m, n) = I_{mx/(mx+n)}(m/2, n/2)
    // where I is the regularized incomplete beta function
    if (x <= 0)
    {
      return 0.0;
    }
    double m_val = p.m();
    double n_val = p.n();
    double z     = (m_val * x) / (m_val * x + n_val);
    return incomplete_beta(m_val / 2.0, n_val / 2.0, z);
  }
};

template <class T>
__host__ __device__ void test()
{
  [[maybe_unused]] const bool test_constexpr = false;
  using D                                    = cuda::std::fisher_f_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;
  cuda::std::array<P, 5> params              = {P(1, 1), P(5, 2), P(10, 10), P(2, 5), P(20, 30)};
  test_distribution<D, true, G, test_constexpr>(params, fisher_f_cdf<T>{});
}

int main(int, char**)
{
  test<double>();
  test<float>();
  return 0;
}
