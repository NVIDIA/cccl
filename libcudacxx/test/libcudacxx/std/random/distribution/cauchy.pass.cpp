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
// class cauchy_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/numbers>

#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct cauchy_cdf
{
  using P = typename cuda::std::cauchy_distribution<T>::param_type;

  __host__ __device__ double operator()(double x, const P& p) const
  {
    // CDF of Cauchy distribution: F(x; a, b) = (1/π) * arctan((x - a) / b) + 0.5
    return (1.0 / cuda::std::numbers::pi) * cuda::std::atan((x - p.a()) / p.b()) + 0.5;
  }
};

template <class T>
__host__ __device__ void test()
{
  [[maybe_unused]] const bool test_constexpr = false;
  using D                                    = cuda::std::cauchy_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;
  cuda::std::array<P, 5> params              = {P(0, 1), P(10, 2), P(-5, 0.5), P(4, 5), P(1000, 100)};
  test_distribution<D, true, G, test_constexpr>(params, cauchy_cdf<T>{});
}

int main(int, char**)
{
  test<double>();
  test<float>();
  return 0;
}
