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
// class uniform_real_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct uniform_real_cdf
{
  using P = typename cuda::std::uniform_real_distribution<T>::param_type;

  __host__ __device__ double operator()(double x, const P& p) const
  {
    // CDF of uniform distribution: F(x) = (x - a) / (b - a) for a <= x <= b
    double a = p.a();
    double b = p.b();
    if (x < a)
    {
      return 0.0;
    }
    if (x > b)
    {
      return 1.0;
    }
    return (x - a) / (b - a);
  }
};

template <class T>
__host__ __device__ void test()
{
  [[maybe_unused]] const bool test_constexpr = true;
  using D                                    = cuda::std::uniform_real_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;
  cuda::std::array<P, 5> params              = {P(0, 1), P(-5, 5), P(10, 20), P(0, 100), P(-10, 0)};
  test_distribution<D, true, G, test_constexpr>(params, uniform_real_cdf<T>{});
}

int main(int, char**)
{
  test<double>();
  test<float>();
  return 0;
}
