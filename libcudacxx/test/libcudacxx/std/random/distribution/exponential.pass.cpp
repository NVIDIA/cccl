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
// class exponential_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct exponential_cdf
{
  using P = typename cuda::std::exponential_distribution<T>::param_type;

  __host__ __device__ double operator()(double x, const P& p) const
  {
    if (x <= 0.0)
    {
      return 0.0;
    }

    // CDF of Exponential distribution: F(x; lambda) = 1 - exp(-lambda * x)
    return 1.0 - cuda::std::exp(-p.lambda() * x);
  }
};

template <class T>
__host__ __device__ void test()
{
  [[maybe_unused]] const bool test_constexpr = false;
  using D                                    = cuda::std::exponential_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;
  cuda::std::array<P, 5> params              = {P(T(1)), P(T(0.5)), P(T(2)), P(T(0.1)), P(T(10))};
  test_distribution<D, true, G, test_constexpr>(params, exponential_cdf<T>{});
}

int main(int, char**)
{
  test<double>();
  test<float>();
  return 0;
}
