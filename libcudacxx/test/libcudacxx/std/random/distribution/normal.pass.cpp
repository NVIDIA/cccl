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
// class normal_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct normal_cdf
{
  using P = typename cuda::std::normal_distribution<T>::param_type;

  __host__ __device__ double operator()(double x, const P& p) const
  {
    return 0.5 * (1 + cuda::std::erf((x - (p.mean())) / (p.stddev() * cuda::std::sqrt(2))));
  }
};

template <class T>
__host__ __device__ void test()
{
  // Can be true if/when cuda::std::log is constexpr
  const bool test_constexpr     = false;
  using D                       = cuda::std::normal_distribution<T>;
  using P                       = typename D::param_type;
  using G                       = cuda::std::philox4x64;
  cuda::std::array<P, 5> params = {P(0, 1), P(10, 2), P(-5, 0.5), P(4, 5), P(1000, 100)};
  test_distribution<D, true, G, test_constexpr>(params, normal_cdf<T>{});
}

int main(int, char**)
{
  test<double>();
  test<float>();
  return 0;
}
