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
// class poisson_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct poisson_cdf
{
  using P = typename cuda::std::poisson_distribution<T>::param_type;

  __host__ __device__ double operator()(cuda::std::uint64_t x, const P& p) const
  {
    double sum  = 0;
    double mean = p.mean();
    for (cuda::std::uint64_t i = 0; i <= x; ++i)
    {
      double term = cuda::std::exp(-mean);
      for (cuda::std::uint64_t j = 1; j <= i; ++j)
      {
        term *= mean / j;
      }
      sum += term;
    }
    return sum;
  }
};

template <class T>
__host__ __device__ void test()
{
  [[maybe_unused]] const bool test_constexpr = false;
  using D                                    = cuda::std::poisson_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;
  cuda::std::array<P, 3> params              = {P(1.0), P(20.0), P(50.0)};
  test_distribution<D, false, G, test_constexpr>(params, poisson_cdf<T>{});
}

int main(int, char**)
{
  test<int>();
  test<long>();
  test<cuda::std::uint32_t>();
  test<cuda::std::uint64_t>();
  return 0;
}
