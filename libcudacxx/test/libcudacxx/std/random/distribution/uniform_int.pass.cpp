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
// class uniform_int_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>

#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct uniform_int_cdf
{
  using P = typename cuda::std::uniform_int_distribution<T>::param_type;

  __host__ __device__ double operator()(double x, const P& p) const
  {
    // CDF: F(x; a, b) = (floor(x) - a + 1) / (b - a + 1) for a <= x <= b
    //                 = 0 for x < a
    //                 = 1 for x > b
    double a = static_cast<double>(p.a());
    double b = static_cast<double>(p.b());
    if (x < a)
    {
      return 0.0;
    }
    if (x > b)
    {
      return 1.0;
    }
    return (cuda::std::floor(x) - a + 1.0) / (b - a + 1.0);
  }
};

template <class T>
__host__ __device__ void test()
{
  [[maybe_unused]] const bool test_constexpr = true;
  using D                                    = cuda::std::uniform_int_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;
  cuda::std::array<P, 5> params              = {
    P(0, 10),
    P(1, 100),
    P(-50, 50),
    P(cuda::std::numeric_limits<T>::min(), cuda::std::numeric_limits<T>::max()),
    P(100, 1000)};
  test_distribution<D, false, G, test_constexpr>(params, uniform_int_cdf<T>{});
}

int main(int, char**)
{
  test<int>();
  test<long>();
  test<short>();
  return 0;
}
