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

// template<class _URNG> result_type operator()(_URNG& g);

#include <cuda/std/__memory_>
#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/numeric>
#include <cuda/std/span>

#include "random_utilities/test_continuous_distribution.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test()
{
  using D                       = cuda::std::normal_distribution<T>;
  using P                       = D::param_type;
  using G                       = cuda::std::philox4x64;
  cuda::std::array<P, 5> params = {P(0, 1), P(10, 2), P(-5, 0.5), P(4, 5), P(1000, 100)};
  for (auto p : params)
  {
    bool res = test_continuous_distribution<D, G>(p, [=] __host__ __device__(double x) {
      // CDF for normal distribution
      return 0.5 * (1 + cuda::std::erf((x - (p.mean())) / (p.stddev() * cuda::std::sqrt(2))));
    });
    assert(res);
  }
}

int main(int, char**)
{
  test<double>();
  test<float>();
  return 0;
}
