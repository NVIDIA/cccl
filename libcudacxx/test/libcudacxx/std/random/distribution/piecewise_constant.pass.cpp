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
// class piecewise_constant_distribution

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct piecewise_constant_cdf
{
  using P = typename cuda::std::piecewise_constant_distribution<T>::param_type;

  __host__ __device__ double operator()(double x, const P& p) const
  {
    // CDF for piecewise constant distribution
    auto intervals = p.intervals();
    auto densities = p.densities();

    if (intervals.size() < 2)
    {
      return 0.0;
    }

    if (x <= intervals[0])
    {
      return 0.0;
    }
    if (x >= intervals[intervals.size() - 1])
    {
      return 1.0;
    }

    // Find which interval x falls into and compute CDF
    double cdf = 0.0;
    for (size_t i = 0; i < intervals.size() - 1; ++i)
    {
      if (x <= intervals[i + 1])
      {
        // x is in this interval
        cdf += densities[i] * (x - intervals[i]);
        break;
      }
      else
      {
        // Add the entire interval's contribution
        cdf += densities[i] * (intervals[i + 1] - intervals[i]);
      }
    }
    return cdf;
  }
};

class constant_weight_functor
{
public:
  __host__ __device__ double operator()(double x) const
  {
    return 1.0;
  }
};

template <class T>
__host__ __device__ void test()
{
  // Cannot be constexpr due to unique_ptr usage
  [[maybe_unused]] const bool test_constexpr = false;
  using D                                    = cuda::std::piecewise_constant_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;

  cuda::std::array<T, 3> intervals1 = {0, 0.5, 1};
  cuda::std::array<T, 2> weights1   = {3, 7};

  // Iterator
  D d(intervals1.begin(), intervals1.end(), weights1.begin());
  assert(d.min() == 0);
  assert(d.max() == 1);
  assert(d.intervals().size() == 3);
  assert(d.densities().size() == 2);

  // Initializer list
  D d2({0, 0.5, 1}, constant_weight_functor{});
  assert(d2.min() == 0);
  assert(d2.max() == 1);
  assert(d2.intervals().size() == 3);
  assert(d2.densities().size() == 2);

  // Constant boundaries
  D d3(5, -10.0, 10.0, constant_weight_functor{});
  assert(d3.min() == -10.0);
  assert(d3.max() == 10.0);
  assert(d3.intervals().size() == 6);
  assert(d3.densities().size() == 5);

  cuda::std::array<P, 3> params = {
    P(intervals1.begin(), intervals1.end(), weights1.begin()),
    P({0, 0.5, 1}, constant_weight_functor{}),
    P(5, -10.0, 10.0, constant_weight_functor{})};

  test_distribution<D, true, G, test_constexpr>(params, piecewise_constant_cdf<T>{});
}

int main(int, char**)
{
  test<double>();
  test<float>();
  return 0;
}
