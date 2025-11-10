//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/std/array>
#include <cuda/std/cstddef>

#include "test_macros.h"

template <class DiscreteDistribution, class URNG>
__host__ __device__ constexpr bool test_constexpr()
{
  typename DiscreteDistribution::param_type param;
  DiscreteDistribution dist(param);
  URNG g{};
  unused(dist(g, param));
  unused(dist(g));
  return true;
}

// Perform a chi-squared test, comparing the observed and expected frequencies
// of outcomes from a discrete distribution. Tests to significance level 0.01. Accepts 2-10 buckets.
template <class DiscreteDistribution, class URNG, cuda::std::size_t N>
__host__ __device__ bool test_discrete_distribution(
  const typename DiscreteDistribution::param_type param,
  const cuda::std::array<double, N>& expected_probabilities,
  const cuda::std::size_t num_samples)
{
  static_assert(test_constexpr<DiscreteDistribution, URNG>());

  // First check the operator with param is equivalent to the constructor param
  {
    DiscreteDistribution d(param);
    URNG g_1{};
    URNG g_2{};
    for (cuda::std::size_t i = 0; i < 100; ++i)
    {
      auto dist_val  = d(g_1, param);
      auto dist2_val = d(g_2);
      assert(dist_val == dist2_val);
    }
  }

  DiscreteDistribution dist(param);
  URNG g{};
  assert(expected_probabilities.size() >= 2 && expected_probabilities.size() <= 10);
  cuda::std::array<cuda::std::size_t, N> observed_frequencies{};

  // Generate samples and record observed frequencies.
  for (cuda::std::size_t i = 0; i < num_samples; ++i)
  {
    ++observed_frequencies[static_cast<int>(dist(g, param))];
  }

  // Compute chi-squared statistic.
  double chi_squared = 0.0;
  for (cuda::std::size_t i = 0; i < N; ++i)
  {
    double expected = expected_probabilities[i] * num_samples;
    // Prevent division by zero when expecting zero occurrences
    if (expected == 0.0)
    {
      expected = 1e-16;
    }
    double observed = static_cast<double>(observed_frequencies[i]);
    double diff     = observed - expected;
    chi_squared += (diff * diff) / expected;
  }
  // Critical value for chi-squared distribution with (num_buckets - 1) degrees of freedom at significance level 0.01
  const cuda::std::array<double, 11> critical_values = {
    0.0, 6.635, 9.210, 11.345, 13.277, 15.086, 16.812, 18.475, 20.090, 21.666, 23.209};

  double critical_value = critical_values.at(N - 1);
  return chi_squared < critical_value;
}
