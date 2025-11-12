//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef LIBCUDACXX_TEST_SUPPORT_RANDOM_UTILITIES_TEST_CONTINUOUS_DISTRIBUTION_H
#define LIBCUDACXX_TEST_SUPPORT_RANDOM_UTILITIES_TEST_CONTINUOUS_DISTRIBUTION_H
#include <cuda/std/__algorithm/partial_sort.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>

#include "test_macros.h"

namespace detail
{
template <class ContinuousDistribution, class URNG>
__host__ __device__ constexpr bool test_constexpr()
{
  typename ContinuousDistribution::param_type param;
  ContinuousDistribution dist(param);
  URNG g{};
  unused(dist(g, param));
  unused(dist(g));
  return true;
}
} // namespace detail

// Perform a kolmogorov-Smirnov test, comparing the observed and expected cumulative
// distribution function from a continuous distribution.
// Generates a fixed size of 10000 samples
template <class ContinuousDistribution, class URNG, bool test_constexpr, class CDF>
__host__ __device__ bool test_continuous_distribution(const typename ContinuousDistribution::param_type param, CDF cdf)
{
  if constexpr (test_constexpr)
  {
    static_assert(detail::test_constexpr<ContinuousDistribution, URNG>());
  }

  //  First check the operator with param is equivalent to the constructor param
  {
    ContinuousDistribution d1(param);
    ContinuousDistribution d2(param);
    URNG g_1{};
    URNG g_2{};
    for (cuda::std::size_t i = 0; i < 100; ++i)
    {
      auto dist_val  = d1(g_1, param);
      auto dist2_val = d2(g_2);
      assert(dist_val == dist2_val);
    }
  }

  ContinuousDistribution dist(param);
  URNG g{};
  const cuda::std::size_t num_samples = 10000;

  auto samples    = cuda::std::make_unique<typename ContinuousDistribution::result_type[]>(num_samples);
  auto cdf_values = cuda::std::make_unique<typename ContinuousDistribution::result_type[]>(num_samples);
  for (cuda::std::size_t i = 0; i < num_samples; ++i)
  {
    samples[i] = dist(g, param);
  }
  // Sort the samples
  // Use sort when available
  cuda::std::partial_sort(samples.get(), samples.get() + num_samples, samples.get() + num_samples);
  // Compute the CDF values for each sample
  for (cuda::std::size_t i = 0; i < num_samples; ++i)
  {
    cdf_values[i] = cdf(samples[i]);
  }
  // Compute the KS statistic
  double d_max = 0.0;
  for (cuda::std::size_t i = 0; i < num_samples; ++i)
  {
    double empirical_cdf_upper = static_cast<double>(i + 1) / static_cast<double>(num_samples);
    double empirical_cdf_lower = static_cast<double>(i) / static_cast<double>(num_samples);
    double diff1               = cuda::std::abs(empirical_cdf_upper - static_cast<double>(cdf_values[i]));
    double diff2               = cuda::std::abs(static_cast<double>(cdf_values[i]) - empirical_cdf_lower);
    double d_i                 = cuda::std::max(diff1, diff2);
    if (d_i > d_max)
    {
      d_max = d_i;
    }
  }

  const double critical_value = 0.016259280113043572; // for alpha = 0.01 and n = 10000
  return d_max < critical_value;
}

#endif // LIBCUDACXX_TEST_SUPPORT_RANDOM_UTILITIES_TEST_CONTINUOUS_DISTRIBUTION_H
