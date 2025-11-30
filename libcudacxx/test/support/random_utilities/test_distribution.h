//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef LIBCUDACXX_TEST_SUPPORT_RANDOM_UTILITIES_TEST_DISTRIBUTION_H
#define LIBCUDACXX_TEST_SUPPORT_RANDOM_UTILITIES_TEST_DISTRIBUTION_H

#include <cuda/std/__algorithm/partial_sort.h>
#include <cuda/std/__memory_>
#include <cuda/std/array>
#include <cuda/std/cstddef>

#if !_CCCL_COMPILER(NVRTC)
#  include <sstream>
#endif // !_CCCL_COMPILER(NVRTC)

#include "test_macros.h"

namespace detail
{
template <class D, class URNG, class Param>
__host__ __device__ constexpr bool test_ctor_assign(Param param)
{
  D d1(param);
  D d2;
  d2 = d1;
  assert(d1 == d2);
  assert(d1.param() == param);
  return true;
}

template <class D, class URNG, class Param>
__host__ __device__ constexpr bool test_copy(Param param)
{
  D d1(param);
  D d2(d1);
  assert(d1 == d2);
  static_assert(noexcept(D(d1)));
  return true;
}

template <class D, class URNG, class Param>
__host__ __device__ constexpr bool test_eq(Param param)
{
  D d1(param);
  D d2(param);
  assert(d1 == d2);
  assert(!(d1 != d2));
  static_assert(noexcept(d1 == d2));
  static_assert(noexcept(d1 != d2));
  return true;
}

template <class D, class URNG, class Param>
__host__ __device__ constexpr bool test_get_param(Param param)
{
  D d1(param);
  assert(d1.param() == param);
  static_assert(noexcept(d1.param()));
  return true;
}

#if !_CCCL_COMPILER(NVRTC)
template <class D, class URNG, class Param>
bool test_io(Param param)
{
  D d1(param);
  std::stringstream ss;
  ss << d1;
  D d2;
  ss >> d2;
  assert(d1 == d2);
  return true;
}
#endif

template <class D, class URNG, class Param>
__host__ __device__ constexpr bool test_min_max(Param param)
{
  D d1(param);
  static_assert(noexcept(d1.min()));
  static_assert(noexcept(d1.max()));
  assert(d1.min() <= d1.max());
  return true;
}

template <class D, class URNG, class Param>
__host__ __device__ constexpr bool test_set_param(Param param)
{
  D d1;
  d1.param(param);
  assert(d1.param() == param);
  return true;
}

template <class D, class URNG, class Param>
__host__ __device__ constexpr bool test_types(Param param)
{
  D d1(param);
  [[maybe_unused]] URNG g{};
  using result_type = typename D::result_type;
  static_assert(cuda::std::is_same_v<result_type, decltype(d1.min())>);
  static_assert(cuda::std::is_same_v<result_type, decltype(d1.max())>);
  static_assert(cuda::std::is_same_v<result_type, decltype(d1(g))>);
  static_assert(cuda::std::is_same_v<result_type, decltype(d1(g, param))>);
  return true;
}

template <class D, class URNG, class Param>
__host__ __device__ constexpr bool test_param(Param param)
{
  static_assert(cuda::std::is_same_v<typename D::param_type, Param>);
  static_assert(cuda::std::is_same_v<typename Param::distribution_type, D>);

  Param p2(param);
  assert(p2 == param);
  assert(!(p2 != param));
  Param p3 = param;
  assert(p3 == param);
  static_assert(noexcept(p3 = p2));
  static_assert(noexcept(p2 == p3));
  static_assert(noexcept(p2 != p3));
  return true;
}

// Compute KS test statistic for continuous distributions
template <class D, class CDF>
__host__ __device__ double ks_test_statistic_continuous(
  const typename D::result_type* samples, cuda::std::size_t num_samples, const typename D::param_type& param, CDF cdf)
{
  double d_max = 0.0;
  for (cuda::std::size_t i = 0; i < num_samples; ++i)
  {
    double f_x       = static_cast<double>(i + 1) / static_cast<double>(num_samples);
    double f_x_lower = static_cast<double>(i) / static_cast<double>(num_samples);
    double g_x       = cdf(samples[i], param);
    double diff1     = cuda::std::abs(f_x - g_x);
    double diff2     = cuda::std::abs(g_x - f_x_lower);
    d_max            = cuda::std::max(d_max, cuda::std::max(diff1, diff2));
  }
  return d_max;
}

// Compute KS test statistic for discrete distributions
template <class D, class CDF>
__host__ __device__ double ks_test_statistic_discrete(
  const typename D::result_type* samples, cuda::std::size_t num_samples, const typename D::param_type& param, CDF cdf)
{
  // Compute empirical CDF
  // Find unique values and their frequencies
  auto unique_values             = cuda::std::make_unique<typename D::result_type[]>(num_samples);
  auto empirical_cdf             = cuda::std::make_unique<double[]>(num_samples);
  cuda::std::size_t unique_count = 0;
  for (cuda::std::size_t i = 0; i < num_samples; ++i)
  {
    if (samples[i] != samples[i + 1] || i == num_samples - 1)
    {
      unique_values[unique_count] = samples[i];
      empirical_cdf[unique_count] = (i + 1) / static_cast<double>(num_samples);
      unique_count++;
    }
  }
  // Compute KS statistic
  double d_max = 0.0;
  for (cuda::std::size_t j = 0; j < unique_count; ++j)
  {
    double f_x       = empirical_cdf[j];
    double f_x_lower = j == 0 ? 0.0 : empirical_cdf[j - 1];
    double g_x       = cdf(unique_values[j], param);
    double g_x_lower = j == 0 ? 0.0 : cdf(unique_values[j] - 1, param);
    double diff1     = cuda::std::abs(f_x - g_x);
    double diff2     = cuda::std::abs(g_x_lower - f_x_lower);
    d_max            = cuda::std::max(d_max, cuda::std::max(diff1, diff2));
  }

  return d_max;
}

// Perform a kolmogorov-Smirnov test, comparing the observed and expected cumulative
// distribution function from a continuous distribution.
// Generates a fixed size of 10000 samples
template <class D, bool continuous, class URNG, bool test_constexpr, class CDF>
__host__ __device__ bool test_eval(const typename D::param_type param, CDF cdf)
{
  // First check the operator with param is equivalent to the constructor param
  {
    D d1(param);
    D d2(param);
    URNG g_1{};
    URNG g_2{};
    for (cuda::std::size_t i = 0; i < 100; ++i)
    {
      auto dist_val  = d1(g_1, param);
      auto dist2_val = d2(g_2);
      assert(dist_val == dist2_val);
    }
  }

  D dist(param);
  URNG g{};
  const cuda::std::size_t num_samples = 10000;

  auto samples = cuda::std::make_unique<typename D::result_type[]>(num_samples);
  for (cuda::std::size_t i = 0; i < num_samples; ++i)
  {
    samples[i] = dist(g, param);
  }
  // Use sort when available
  cuda::std::partial_sort(samples.get(), samples.get() + num_samples, samples.get() + num_samples);

  // Compute the KS statistic - specially handle discrete case
  // Arnold, Taylor B., and John W. Emerson. "Nonparametric goodness-of-fit tests for discrete null distributions."
  // (2011).
  double d_max = 0.0;
  if constexpr (continuous)
  {
    d_max = ks_test_statistic_continuous<D>(samples.get(), num_samples, param, cdf);
  }
  else
  {
    d_max = ks_test_statistic_discrete<D>(samples.get(), num_samples, param, cdf);
  }

  // Note that this critical value from the KS distribution is only valid for discrete distributions when num_samples is
  // large
  const double critical_value = 0.016259280113043572; // for alpha = 0.01 and n = 10000
  assert(d_max < critical_value);
  return true;
}
template <class D, class URNG>
__host__ __device__ constexpr bool test_eval_constexpr()
{
  typename D::param_type param;
  D dist(param);
  URNG g{};
  unused(dist(g, param));
  unused(dist(g));
  return true;
}
} // namespace detail

template <class D, bool continuous, class URNG, bool test_constexpr, class CDF, cuda::std::size_t N>
__host__ __device__ void constexpr test_distribution(cuda::std::array<typename D::param_type, N> params, CDF cdf)
{
  for (cuda::std::size_t i = 0; i < N; ++i)
  {
    detail::test_eval<D, continuous, URNG, test_constexpr>(params[i], cdf);
    detail::test_ctor_assign<D, URNG>(params[i]);
    detail::test_copy<D, URNG>(params[i]);
    detail::test_eq<D, URNG>(params[i]);
    detail::test_get_param<D, URNG>(params[i]);
    detail::test_min_max<D, URNG>(params[i]);
    detail::test_set_param<D, URNG>(params[i]);
    detail::test_types<D, URNG>(params[i]);
    detail::test_param<D, URNG, typename D::param_type>(params[i]);
    NV_IF_TARGET(NV_IS_HOST, ({ detail::test_io<D, URNG>(params[i]); }));
  }
  if constexpr (test_constexpr)
  {
    constexpr typename D::param_type param{};
    static_assert(detail::test_eval_constexpr<D, URNG>());
    static_assert(detail::test_ctor_assign<D, URNG>(param));
    static_assert(detail::test_eq<D, URNG>(param));
    static_assert(detail::test_get_param<D, URNG>(param));
    static_assert(detail::test_min_max<D, URNG>(param));
    static_assert(detail::test_set_param<D, URNG>(param));
    static_assert(detail::test_types<D, URNG>(param));
    static_assert(detail::test_param<D, URNG, typename D::param_type>(param));
  }
}

#endif // LIBCUDACXX_TEST_SUPPORT_RANDOM_UTILITIES_TEST_DISTRIBUTION_H
