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
// Is this needed?
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

template <class D, class URNG, class Param>
__host__ __device__ constexpr bool test_ctor(Param param)
{
  D d1(param);
  D d2;
  assert(d1.param() == param);
  static_assert(noexcept(D()));
  static_assert(noexcept(D(param)));
  static_assert(noexcept(d2 = d1));
  return true;
}
template <class D, class URNG, class Param>
__host__ __device__ constexpr bool test_assign(Param param)
{
  D d1(param);
  D d2;
  d2 = d1;
  assert(d1 == d2);
  static_assert(noexcept(d2 = d1));
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
  static_assert(noexcept(d1.param(param)));
  return true;
}

template <class D, class URNG, class Param>
__host__ __device__ constexpr bool test_types(Param param)
{
  D d1(param);
  URNG g{};
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

  Param p1;
  Param p2(param);
  assert(p2 == param);
  assert(!(p2 != param));
  Param p3 = param;
  assert(p3 == param);
  static_assert(noexcept(Param()));
  static_assert(noexcept(Param(param)));
  static_assert(noexcept(p3 = p2));
  static_assert(noexcept(p2 == p3));
  static_assert(noexcept(p2 != p3));
  return true;
}

// Perform a kolmogorov-Smirnov test, comparing the observed and expected cumulative
// distribution function from a continuous distribution.
// Generates a fixed size of 10000 samples
template <class D, bool continuous, class URNG, bool test_constexpr, class CDF>
__host__ __device__ bool test_eval(const typename D::param_type param, CDF cdf)
{
  static_assert(continuous, "This test is only for continuous distributions");
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

  auto samples    = cuda::std::make_unique<typename D::result_type[]>(num_samples);
  auto cdf_values = cuda::std::make_unique<typename D::result_type[]>(num_samples);
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
    cdf_values[i] = cdf(samples[i], param);
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
  assert(d_max < critical_value);
  return true;
}
} // namespace detail

template <class D, bool continuous, class URNG, bool test_constexpr, class CDF, cuda::std::size_t N>
__host__ __device__ void test_distribution(const cuda::std::array<typename D::param_type, N>& params, CDF cdf)
{
  if constexpr (test_constexpr)
  {
    static_assert(detail::test_eval_constexpr<D, URNG>());
  }

  for (auto param : params)
  {
    detail::test_eval<D, continuous, URNG, test_constexpr>(param, cdf);
    detail::test_assign<D, URNG>(param);
    detail::test_ctor<D, URNG>(param);
    detail::test_copy<D, URNG>(param);
    detail::test_eq<D, URNG>(param);
    detail::test_get_param<D, URNG>(param);
    detail::test_min_max<D, URNG>(param);
    detail::test_set_param<D, URNG>(param);
    detail::test_types<D, URNG>(param);
    detail::test_param<D, URNG, typename D::param_type>(param);
    NV_IF_TARGET(NV_IS_HOST, ({ detail::test_io<D, URNG>(param); }));
  }
}

#endif // LIBCUDACXX_TEST_SUPPORT_RANDOM_UTILITIES_TEST_DISTRIBUTION_H
