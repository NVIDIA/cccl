//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Test uniformity of shuffle_iterator permutation distribution
// This test checks that different seeds produce a reasonably uniform
// distribution across all possible permutations.

#include <cuda/iterator>
#include <cuda/std/__random_>
#include <cuda/std/array>
#include <cuda/std/cassert>

#include <nv/target>

#include "test_macros.h"
#include "types.h"

TEST_DIAG_SUPPRESS_MSVC(4146) // unary minus operator applied to unsigned type, result still unsigned

// A lehmer code is a unique index for a permutation
template <size_t N>
__host__ __device__ size_t lehmer_code(const cuda::std::array<int, N>& perm)
{
  // This algorithm is N^2 but is faster for small N
  size_t rank = 0;
  size_t fact = 1; // 0!

  for (int i = (int) N - 1; i >= 0; --i)
  {
    int count = 0;
    for (size_t j = i + 1; j < N; ++j)
    {
      if (perm[j] < perm[i])
      {
        ++count;
      }
    }

    rank += count * fact;
    fact *= (N - i);
  }

  return rank;
}

__host__ __device__ constexpr size_t factorial(size_t n)
{
  return n <= 1 ? 1 : n * factorial(n - 1);
}

// Compute chi-squared statistic
template <size_t NumCategories>
__host__ __device__ double compute_chi_squared(const cuda::std::array<size_t, NumCategories>& counts, double expected)
{
  double chi2 = 0.0;
  for (size_t c : counts)
  {
    double diff = static_cast<double>(c) - expected;
    chi2 += (diff * diff) / expected;
  }
  return chi2;
}

// Exhaustively generate permutations for a small N and count occurrences
template <size_t N>
__host__ __device__ void test_small_n()
{
  static_assert(N <= 5, "N too large for exhaustive permutation test");
  constexpr size_t num_permutations = factorial(N);
  cuda::std::array<size_t, num_permutations> counts{};
  const int num_samples = 10000;

  cuda::std::philox4x64 rng;
  for (size_t i = 0; i < num_samples; ++i)
  {
    cuda::random_bijection<uint64_t> bijection{static_cast<uint64_t>(N), rng};
    cuda::shuffle_iterator iter{bijection};

    // Read the permutation
    cuda::std::array<int, N> perm{};
    for (size_t j = 0; j < N; ++j)
    {
      perm[j] = static_cast<int>(iter[j]);
    }
    // Convert to index and count
    counts[lehmer_code(perm)]++;
  }

  // Check that all permutations were seen
  size_t unique = 0;
  for (size_t c : counts)
  {
    unique += c > 0;
  }
  assert(unique == num_permutations);

  // Chi-squared test
  double expected = static_cast<double>(num_samples) / num_permutations;
  double chi2     = compute_chi_squared(counts, expected);
  // Degrees of freedom = num_permutations - 1, alpha=0.05
  // DoF: -, -, 1, 5, 23, 119 for N=0,1,2,3,4,5
  const double critical_values_N[6] = {0, 0, 3.841, 11.070, 35.172, 145.461};
  assert(chi2 < critical_values_N[N]);
}

__host__ __device__ void chi_squared_tests()
{
  test_small_n<2>();
  test_small_n<3>();
  test_small_n<4>();
  test_small_n<5>();
}

template <size_t N>
struct Fenwick
{
  cuda::std::array<int, N + 1> f{};

  Fenwick() = default;

  __host__ __device__ void add(size_t i, int v = 1)
  {
    for (++i; i < f.size(); i += i & -i)
    {
      f[i] += v;
    }
  }

  __host__ __device__ int sum(size_t i) const
  {
    int s = 0;
    for (++i; i > 0; i -= i & -i)
    {
      s += f[i];
    }
    return s;
  }
};

// O(n log n) Kendall distance
template <size_t N>
__host__ __device__ size_t kendall_distance(const cuda::std::array<int, N>& a, const cuda::std::array<int, N>& b)
{
  cuda::std::array<int, N> inv{};
  cuda::std::array<int, N> c{};

  for (size_t i = 0; i < N; ++i)
  {
    inv[a[i]] = static_cast<int>(i);
  }

  for (size_t i = 0; i < N; ++i)
  {
    c[i] = inv[b[i]];
  }

  Fenwick<N> fw;
  size_t inv_count = 0;

  for (size_t i = 0; i < N; ++i)
  {
    inv_count += i - static_cast<size_t>(fw.sum(static_cast<size_t>(c[i])));
    fw.add(static_cast<size_t>(c[i]), 1);
  }

  return inv_count;
}

// Mallows kernel
template <size_t N>
__host__ __device__ double
mallows_kernel(const cuda::std::array<int, N>& a, const cuda::std::array<int, N>& b, double lambda)
{
  const double n = static_cast<double>(N);
  double d       = static_cast<double>(kendall_distance<N>(a, b));
  return cuda::std::exp(-lambda * d / (n * n));
}

// E[K] under uniform distribution (closed form)
__host__ __device__ double expected_K(size_t n, double lambda)
{
  double prod = 1.0;
  double n2   = double(n) * double(n);

  for (size_t j = 1; j <= n; ++j)
  {
    double num = 1.0 - cuda::std::exp(-lambda * j / n2);
    double den = j * (1.0 - cuda::std::exp(-lambda / n2));
    prod *= num / den;
  }
  return prod;
}

// E[K^2] under uniform distribution
__host__ __device__ double expected_K2(size_t n, double lambda)
{
  double prod = 1.0;
  double n2   = double(n) * double(n);

  for (size_t j = 1; j <= n; ++j)
  {
    double num = 1.0 - cuda::std::exp(-2.0 * lambda * j / n2);
    double den = j * (1.0 - cuda::std::exp(-2.0 * lambda / n2));
    prod *= num / den;
  }
  return prod;
}

__host__ __device__ double inverse_erf(double x)
{
  double tt1, tt2, lnx, sgn;
  sgn = (x < 0) ? -1.0 : 1.0;

  x   = (1 - x) * (1 + x);
  lnx = cuda::std::log(x);

  tt1 = 2 / (3.14159265358979323846 * 0.147) + 0.5f * lnx;
  tt2 = 1 / (0.147) * lnx;

  return (sgn * cuda::std::sqrt(-tt1 + cuda::std::sqrt(tt1 * tt1 - tt2)));
}

// Formula (7): acceptance threshold
__host__ __device__ double mmd_threshold(size_t n, size_t M, double lambda, double alpha)
{
  double EK  = expected_K(n, lambda);
  double EK2 = expected_K2(n, lambda);

  double varK   = EK2 - EK * EK;
  double varMMD = 2.0 * varK / double(M);

  return cuda::std::sqrt(2.0 * varMMD) * inverse_erf(1.0 - alpha);
}

template <size_t N>
__host__ __device__ void test_mmd()
{
  const double lambda   = 5.0;
  const int num_samples = 1000;
  cuda::std::philox4x64 rng;
  double sum = 0.0;
  for (size_t i = 0; i < num_samples; i += 2)
  {
    cuda::random_bijection<uint64_t> bijection_a{static_cast<uint64_t>(N), rng};
    cuda::shuffle_iterator iter{bijection_a};
    cuda::random_bijection<uint64_t> bijection_b{static_cast<uint64_t>(N), rng};
    cuda::shuffle_iterator iter_b{bijection_b};

    // Get a pair of permutations
    cuda::std::array<int, N> a{};
    cuda::std::array<int, N> b{};
    for (size_t j = 0; j < N; ++j)
    {
      a[j] = static_cast<int>(iter[j]);
      b[j] = static_cast<int>(iter_b[j]);
    }
    sum += mallows_kernel<N>(a, b, lambda);
  }

  double EK        = expected_K(N, lambda);
  double mmd_hat   = cuda::std::abs((2.0 / num_samples) * sum - EK);
  double threshold = mmd_threshold(N, num_samples, lambda, 0.05);
  assert(mmd_hat < threshold);
}

// Mitchell, Rory, et al. "Bandwidth-optimal random shuffling for GPUs." ACM Transactions on Parallel Computing 9.1
// (2022): 1-20.
__host__ __device__ void maximum_mean_discrepency_tests()
{
  test_mmd<50>();
  test_mmd<100>();
  test_mmd<2500>();
}

template <size_t N>
__host__ __device__ void expected_value_test()
{
  const int num_samples = 1000;
  cuda::std::philox4x64 rng;
  cuda::std::array<double, N> expected_value{};
  for (size_t i = 0; i < num_samples; ++i)
  {
    cuda::random_bijection<uint64_t> bijection{static_cast<uint64_t>(N), rng};
    cuda::shuffle_iterator iter{bijection};
    for (size_t j = 0; j < N; ++j)
    {
      expected_value[j] += static_cast<double>(iter[j]);
    }
  }

  double mu    = (N - 1) / 2.0;
  double sigma = cuda::std::sqrt((N * N - 1) / 12.0);
  double zmax  = 0.0;
  for (size_t i = 0; i < N; ++i)
  {
    double mean = expected_value[i] / double(num_samples);
    double z    = cuda::std::abs(mean - mu) / (sigma / cuda::std::sqrt(double(num_samples)));
    zmax        = cuda::std::max(zmax, cuda::std::abs(z));
  }

  double alpha = 0.05;
  double zcrit = inverse_erf(1.0 - alpha / (2.0 * N)) * cuda::std::sqrt(2.0);
  assert(zmax < zcrit);
}

// Test the expected value at each index of the shuffle_iterator
__host__ __device__ void expected_value_tests()
{
  expected_value_test<50>();
  expected_value_test<100>();
  expected_value_test<2500>();
}

template <size_t N>
__host__ __device__ void adjacent_inversion_test()
{
  const double alpha    = 0.05;
  const int num_samples = 1000;
  cuda::std::philox4x64 rng;
  double z_max = 0.0;
  for (size_t i = 0; i < num_samples; ++i)
  {
    cuda::random_bijection<uint64_t> bijection{static_cast<uint64_t>(N), rng};
    cuda::shuffle_iterator iter{bijection};
    double inversions = 0.0;
    for (size_t j = 0; j < N - 1; ++j)
    {
      if (iter[j] > iter[j + 1])
      {
        inversions += 1.0;
      }
    }
    double expected_inversions = (N - 1) / 2.0;
    double variance            = (N - 1) / 4.0;
    double z                   = (inversions - expected_inversions) / cuda::std::sqrt(variance);
    z_max                      = cuda::std::max(z_max, cuda::std::abs(z));
  }
  double zcrit = inverse_erf(1.0 - alpha / (2.0 * num_samples)) * cuda::std::sqrt(2.0);
  assert(z_max < zcrit);
}

__host__ __device__ void adjacent_inversion_tests()
{
  adjacent_inversion_test<5>();
  adjacent_inversion_test<694>();
  adjacent_inversion_test<2322>();
}

int main(int, char**)
{
  chi_squared_tests();
  maximum_mean_discrepency_tests();
  expected_value_tests();
  adjacent_inversion_tests();
  return 0;
}
