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

// template<class _IntType = int>
// class uniform_int_distribution

// template<class _URng> result_type operator()(_URng& g);

#include <cuda/std/__random_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/climits>
#include <cuda/std/cmath>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/numeric>

#include "test_macros.h"

template <class T>
__host__ __device__ T sqr(T x)
{
  return x * x;
}

template <class ResultType, class EngineType>
__host__ __device__ void test_statistics(ResultType a, ResultType b)
{
  static_assert(
    cuda::std::is_same_v<typename cuda::std::uniform_int_distribution<ResultType>::result_type, ResultType>);

  EngineType g;
  cuda::std::uniform_int_distribution<ResultType> dist(a, b);
  assert(dist.a() == a);
  assert(dist.b() == b);
  constexpr int N = 10000;
  cuda::std::array<ResultType, N> u;
  for (int i = 0; i < 10000; ++i)
  {
    ResultType v = dist(g);
    assert(a <= v && v <= b);
    u[i] = v;
  }

  // Quick check: The chance of getting *no* hits in any given tenth of the range
  // is (0.9)^10000, or "ultra-astronomically low."
  bool bottom_tenth = false;
  bool top_tenth    = false;
  for (cuda::std::size_t i = 0; i < u.size(); ++i)
  {
    bottom_tenth = bottom_tenth || (u[i] <= (a + (b / 10) - (a / 10)));
    top_tenth    = top_tenth || (u[i] >= (b - (b / 10) + (a / 10)));
  }
  assert(bottom_tenth); // ...is populated
  assert(top_tenth); // ...is populated

  // Now do some more involved statistical math.
  double mean     = cuda::std::accumulate(u.begin(), u.end(), 0.0) / u.size();
  double var      = 0;
  double skew     = 0;
  double kurtosis = 0;
  for (cuda::std::size_t i = 0; i < u.size(); ++i)
  {
    double dbl = (u[i] - mean);
    double d2  = dbl * dbl;
    var += d2;
    skew += dbl * d2;
    kurtosis += d2 * d2;
  }
  var /= u.size();
  double dev = cuda::std::sqrt(var);
  skew /= u.size() * dev * var;
  kurtosis /= u.size() * var * var;

  double expected_mean = double(a) + double(b) / 2 - double(a) / 2;
  double expected_var  = (sqr(double(b) - double(a) + 1) - 1) / 12;

  double range = double(b) - double(a) + 1.0;
  assert(range > range / 10); // i.e., it's not infinity

  assert(cuda::std::abs(mean - expected_mean) < range / 100);
  assert(cuda::std::abs(var - expected_var) < expected_var / 50);
  assert(-0.1 < skew && skew < 0.1);
  assert(1.6 < kurtosis && kurtosis < 2.0);
}

template <class ResultType, class EngineType>
__host__ __device__ void test_statistics()
{
  test_statistics<ResultType, EngineType>(0, cuda::std::numeric_limits<ResultType>::max());
}

int main(int, char**)
{
  test_statistics<int, cuda::std::minstd_rand0>();
  test_statistics<int, cuda::std::minstd_rand0>(-6, 106);
#if 0 // not implemented
  test_statistics<int, cuda::std::minstd_rand>();
  test_statistics<int, cuda::std::mt19937>();
  test_statistics<int, cuda::std::mt19937_64>();
  test_statistics<int, cuda::std::ranlux24_base>();
  test_statistics<int, cuda::std::ranlux48_base>();
  test_statistics<int, cuda::std::ranlux24>();
  test_statistics<int, cuda::std::ranlux48>();
  test_statistics<int, cuda::std::knuth_b>();
  test_statistics<int, cuda::std::minstd_rand>(5, 100);
#endif // not implemented

  test_statistics<short, cuda::std::minstd_rand0>();
  test_statistics<int, cuda::std::minstd_rand0>();
  test_statistics<long, cuda::std::minstd_rand0>();
  test_statistics<long long, cuda::std::minstd_rand0>();

  test_statistics<unsigned short, cuda::std::minstd_rand0>();
  test_statistics<unsigned int, cuda::std::minstd_rand0>();
  test_statistics<unsigned long, cuda::std::minstd_rand0>();
  test_statistics<unsigned long long, cuda::std::minstd_rand0>();

  test_statistics<short, cuda::std::minstd_rand0>(SHRT_MIN, SHRT_MAX);

  test_statistics<cuda::std::int8_t, cuda::std::minstd_rand0>();
  test_statistics<cuda::std::uint8_t, cuda::std::minstd_rand0>();

#if _CCCL_HAS_INT128()
  test_statistics<__int128_t, cuda::std::minstd_rand0>();
  test_statistics<__uint128_t, cuda::std::minstd_rand0>();

  test_statistics<__int128_t, cuda::std::minstd_rand0>(-100, 900);
  test_statistics<__int128_t, cuda::std::minstd_rand0>(0, UINT64_MAX);
  test_statistics<__int128_t, cuda::std::minstd_rand0>(
    cuda::std::numeric_limits<__int128_t>::min(), cuda::std::numeric_limits<__int128_t>::max());
  test_statistics<__uint128_t, cuda::std::minstd_rand0>(0, UINT64_MAX);
#endif // _CCCL_HAS_INT128()

  return 0;
}
