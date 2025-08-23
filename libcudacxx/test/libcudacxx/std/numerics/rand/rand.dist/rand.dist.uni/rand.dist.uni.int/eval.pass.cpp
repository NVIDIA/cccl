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

#include <cuda/std/__memory_>
#include <cuda/std/__random_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/climits>
#include <cuda/std/cmath>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/numeric>
#include <cuda/std/span>

#include "test_macros.h"

template <class T>
__host__ __device__ T sqr(T x)
{
  return x * x;
}

constexpr int N = 10000;

template <class ResultType, class EngineType>
__host__ __device__ void test_statistics(cuda::std::span<ResultType, N> arr, ResultType a, ResultType b)
{
  static_assert(
    cuda::std::is_same_v<typename cuda::std::uniform_int_distribution<ResultType>::result_type, ResultType>);

  EngineType g;
  cuda::std::uniform_int_distribution<ResultType> dist(a, b);
  assert(dist.a() == a);
  assert(dist.b() == b);
  for (int i = 0; i < N; ++i)
  {
    ResultType v = dist(g);
    assert(a <= v && v <= b);
    arr[i] = v;
  }

  // Quick check: The chance of getting *no* hits in any given tenth of the range
  // is (0.9)^10000, or "ultra-astronomically low."
  bool bottom_tenth = false;
  bool top_tenth    = false;
  for (cuda::std::size_t i = 0; i < arr.size(); ++i)
  {
    bottom_tenth = bottom_tenth || (arr[i] <= (a + (b / 10) - (a / 10)));
    top_tenth    = top_tenth || (arr[i] >= (b - (b / 10) + (a / 10)));
  }
  assert(bottom_tenth); // ...is populated
  assert(top_tenth); // ...is populated

  // Now do some more involved statistical math.
  double mean     = cuda::std::accumulate(arr.begin(), arr.end(), 0.0) / arr.size();
  double var      = 0;
  double skew     = 0;
  double kurtosis = 0;
  for (cuda::std::size_t i = 0; i < arr.size(); ++i)
  {
    double dbl = (arr[i] - mean);
    double d2  = dbl * dbl;
    var += d2;
    skew += dbl * d2;
    kurtosis += d2 * d2;
  }
  var /= arr.size();
  double dev = cuda::std::sqrt(var);
  skew /= arr.size() * dev * var;
  kurtosis /= arr.size() * var * var;

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
__host__ __device__ void test_statistics(cuda::std::span<ResultType, N> arr)
{
  test_statistics<ResultType, EngineType>(arr, 0, cuda::std::numeric_limits<ResultType>::max());
}

template <typename ResultType>
__host__ __device__ auto convert_to(cuda::std::span<unsigned char, N * 16> arr)
{
  ResultType* data = reinterpret_cast<ResultType*>(arr.data());
  return cuda::std::span<ResultType, N>{data, data + N};
}

template <class EngineType>
__host__ __device__ void test_statistics(cuda::std::span<unsigned char, N * 16> arr)
{
  test_statistics<int, EngineType>(convert_to<int>(arr));
  test_statistics<short, EngineType>(convert_to<short>(arr));
  test_statistics<long, EngineType>(convert_to<long>(arr));
  test_statistics<long long, EngineType>(convert_to<long long>(arr));

  test_statistics<unsigned short, EngineType>(convert_to<unsigned short>(arr));
  test_statistics<unsigned int, EngineType>(convert_to<unsigned int>(arr));
  test_statistics<unsigned long, EngineType>(convert_to<unsigned long>(arr));
  test_statistics<unsigned long long, EngineType>(convert_to<unsigned long long>(arr));

  test_statistics<cuda::std::int8_t, EngineType>(convert_to<cuda::std::int8_t>(arr));
  test_statistics<cuda::std::uint8_t, EngineType>(convert_to<cuda::std::uint8_t>(arr));

  // clang-cuda segfaults with 128 bit
  // fatal error: error in backend: Undefined external symbol ""
#if _CCCL_HAS_INT128() && !TEST_CUDA_COMPILER(CLANG)
  test_statistics<__int128_t, EngineType>(convert_to<__int128_t>(arr));
  test_statistics<__uint128_t, EngineType>(convert_to<__uint128_t>(arr));
#endif // _CCCL_HAS_INT128() && !TEST_CUDA_COMPILER(CLANG)
}

__host__ __device__ void test()
{
  cuda::std::unique_ptr<unsigned char[]> array = cuda::std::make_unique<unsigned char[]>(N * 16);
  cuda::std::span<unsigned char, N * 16> span{array.get(), array.get() + N * 16};

  test_statistics<cuda::std::minstd_rand0>(span);

#if 0 // not implemented
  test_statistics<int, cuda::std::minstd_rand>(span);
  test_statistics<int, cuda::std::mt19937>(span);
  test_statistics<int, cuda::std::mt19937_64>(span);
  test_statistics<int, cuda::std::ranlux24_base>(span);
  test_statistics<int, cuda::std::ranlux48_base>(span);
  test_statistics<int, cuda::std::ranlux24>(span);
  test_statistics<int, cuda::std::ranlux48>(span);
  test_statistics<int, cuda::std::knuth_b>(span);
  test_statistics<int, cuda::std::minstd_rand>(span, 5, 100);
#endif // not implemented
  test_statistics<int, cuda::std::minstd_rand0>(convert_to<int>(span), -6, 106);
  test_statistics<short, cuda::std::minstd_rand0>(convert_to<short>(span), SHRT_MIN, SHRT_MAX);

#if _CCCL_HAS_INT128() && !TEST_CUDA_COMPILER(CLANG)
  test_statistics<__int128_t, cuda::std::minstd_rand0>(convert_to<__int128_t>(span), -100, 900);
  test_statistics<__int128_t, cuda::std::minstd_rand0>(convert_to<__int128_t>(span), 0, UINT64_MAX);
  test_statistics<__int128_t, cuda::std::minstd_rand0>(
    convert_to<__int128_t>(span),
    cuda::std::numeric_limits<__int128_t>::min(),
    cuda::std::numeric_limits<__int128_t>::max());
  test_statistics<__uint128_t, cuda::std::minstd_rand0>(convert_to<__uint128_t>(span), 0, UINT64_MAX);
#endif // _CCCL_HAS_INT128() && !TEST_CUDA_COMPILER(CLANG)
}

int main(int, char**)
{
  test();
  return 0;
}
