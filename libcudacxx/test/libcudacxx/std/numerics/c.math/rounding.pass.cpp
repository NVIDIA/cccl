//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cmath>

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "comparison.h"
#include "fp_compare.h"
#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4244) // conversion from 'double' to 'float', possible loss of data
TEST_DIAG_SUPPRESS_MSVC(4305) // 'argument': truncation from 'T' to 'float'
TEST_DIAG_SUPPRESS_MSVC(4146) // unary minus operator applied to unsigned type, result still unsigned

template <typename T>
__host__ __device__ void test_ceil(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ceil(T{})), ret>, "");

  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::ceil(val), val));
    assert(eq(cuda::std::ceil(-val), -val));
  }
  else
  {
    assert(eq(cuda::std::ceil(val), T(2)));
    assert(eq(cuda::std::ceil(-val), T(-1)));
  }
  assert(eq(cuda::std::ceil(T(-0.0)), T(-0.0)));
  assert(eq(cuda::std::ceil(T(0.0)), T(0.0)));
  assert(eq(cuda::std::ceil(T(-cuda::std::numeric_limits<T>::infinity())), -cuda::std::numeric_limits<T>::infinity()));
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::ceilf(val), T(2)));
    assert(eq(cuda::std::ceilf(-val), T(-1)));
    assert(eq(cuda::std::ceilf(T(-0.0)), T(-0.0)));
    assert(eq(cuda::std::ceilf(T(0.0)), T(0.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::ceill(val), T(2)));
    assert(eq(cuda::std::ceill(-val), T(-1)));
    assert(eq(cuda::std::ceill(T(-0.0)), T(-0.0)));
    assert(eq(cuda::std::ceill(T(0.0)), T(0.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_floor(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::floor(T{})), ret>, "");

  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::floor(val), val));
    assert(eq(cuda::std::floor(-val), -val));
  }
  else
  {
    assert(eq(cuda::std::floor(val), T(1)));
    assert(eq(cuda::std::floor(-val), T(-2)));
  }
  assert(eq(cuda::std::floor(T(-0.0)), T(-0.0)));
  assert(eq(cuda::std::floor(T(0.0)), T(0.0)));
  assert(eq(cuda::std::floor(T(-cuda::std::numeric_limits<T>::infinity())), -cuda::std::numeric_limits<T>::infinity()));
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::floorf(val), T(1)));
    assert(eq(cuda::std::floorf(-val), T(-2)));
    assert(eq(cuda::std::floorf(T(-0.0)), T(-0.0)));
    assert(eq(cuda::std::floorf(T(0.0)), T(0.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::floorl(val), T(1)));
    assert(eq(cuda::std::floorl(-val), T(-2)));
    assert(eq(cuda::std::floorl(T(-0.0)), T(-0.0)));
    assert(eq(cuda::std::floorl(T(0.0)), T(0.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_llrint(T val)
{
  static_assert(cuda::std::is_same_v<decltype(cuda::std::llrint(T{})), long long>, "");

  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::llrint(val), val));
  }
  else
  {
    assert(cuda::std::llrint(val) == 2);
    assert(cuda::std::llrint(-val) == -2);
    assert(cuda::std::llrint(val - T(0.2)) == 1);
    assert(cuda::std::llrint(-val + T(0.2)) == -1);
  }
  assert(cuda::std::llrint(T(-0.0)) == -0);
  assert(cuda::std::llrint(T(0.0)) == 0);
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(cuda::std::llrintf(val) == 2);
    assert(cuda::std::llrintf(-val) == -2);
    assert(cuda::std::llrintf(val - T(0.2)) == 1);
    assert(cuda::std::llrintf(-val + T(0.2)) == -1);
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(cuda::std::llrintl(val) == 2);
    assert(cuda::std::llrintl(-val) == -2);
    assert(cuda::std::llrintl(val - T(0.2)) == 1);
    assert(cuda::std::llrintl(-val + T(0.2)) == -1);
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_llround(T val)
{
  static_assert(cuda::std::is_same_v<decltype(cuda::std::llround(T{})), long long>, "");

  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::llround(val), val));
  }
  else
  {
    assert(cuda::std::llround(val) == 2);
    assert(cuda::std::llround(-val) == -2);
    assert(cuda::std::llround(val - T(0.2)) == 1);
    assert(cuda::std::llround(-val + T(0.2)) == -1);
  }
  assert(cuda::std::llround(T(-0.0)) == -0);
  assert(cuda::std::llround(T(0.0)) == 0);
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(cuda::std::llroundf(val) == 2);
    assert(cuda::std::llroundf(-val) == -2);
    assert(cuda::std::llroundf(val - T(0.2)) == 1);
    assert(cuda::std::llroundf(-val + T(0.2)) == -1);
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(cuda::std::llroundl(val) == 2);
    assert(cuda::std::llroundl(-val) == -2);
    assert(cuda::std::llroundl(val - T(0.2)) == 1);
    assert(cuda::std::llroundl(-val + T(0.2)) == -1);
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_lrint(T val)
{
  static_assert(cuda::std::is_same_v<decltype(cuda::std::lrint(T{})), long>, "");

  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::lrint(val), val));
  }
  else
  {
    assert(cuda::std::lrint(val) == 2);
    assert(cuda::std::lrint(-val) == -2);
    assert(cuda::std::lrint(val - T(0.2)) == 1);
    assert(cuda::std::lrint(-val + T(0.2)) == -1);
  }
  assert(cuda::std::lrint(T(-0.0)) == -0);
  assert(cuda::std::lrint(T(0.0)) == 0);
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(cuda::std::lrintf(val) == 2);
    assert(cuda::std::lrintf(-val) == -2);
    assert(cuda::std::lrintf(val - T(0.2)) == 1);
    assert(cuda::std::lrintf(-val + T(0.2)) == -1);
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(cuda::std::lrintl(val) == 2);
    assert(cuda::std::lrintl(-val) == -2);
    assert(cuda::std::lrintl(val - T(0.2)) == 1);
    assert(cuda::std::lrintl(-val + T(0.2)) == -1);
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_lround(T val)
{
  static_assert(cuda::std::is_same_v<decltype(cuda::std::lround(T{})), long>, "");

  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::lround(val), val));
  }
  else
  {
    assert(cuda::std::lround(val) == 2);
    assert(cuda::std::lround(-val) == -2);
    assert(cuda::std::lround(val - T(0.2)) == 1);
    assert(cuda::std::lround(-val + T(0.2)) == -1);
  }
  assert(cuda::std::lround(T(-0.0)) == -0);
  assert(cuda::std::lround(T(0.0)) == 0);
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(cuda::std::lroundf(val) == 2);
    assert(cuda::std::lroundf(-val) == -2);
    assert(cuda::std::lroundf(val - T(0.2)) == 1);
    assert(cuda::std::lroundf(-val + T(0.2)) == -1);
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(cuda::std::lroundl(val) == 2);
    assert(cuda::std::lroundl(-val) == -2);
    assert(cuda::std::lroundl(val - T(0.2)) == 1);
    assert(cuda::std::lroundl(-val + T(0.2)) == -1);
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_nearbyint(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::nearbyint(T{})), ret>, "");

  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::nearbyint(val), val));
    assert(eq(cuda::std::nearbyint(-val), -val));
  }
  else
  {
    assert(eq(cuda::std::nearbyint(val), 2));
    assert(eq(cuda::std::nearbyint(-val), -2));
    assert(eq(cuda::std::nearbyint(val - T(0.2)), T(1)));
    assert(eq(cuda::std::nearbyint(-val + T(0.2)), T(-1)));
  }
  assert(eq(cuda::std::nearbyint(T(-0.0)), T(-0.0)));
  assert(eq(cuda::std::nearbyint(T(0.0)), T(0.0)));
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::nearbyintf(val), T(2)));
    assert(eq(cuda::std::nearbyintf(-val), T(-2)));
    assert(eq(cuda::std::nearbyintf(val - T(0.2)), T(1)));
    assert(eq(cuda::std::nearbyintf(-val + T(0.2)), T(-1)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::nearbyintl(val), T(2)));
    assert(eq(cuda::std::nearbyintl(-val), T(-2)));
    assert(eq(cuda::std::nearbyintl(val - T(0.2)), T(1)));
    assert(eq(cuda::std::nearbyintl(-val + T(0.2)), T(-1)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_nextafter(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::nextafter(T{}, T{})), ret>, "");

  unused(val);
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::nextafterf(cuda::std::nextafterf(val, T(10)), T(-10)), val));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::nextafterl(cuda::std::nextafterl(val, T(10)), T(-10)), val));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

#if _CCCL_HAS_LONG_DOUBLE()
template <typename T>
__host__ __device__ void test_nexttoward(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::nexttoward(T{}, long double{})), ret>, "");

  assert(eq(cuda::std::nexttoward(cuda::std::nexttoward(val, long double(10.0)), long double(-10.0)), val));
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::nexttowardf(cuda::std::nexttowardf(val, long double(10.0)), long double(-10.0)), val));
  }
  else if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::nexttowardl(cuda::std::nexttowardl(val, long double(10.0)), long double(-10.0)), val));
  }
}
#endif // _CCCL_HAS_LONG_DOUBLE()

template <typename T>
__host__ __device__ void test_rint(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::rint(T{})), ret>, "");

  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::rint(val), val));
    assert(eq(cuda::std::rint(-val), -val));
  }
  else
  {
    assert(eq(cuda::std::rint(val), 2));
    assert(eq(cuda::std::rint(-val), -2));
    assert(eq(cuda::std::rint(val - T(0.2)), T(1)));
    assert(eq(cuda::std::rint(-val + T(0.2)), T(-1)));
  }
  assert(eq(cuda::std::rint(T(-0.0)), T(-0.0)));
  assert(eq(cuda::std::rint(T(0.0)), T(0.0)));
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::rintf(val), T(2)));
    assert(eq(cuda::std::rintf(-val), T(-2)));
    assert(eq(cuda::std::rintf(val - T(0.2)), T(1)));
    assert(eq(cuda::std::rintf(-val + T(0.2)), T(-1)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::rintl(val), T(2)));
    assert(eq(cuda::std::rintl(-val), T(-2)));
    assert(eq(cuda::std::rintl(val - T(0.2)), T(1)));
    assert(eq(cuda::std::rintl(-val + T(0.2)), T(-1)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_round(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::round(T{})), ret>, "");

  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::round(val), val));
    assert(eq(cuda::std::round(-val), -val));
  }
  else
  {
    assert(eq(cuda::std::round(val), 2));
    assert(eq(cuda::std::round(-val), -2));
    assert(eq(cuda::std::round(val - T(0.2)), T(1)));
    assert(eq(cuda::std::round(-val + T(0.2)), T(-1)));
  }
  assert(eq(cuda::std::round(T(-0.0)), T(-0.0)));
  assert(eq(cuda::std::round(T(0.0)), T(0.0)));
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::roundf(val), T(2)));
    assert(eq(cuda::std::roundf(-val), T(-2)));
    assert(eq(cuda::std::roundf(val - T(0.2)), T(1)));
    assert(eq(cuda::std::roundf(-val + T(0.2)), T(-1)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::roundl(val), T(2)));
    assert(eq(cuda::std::roundl(-val), T(-2)));
    assert(eq(cuda::std::roundl(val - T(0.2)), T(1)));
    assert(eq(cuda::std::roundl(-val + T(0.2)), T(-1)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_trunc(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::trunc(T{})), ret>, "");
  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::trunc(val), val));
    assert(eq(cuda::std::trunc(-val), -val));
  }
  else
  {
    assert(eq(cuda::std::trunc(val), T(1)));
    assert(eq(cuda::std::trunc(-val), T(-1)));
  }
  assert(eq(cuda::std::trunc(T(-0.0)), T(-0.0)));
  assert(eq(cuda::std::trunc(T(0.0)), T(0.0)));
  assert(eq(cuda::std::trunc(T(-cuda::std::numeric_limits<T>::infinity())), -cuda::std::numeric_limits<T>::infinity()));
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::truncf(val), T(1)));
    assert(eq(cuda::std::truncf(-val), T(-1)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::truncl(val), T(1)));
    assert(eq(cuda::std::truncl(-val), T(-1)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test(const T val)
{
  test_ceil<T>(val);
  test_floor<T>(val);
  test_llrint<T>(val);
  test_llround<T>(val);
  test_lrint<T>(val);
  test_lround<T>(val);
  test_nearbyint<T>(val);
  test_nextafter<T>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test_nexttoward<T>(val);
#endif // _CCCL_HAS_LONG_DOUBLE()
  test_rint<T>(val);
  test_round<T>(val);
  test_trunc<T>(val);
}

__host__ __device__ void test(const float val)
{
  test<float>(val);
  test<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>(val);
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>(val);
#endif // _LIBCUDACXX_HAS_NVBF16()

  test<unsigned short>(static_cast<unsigned short>(val));
  test<int>(static_cast<int>(val));
  test<unsigned int>(static_cast<unsigned int>(val));
  test<long>(static_cast<long>(val));
  test<unsigned long>(static_cast<unsigned long>(val));
  test<long long>(static_cast<long long>(val));
  test<unsigned long long>(static_cast<unsigned long long>(val));
}

__global__ void test_global_kernel(float* val)
{
  test(*val);
}

int main(int, char**)
{
  volatile float val = 1.6f;
  test(val);
  return 0;
}
