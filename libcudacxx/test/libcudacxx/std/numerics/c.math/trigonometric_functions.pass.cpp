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
#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4244) // conversion from 'double' to 'float', possible loss of data
TEST_DIAG_SUPPRESS_MSVC(4305) // 'argument': truncation from 'T' to 'float'
TEST_DIAG_SUPPRESS_MSVC(4146) // unary minus operator applied to unsigned type, result still unsigned

template <typename T>
__host__ __device__ void test_cos(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::cos(T{})), ret>, "");

  // 0 is returned unmodified
  assert(eq(cuda::std::cos(val), T(1.0)));
  assert(eq(cuda::std::cos(-val), T(1.0)));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(cuda::std::isnan(cuda::std::cos(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::cos(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::cos(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::cos(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(is_about(cuda::std::cos(pi * T(0.25)), -cuda::std::cos(pi * T(0.75))));
    assert(is_about(cuda::std::cos(pi / T(2.0)), cuda::std::cos(-pi / T(2.0))));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(cuda::std::isnan(cuda::std::cosf(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::cosf(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::cosf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::cosf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(eq(cuda::std::cosf(pi * T(0.25)), -cuda::std::cosf(pi * T(0.75))));
    assert(eq(cuda::std::cosf(pi / T(2.0)), cuda::std::cosf(-pi / T(2.0))));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(cuda::std::isnan(cuda::std::cosl(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::cosl(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::cosl(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::cosl(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(eq(cuda::std::cosl(pi * T(0.25)), -cuda::std::cosl(pi * T(0.75))));
    assert(eq(cuda::std::cosl(pi / T(2.0)), cuda::std::cosl(-pi / T(2.0))));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_sin(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::sin(T{})), ret>, "");

  // 0 is returned unmodified
  assert(eq(cuda::std::sin(val), val));
  assert(eq(cuda::std::sin(-val), -val));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(cuda::std::isnan(cuda::std::sin(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::sin(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::sin(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::sin(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(is_about(cuda::std::sin(pi / T(6.0)), T(0.5)));
    assert(is_about(cuda::std::sin(-pi / T(6.0)), T(-0.5)));
    assert(is_about(cuda::std::sin(pi / T(2.0)), T(1.0)));
    assert(is_about(cuda::std::sin(-pi / T(2.0)), T(-1.0)));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(cuda::std::isnan(cuda::std::sinf(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::sinf(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::sinf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::sinf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(eq(cuda::std::sinf(pi / T(6.0)), T(0.5)));
    assert(eq(cuda::std::sinf(-pi / T(6.0)), T(-0.5)));
    assert(eq(cuda::std::sinf(pi / T(2.0)), T(1.0)));
    assert(eq(cuda::std::sinf(-pi / T(2.0)), T(-1.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(cuda::std::isnan(cuda::std::sinl(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::sinl(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::sinl(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::sinl(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(eq(cuda::std::sinl(pi / T(6.0)), T(0.5)));
    assert(eq(cuda::std::sinl(-pi / T(6.0)), T(-0.5)));
    assert(eq(cuda::std::sinl(pi / T(2.0)), T(1.0)));
    assert(eq(cuda::std::sinl(-pi / T(2.0)), T(-1.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_tan(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::tan(T{})), ret>, "");

  // 0 is returned unmodified
  assert(eq(cuda::std::tan(val), val));
  assert(eq(cuda::std::tan(-val), -val));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(cuda::std::isnan(cuda::std::tan(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::tan(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::tan(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::tan(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(is_about(cuda::std::tan(pi * T(0.25)), T(1.0)));
    assert(is_about(cuda::std::tan(pi * T(0.75)), T(-1.0)));

    // half and bfloat suffer from precision here
    if constexpr (cuda::std::is_floating_point_v<T>)
    {
      assert(is_about(cuda::std::tan(pi * T(1.25)), T(1.0)));
      assert(is_about(cuda::std::tan(pi * T(1.75)), T(-1.0)));
    }
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(cuda::std::isnan(cuda::std::tanf(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::tanf(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::tanf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::tanf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(is_about(cuda::std::tanf(pi * T(0.25)), T(1.0)));
    assert(is_about(cuda::std::tanf(pi * T(0.75)), T(-1.0)));
    assert(is_about(cuda::std::tanf(pi * T(1.25)), T(1.0)));
    assert(is_about(cuda::std::tanf(pi * T(1.75)), T(-1.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(cuda::std::isnan(cuda::std::tanl(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::tanl(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::tanl(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::tanl(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(is_about(cuda::std::tanl(pi * T(0.25)), T(1.0)));
    assert(is_about(cuda::std::tanl(pi * T(0.75)), T(-1.0)));
    assert(is_about(cuda::std::tanl(pi * T(1.25)), T(1.0)));
    assert(is_about(cuda::std::tanl(pi * T(1.75)), T(-1.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test(const T val)
{
  test_cos<T>(val);
  test_sin<T>(val);
  test_tan<T>(val);
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
  volatile float val = 0.0f;
  test(val);
  return 0;
}
