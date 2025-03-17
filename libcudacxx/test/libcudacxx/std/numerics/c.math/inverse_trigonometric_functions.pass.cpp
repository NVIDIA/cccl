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
__host__ __device__ void test_acos(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::acos(T{})), ret>, "");

  assert(eq(cuda::std::acos(T(1.0)), val));
  assert(cuda::std::isnan(cuda::std::acos(T(2.0))));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(cuda::std::isnan(cuda::std::acos(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::acos(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::acos(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::acos(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(is_about(cuda::std::acos(T(0.5)), pi / T(3.0)));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(cuda::std::isnan(cuda::std::acosf(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::acosf(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::acosf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::acosf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(is_about(cuda::std::acosf(T(0.5)), pi / T(3.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(cuda::std::isnan(cuda::std::acosl(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::acosl(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::acosl(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::acosl(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(is_about(cuda::std::acosl(T(0.5)), pi / T(3.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_asin(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::asin(T{})), ret>, "");

  // 0 is returned unmodified
  assert(eq(cuda::std::asin(val), val));
  assert(eq(cuda::std::asin(-val), -val));
  assert(cuda::std::isnan(cuda::std::asin(T(2.0))));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(cuda::std::isnan(cuda::std::asin(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::asin(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::asin(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::asin(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(is_about(cuda::std::asin(T(1.0)), pi / T(2.0)));
    assert(is_about(cuda::std::asin(T(-0.5)), -pi / T(6.0)));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(cuda::std::isnan(cuda::std::asinf(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::asinf(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::asinf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::asinf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(is_about(cuda::std::asinf(T(1.0)), pi / T(2.0)));
    assert(is_about(cuda::std::asinf(T(-0.5)), -pi / T(6.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(cuda::std::isnan(cuda::std::asinl(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::asinl(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::asinl(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::asinl(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T pi = T(3.141592653589793238462643383279502);
    assert(is_about(cuda::std::asinl(T(1.0)), pi / T(2.0)));
    assert(is_about(cuda::std::asinl(T(-0.5)), -pi / T(6.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_atan(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::atan(T{})), ret>, "");

  // 0 is returned unmodified
  assert(eq(cuda::std::atan(val), val));
  assert(eq(cuda::std::atan(-val), -val));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    const T pi = T(3.141592653589793238462643383279502);
    assert(eq(cuda::std::atan(cuda::std::numeric_limits<T>::infinity()), pi / T(2.0)));
    assert(eq(cuda::std::atan(-cuda::std::numeric_limits<T>::infinity()), -pi / T(2.0)));
    assert(cuda::std::isnan(cuda::std::atan(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::atan(-cuda::std::numeric_limits<T>::quiet_NaN())));

    assert(is_about(cuda::std::atan(T(1.0)), pi / T(4.0)));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    const T pi = T(3.141592653589793238462643383279502);
    assert(eq(cuda::std::atanf(cuda::std::numeric_limits<T>::infinity()), pi / T(2.0)));
    assert(eq(cuda::std::atanf(-cuda::std::numeric_limits<T>::infinity()), -pi / T(2.0)));
    assert(cuda::std::isnan(cuda::std::atanf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::atanf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    assert(is_about(cuda::std::atanf(T(1.0)), pi / T(4.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    const T pi = T(3.141592653589793238462643383279502);
    assert(eq(cuda::std::atanl(cuda::std::numeric_limits<T>::infinity()), pi / T(2.0)));
    assert(eq(cuda::std::atanl(-cuda::std::numeric_limits<T>::infinity()), -pi / T(2.0)));
    assert(cuda::std::isnan(cuda::std::atanl(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::atanl(-cuda::std::numeric_limits<T>::quiet_NaN())));

    assert(is_about(cuda::std::atanl(T(1.0)), pi / T(4.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_atan2(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::atan2(T{}, T{})), ret>, "");

  // If y is ±0 and x is positive or +0, ±0 is returned.
  assert(eq(cuda::std::atan2(val, val), val));
  assert(eq(cuda::std::atan2(-val, val), -val));
  assert(eq(cuda::std::atan2(val, T(2.0)), val));
  assert(eq(cuda::std::atan2(-val, T(2.0)), val));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    const T pi = T(3.141592653589793238462643383279502);
    assert(cuda::std::isnan(cuda::std::atan2(val, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::atan2(val, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::atan2(cuda::std::numeric_limits<T>::signaling_NaN(), val)));
    assert(cuda::std::isnan(cuda::std::atan2(cuda::std::numeric_limits<T>::quiet_NaN(), val)));

    // If y is ±0 and x is negative or -0, ±π is returned.
    assert(eq(cuda::std::atan2(-val, -val), -pi));
    assert(eq(cuda::std::atan2(-val, T(-2.0)), -pi));
    assert(eq(cuda::std::atan2(val, -val), pi));
    assert(eq(cuda::std::atan2(val, T(-2.0)), pi));

    // If y is ±∞ and x is finite, ±π/2 is returned.
    assert(eq(cuda::std::atan2(cuda::std::numeric_limits<T>::infinity(), T(2.0)), pi / T(2.0)));
    assert(eq(cuda::std::atan2(cuda::std::numeric_limits<T>::infinity(), T(-2.0)), pi / T(2.0)));
    assert(eq(cuda::std::atan2(-cuda::std::numeric_limits<T>::infinity(), T(2.0)), -pi / T(2.0)));
    assert(eq(cuda::std::atan2(-cuda::std::numeric_limits<T>::infinity(), T(-2.0)), -pi / T(2.0)));

    // If y is ±∞ and x is -∞, ±3π/4 is returned.
    assert(eq(cuda::std::atan2(cuda::std::numeric_limits<T>::infinity(), -cuda::std::numeric_limits<T>::infinity()),
              pi * T(0.75)));
    assert(eq(cuda::std::atan2(-cuda::std::numeric_limits<T>::infinity(), -cuda::std::numeric_limits<T>::infinity()),
              -pi * T(0.75)));

    // If y is ±∞ and x is +∞, ±π/4 is returned.
    assert(eq(cuda::std::atan2(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity()),
              pi / T(4.0)));
    assert(eq(cuda::std::atan2(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity()),
              -pi / T(4.0)));

    // If x is ±0 and y is negative, -π/2 is returned.
    assert(eq(cuda::std::atan2(T(-2.0), val), -pi / T(2.0)));
    assert(eq(cuda::std::atan2(T(-2.0), -val), -pi / T(2.0)));

    // If x is ±0 and y is positive, +π/2 is returned.
    assert(eq(cuda::std::atan2(T(2.0), val), pi / T(2.0)));
    assert(eq(cuda::std::atan2(T(2.0), -val), pi / T(2.0)));

    // If x is -∞ and y is finite and positive, +π is returned.
    assert(eq(cuda::std::atan2(T(2.0), -cuda::std::numeric_limits<T>::infinity()), pi));
    assert(eq(cuda::std::atan2(T(2.0), -cuda::std::numeric_limits<T>::infinity()), pi));

    // If x is -∞ and y is finite and negative, -π is returned.
    assert(eq(cuda::std::atan2(T(-2.0), -cuda::std::numeric_limits<T>::infinity()), -pi));
    assert(eq(cuda::std::atan2(T(-2.0), -cuda::std::numeric_limits<T>::infinity()), -pi));

    // If x is +∞ and y is finite and positive, +0 is returned.
    assert(eq(cuda::std::atan2(T(2.0), cuda::std::numeric_limits<T>::infinity()), val));
    assert(eq(cuda::std::atan2(T(2.0), cuda::std::numeric_limits<T>::infinity()), val));

    // If x is +∞ and y is finite and negative, -0 is returned.
    assert(eq(cuda::std::atan2(T(-2.0), cuda::std::numeric_limits<T>::infinity()), -val));
    assert(eq(cuda::std::atan2(T(-2.0), cuda::std::numeric_limits<T>::infinity()), -val));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    const T pi = T(3.141592653589793238462643383279502);
    assert(cuda::std::isnan(cuda::std::atan2f(val, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::atan2f(val, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::atan2f(cuda::std::numeric_limits<T>::signaling_NaN(), val)));
    assert(cuda::std::isnan(cuda::std::atan2f(cuda::std::numeric_limits<T>::quiet_NaN(), val)));

    // If y is ±0 and x is negative or -0, ±π is returned.
    assert(eq(cuda::std::atan2f(-val, -val), -pi));
    assert(eq(cuda::std::atan2f(-val, T(-2.0)), -pi));
    assert(eq(cuda::std::atan2f(val, -val), pi));
    assert(eq(cuda::std::atan2f(val, T(-2.0)), pi));

    // If y is ±∞ and x is finite, ±π/2 is returned.
    assert(eq(cuda::std::atan2f(cuda::std::numeric_limits<T>::infinity(), T(2.0)), pi / T(2.0)));
    assert(eq(cuda::std::atan2f(cuda::std::numeric_limits<T>::infinity(), T(-2.0)), pi / T(2.0)));
    assert(eq(cuda::std::atan2f(-cuda::std::numeric_limits<T>::infinity(), T(2.0)), -pi / T(2.0)));
    assert(eq(cuda::std::atan2f(-cuda::std::numeric_limits<T>::infinity(), T(-2.0)), -pi / T(2.0)));

    // If y is ±∞ and x is -∞, ±3π/4 is returned.
    assert(eq(cuda::std::atan2f(cuda::std::numeric_limits<T>::infinity(), -cuda::std::numeric_limits<T>::infinity()),
              pi * T(0.75)));
    assert(eq(cuda::std::atan2f(-cuda::std::numeric_limits<T>::infinity(), -cuda::std::numeric_limits<T>::infinity()),
              -pi * T(0.75)));

    // If y is ±∞ and x is +∞, ±π/4 is returned.
    assert(eq(cuda::std::atan2f(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity()),
              pi / T(4.0)));
    assert(eq(cuda::std::atan2f(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity()),
              -pi / T(4.0)));

    // If x is ±0 and y is negative, -π/2 is returned.
    assert(eq(cuda::std::atan2f(T(-2.0), val), -pi / T(2.0)));
    assert(eq(cuda::std::atan2f(T(-2.0), -val), -pi / T(2.0)));

    // If x is ±0 and y is positive, +π/2 is returned.
    assert(eq(cuda::std::atan2f(T(2.0), val), pi / T(2.0)));
    assert(eq(cuda::std::atan2f(T(2.0), -val), pi / T(2.0)));

    // If x is -∞ and y is finite and positive, +π is returned.
    assert(eq(cuda::std::atan2f(T(2.0), -cuda::std::numeric_limits<T>::infinity()), pi));
    assert(eq(cuda::std::atan2f(T(2.0), -cuda::std::numeric_limits<T>::infinity()), pi));

    // If x is -∞ and y is finite and negative, -π is returned.
    assert(eq(cuda::std::atan2f(T(-2.0), -cuda::std::numeric_limits<T>::infinity()), -pi));
    assert(eq(cuda::std::atan2f(T(-2.0), -cuda::std::numeric_limits<T>::infinity()), -pi));

    // If x is +∞ and y is finite and positive, +0 is returned.
    assert(eq(cuda::std::atan2f(T(2.0), cuda::std::numeric_limits<T>::infinity()), val));
    assert(eq(cuda::std::atan2f(T(2.0), cuda::std::numeric_limits<T>::infinity()), val));

    // If x is +∞ and y is finite and negative, -0 is returned.
    assert(eq(cuda::std::atan2f(T(-2.0), cuda::std::numeric_limits<T>::infinity()), -val));
    assert(eq(cuda::std::atan2f(T(-2.0), cuda::std::numeric_limits<T>::infinity()), -val));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    const T pi = T(3.141592653589793238462643383279502);
    assert(cuda::std::isnan(cuda::std::atan2l(val, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::atan2l(val, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::atan2l(cuda::std::numeric_limits<T>::signaling_NaN(), val)));
    assert(cuda::std::isnan(cuda::std::atan2l(cuda::std::numeric_limits<T>::quiet_NaN(), val)));

    // If y is ±0 and x is negative or -0, ±π is returned.
    assert(eq(cuda::std::atan2l(-val, -val), -pi));
    assert(eq(cuda::std::atan2l(-val, T(-2.0)), -pi));
    assert(eq(cuda::std::atan2l(val, -val), pi));
    assert(eq(cuda::std::atan2l(val, T(-2.0)), pi));

    // If y is ±∞ and x is finite, ±π/2 is returned.
    assert(eq(cuda::std::atan2l(cuda::std::numeric_limits<T>::infinity(), T(2.0)), pi / T(2.0)));
    assert(eq(cuda::std::atan2l(cuda::std::numeric_limits<T>::infinity(), T(-2.0)), pi / T(2.0)));
    assert(eq(cuda::std::atan2l(-cuda::std::numeric_limits<T>::infinity(), T(2.0)), -pi / T(2.0)));
    assert(eq(cuda::std::atan2l(-cuda::std::numeric_limits<T>::infinity(), T(-2.0)), -pi / T(2.0)));

    // If y is ±∞ and x is -∞, ±3π/4 is returned.
    assert(eq(cuda::std::atan2l(cuda::std::numeric_limits<T>::infinity(), -cuda::std::numeric_limits<T>::infinity()),
              pi * T(0.75)));
    assert(eq(cuda::std::atan2l(-cuda::std::numeric_limits<T>::infinity(), -cuda::std::numeric_limits<T>::infinity()),
              -pi * T(0.75)));

    // If y is ±∞ and x is +∞, ±π/4 is returned.
    assert(eq(cuda::std::atan2l(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity()),
              pi / T(4.0)));
    assert(eq(cuda::std::atan2l(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity()),
              -pi / T(4.0)));

    // If x is ±0 and y is negative, -π/2 is returned.
    assert(eq(cuda::std::atan2l(T(-2.0), val), -pi / T(2.0)));
    assert(eq(cuda::std::atan2l(T(-2.0), -val), -pi / T(2.0)));

    // If x is ±0 and y is positive, +π/2 is returned.
    assert(eq(cuda::std::atan2l(T(2.0), val), pi / T(2.0)));
    assert(eq(cuda::std::atan2l(T(2.0), -val), pi / T(2.0)));

    // If x is -∞ and y is finite and positive, +π is returned.
    assert(eq(cuda::std::atan2l(T(2.0), -cuda::std::numeric_limits<T>::infinity()), pi));
    assert(eq(cuda::std::atan2l(T(2.0), -cuda::std::numeric_limits<T>::infinity()), pi));

    // If x is -∞ and y is finite and negative, -π is returned.
    assert(eq(cuda::std::atan2l(T(-2.0), -cuda::std::numeric_limits<T>::infinity()), -pi));
    assert(eq(cuda::std::atan2l(T(-2.0), -cuda::std::numeric_limits<T>::infinity()), -pi));

    // If x is +∞ and y is finite and positive, +0 is returned.
    assert(eq(cuda::std::atan2l(T(2.0), cuda::std::numeric_limits<T>::infinity()), val));
    assert(eq(cuda::std::atan2l(T(2.0), cuda::std::numeric_limits<T>::infinity()), val));

    // If x is +∞ and y is finite and negative, -0 is returned.
    assert(eq(cuda::std::atan2l(T(-2.0), cuda::std::numeric_limits<T>::infinity()), -val));
    assert(eq(cuda::std::atan2l(T(-2.0), cuda::std::numeric_limits<T>::infinity()), -val));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test(const T val)
{
  test_acos<T>(val);
  test_asin<T>(val);
  test_atan<T>(val);
  test_atan2<T>(val);
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
