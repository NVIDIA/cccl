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
__host__ __device__ void test_fma(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::fma(T{}, T{}, T{})), ret>, "");

  // fma(x, y, z), fma(y, x, z) are equivalent.
  assert(eq(cuda::std::fma(val, T(3.0), T(2.0)), cuda::std::fma(T(3.0), val, T(2.0))));

  // If one of the arguments is ±0, std::fma(x, y, z) is equivalent to z.
  assert(eq(cuda::std::fma(T(0.0), val, T(2.0)), T(2.0)));
  assert(eq(cuda::std::fma(T(-0.0), val, T(2.0)), T(2.0)));
  assert(eq(cuda::std::fma(val, T(0.0), T(2.0)), T(2.0)));
  assert(eq(cuda::std::fma(val, T(-0.0), T(2.0)), T(2.0)));

  if constexpr (!cuda::std::is_integral_v<T>)
  {
    // If x is zero and y is infinite or if x is infinite and y is zero, and
    //   if z is not a NaN, then NaN is returned and FE_INVALID is raised,
    assert(cuda::std::isnan(cuda::std::fma(cuda::std::numeric_limits<T>::infinity(), T(0.0), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fma(cuda::std::numeric_limits<T>::infinity(), T(-0.0), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fma(T(0.0), cuda::std::numeric_limits<T>::infinity(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fma(T(-0.0), cuda::std::numeric_limits<T>::infinity(), T(3.0))));

    // If x is zero and y is infinite or if x is infinite and y is zero, and
    //   if z is a NaN, then NaN is returned and FE_INVALID may be raised.
    assert(cuda::std::isnan(
      cuda::std::fma(cuda::std::numeric_limits<T>::infinity(), T(0.0), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::fma(cuda::std::numeric_limits<T>::infinity(), T(-0.0), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::fma(T(0.0), cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::fma(T(-0.0), cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::quiet_NaN())));

    // If x * y is an exact infinity and z is an infinity with the opposite sign, NaN is returned and FE_INVALID is
    // raised.
    assert(cuda::std::isnan(
      cuda::std::fma(cuda::std::numeric_limits<T>::infinity(), T(1.0), -cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::fma(-cuda::std::numeric_limits<T>::infinity(), T(1.0), cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::fma(T(1.0), cuda::std::numeric_limits<T>::infinity(), -cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::fma(T(1.0), -cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));

    // If x or y are NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::fma(cuda::std::numeric_limits<T>::quiet_NaN(), val, T(3.0))));
    assert(cuda::std::isnan(cuda::std::fma(val, cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fma(cuda::std::numeric_limits<T>::signaling_NaN(), val, T(3.0))));
    assert(cuda::std::isnan(cuda::std::fma(val, cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));

    assert(cuda::std::isnan(
      cuda::std::fma(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(
      cuda::std::fma(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fma(
      cuda::std::numeric_limits<T>::signaling_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fma(
      cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));

    // If z is NaN, and x * y is not 0 * Inf or Inf * 0, then NaN is returned (without FE_INVALID).
    assert(cuda::std::isnan(cuda::std::fma(T(1.0), val, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::fma(T(1.0), val, cuda::std::numeric_limits<T>::signaling_NaN())));

    // some random value
    assert(is_about(cuda::std::fma(val, T(3.0), T(10.0)), T(16.0)));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    // If x is zero and y is infinite or if x is infinite and y is zero, and
    //   if z is not a NaN, then NaN is returned and FE_INVALID is raised,
    assert(cuda::std::isnan(cuda::std::fmaf(cuda::std::numeric_limits<T>::infinity(), T(0.0), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmaf(cuda::std::numeric_limits<T>::infinity(), T(-0.0), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmaf(T(0.0), cuda::std::numeric_limits<T>::infinity(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmaf(T(-0.0), cuda::std::numeric_limits<T>::infinity(), T(3.0))));

    // If x is zero and y is infinite or if x is infinite and y is zero, and
    //   if z is a NaN, then NaN is returned and FE_INVALID may be raised.
    assert(cuda::std::isnan(
      cuda::std::fmaf(cuda::std::numeric_limits<T>::infinity(), T(0.0), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::fmaf(cuda::std::numeric_limits<T>::infinity(), T(-0.0), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::fmaf(T(0.0), cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::fmaf(T(-0.0), cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::quiet_NaN())));

    // If x * y is an exact infinity and z is an infinity with the opposite sign, NaN is returned and FE_INVALID is
    // raised.
    assert(cuda::std::isnan(
      cuda::std::fmaf(cuda::std::numeric_limits<T>::infinity(), T(1.0), -cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::fmaf(-cuda::std::numeric_limits<T>::infinity(), T(1.0), cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::fmaf(T(1.0), cuda::std::numeric_limits<T>::infinity(), -cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::fmaf(T(1.0), -cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));

    // If x or y are NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::fmaf(cuda::std::numeric_limits<T>::quiet_NaN(), val, T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmaf(val, cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmaf(cuda::std::numeric_limits<T>::signaling_NaN(), val, T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmaf(val, cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));

    assert(cuda::std::isnan(
      cuda::std::fmaf(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(
      cuda::std::fmaf(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmaf(
      cuda::std::numeric_limits<T>::signaling_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmaf(
      cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));

    // If z is NaN, and x * y is not 0 * Inf or Inf * 0, then NaN is returned (without FE_INVALID).
    assert(cuda::std::isnan(cuda::std::fmaf(T(1.0), val, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::fmaf(T(1.0), val, cuda::std::numeric_limits<T>::signaling_NaN())));

    // some random value
    assert(is_about(cuda::std::fmaf(val, T(3.0), T(10.0)), T(16.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    // If x is zero and y is infinite or if x is infinite and y is zero, and
    //   if z is not a NaN, then NaN is returned and FE_INVALID is raised,
    assert(cuda::std::isnan(cuda::std::fmal(cuda::std::numeric_limits<T>::infinity(), T(0.0), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmal(cuda::std::numeric_limits<T>::infinity(), T(-0.0), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmal(T(0.0), cuda::std::numeric_limits<T>::infinity(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmal(T(-0.0), cuda::std::numeric_limits<T>::infinity(), T(3.0))));

    // If x is zero and y is infinite or if x is infinite and y is zero, and
    //   if z is a NaN, then NaN is returned and FE_INVALID may be raised.
    assert(cuda::std::isnan(
      cuda::std::fmal(cuda::std::numeric_limits<T>::infinity(), T(0.0), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::fmal(cuda::std::numeric_limits<T>::infinity(), T(-0.0), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::fmal(T(0.0), cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::fmal(T(-0.0), cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::quiet_NaN())));

    // If x * y is an exact infinity and z is an infinity with the opposite sign, NaN is returned and FE_INVALID is
    // raised.
    assert(cuda::std::isnan(
      cuda::std::fmal(cuda::std::numeric_limits<T>::infinity(), T(1.0), -cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::fmal(-cuda::std::numeric_limits<T>::infinity(), T(1.0), cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::fmal(T(1.0), cuda::std::numeric_limits<T>::infinity(), -cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::fmal(T(1.0), -cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));

    // If x or y are NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::fmal(cuda::std::numeric_limits<T>::quiet_NaN(), val, T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmal(val, cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmal(cuda::std::numeric_limits<T>::signaling_NaN(), val, T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmal(val, cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));

    assert(cuda::std::isnan(
      cuda::std::fmal(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(
      cuda::std::fmal(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmal(
      cuda::std::numeric_limits<T>::signaling_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::fmal(
      cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));

    // If z is NaN, and x * y is not 0 * Inf or Inf * 0, then NaN is returned (without FE_INVALID).
    assert(cuda::std::isnan(cuda::std::fmal(T(1.0), val, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::fmal(T(1.0), val, cuda::std::numeric_limits<T>::signaling_NaN())));

    // some random value
    assert(is_about(cuda::std::fmal(val, T(3.0), T(10.0)), T(16.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test(const T val)
{
  test_fma<T>(val);
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
  volatile float val = 2.0f;
  test(val);
  return 0;
}
