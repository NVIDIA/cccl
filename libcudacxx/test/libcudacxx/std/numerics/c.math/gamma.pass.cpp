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
__host__ __device__ void test_lgamma(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::lgamma(T{})), ret>, "");

  // If the argument is 1, +0 is returned.
  assert(eq(cuda::std::lgamma(T(1.0)), val));

  // If the argument is 2, +0 is returned.
  assert(eq(cuda::std::lgamma(T(2.0)), val));

  // If the argument is a negative integer, +∞ is returned.
  assert(eq(cuda::std::lgamma(-2), cuda::std::numeric_limits<double>::infinity()));

  // If the argument is ±0, +∞ is returned.
  assert(eq(cuda::std::lgamma(val), cuda::std::numeric_limits<ret>::infinity()));

  if constexpr (!cuda::std::is_integral_v<T>)
  {
    // If the argument is ±0, +∞ is returned.
    assert(eq(cuda::std::lgamma(-val), cuda::std::numeric_limits<T>::infinity()));

    // If the argument is ±∞, +∞ is returned.
    assert(eq(cuda::std::lgamma(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::lgamma(-cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::lgamma(cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::lgamma(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::lgamma(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::lgamma(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value
    const T expected = T(0.5723649429247000819387380943226162);
    assert(is_about(cuda::std::lgamma(T(0.5)), expected));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    // If the argument is 1, +0 is returned.
    assert(eq(cuda::std::lgammaf(T(1.0)), val));

    // If the argument is 2, +0 is returned.
    assert(eq(cuda::std::lgammaf(T(2.0)), val));

    // If the argument is ±0, +∞ is returned.
    assert(eq(cuda::std::lgammaf(val), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::lgammaf(-val), cuda::std::numeric_limits<T>::infinity()));

    // If the argument is ±∞, +∞ is returned.
    assert(eq(cuda::std::lgammaf(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::lgammaf(-cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::lgammaf(cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::lgammaf(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::lgammaf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::lgammaf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value
    const T expected = T(0.5723649429247000819387380943226162);
    assert(is_about(cuda::std::lgammaf(T(0.5)), expected));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    // If the argument is 1, +0 is returned.
    assert(eq(cuda::std::lgammal(T(1.0)), val));

    // If the argument is 2, +0 is returned.
    assert(eq(cuda::std::lgammal(T(2.0)), val));

    // If the argument is ±0, +∞ is returned.
    assert(eq(cuda::std::lgammal(val), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::lgammal(-val), cuda::std::numeric_limits<T>::infinity()));

    // If the argument is ±∞, +∞ is returned.
    assert(eq(cuda::std::lgammal(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::lgammal(-cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::lgammal(cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::lgammal(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::lgammal(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::lgammal(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value
    const T expected = T(0.5723649429247000819387380943226162);
    assert(is_about(cuda::std::lgammal(T(0.5)), expected));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_tgamma(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::tgamma(T{})), ret>, "");

  // If the argument is a negative integer.
  assert(cuda::std::isnan(cuda::std::tgamma(-2)));

  // Some characteristic value
  assert(eq(cuda::std::tgamma(T(2.0)), T(1.0)));

  // If the argument is ±0, ±∞ is returned.
  assert(eq(cuda::std::tgamma(val), cuda::std::numeric_limits<ret>::infinity()));

  if constexpr (!cuda::std::is_integral_v<T>)
  {
    // If the argument is ±0, ±∞ is returned.
    assert(eq(cuda::std::tgamma(-val), -cuda::std::numeric_limits<ret>::infinity()));

    // If the argument is -∞, NaN is returned.
    assert(cuda::std::isnan(cuda::std::tgamma(-cuda::std::numeric_limits<T>::infinity())));

    // If the argument is +∞, +∞ is returned.
    assert(eq(cuda::std::tgamma(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::tgamma(cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::tgamma(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::tgamma(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::tgamma(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value
    const T expected = T(1.772453850905516103964032481599133);
    assert(is_about(cuda::std::tgamma(T(0.5)), expected));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    // If the argument is ±0, ±∞ is returned.
    assert(eq(cuda::std::tgammaf(val), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::tgammaf(-val), -cuda::std::numeric_limits<T>::infinity()));

    // If the argument is -∞, NaN is returned.
    assert(cuda::std::isnan(cuda::std::tgammaf(-cuda::std::numeric_limits<T>::infinity())));

    // If the argument is +∞, +∞ is returned.
    assert(eq(cuda::std::tgammaf(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::tgammaf(cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::tgammaf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::tgammaf(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::tgammaf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value
    const T expected = T(1.772453850905516103964032481599133);
    assert(is_about(cuda::std::tgammaf(T(0.5)), expected));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    // If the argument is ±0, ±∞ is returned.
    assert(eq(cuda::std::tgammal(val), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::tgammal(-val), -cuda::std::numeric_limits<T>::infinity()));

    // If the argument is -∞, NaN is returned.
    assert(cuda::std::isnan(cuda::std::tgammal(-cuda::std::numeric_limits<T>::infinity())));

    // If the argument is +∞, +∞ is returned.
    assert(eq(cuda::std::tgammal(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::tgammal(cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::tgammal(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::tgammal(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::tgammal(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value
    const T expected = T(1.772453850905516103964032481599133);
    assert(is_about(cuda::std::tgammal(T(0.5)), expected));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test(const T val)
{
  test_lgamma<T>(val);
  test_tgamma<T>(val);
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
