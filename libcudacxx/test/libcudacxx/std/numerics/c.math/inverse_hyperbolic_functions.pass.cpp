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
__host__ __device__ void test_acosh(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::acosh(T{})), ret>, "");

  // If the argument is less than 1
  assert(cuda::std::isnan(cuda::std::acosh(val)));
  assert(cuda::std::isnan(cuda::std::acosh(-val)));
  assert(cuda::std::isnan(cuda::std::acosh(-cuda::std::numeric_limits<T>::infinity())));

  // If the argument is 1, +0 is returned.
  assert(eq(cuda::std::acosh(T(1.0)), val));

  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::acosh(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(cuda::std::isnan(cuda::std::acosh(cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::acosh(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::acosh(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::acosh(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value
    const T expected = T(1.316957896924816795447554795828182);
    assert(is_about(cuda::std::acosh(T(2.0)), expected));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::acoshf(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(cuda::std::isnan(cuda::std::acoshf(cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::acoshf(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::acoshf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::acoshf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value
    const T expected = T(1.316957896924816795447554795828182);
    assert(is_about(cuda::std::acoshf(T(2.0)), expected));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::acoshl(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(cuda::std::isnan(cuda::std::acoshl(cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::acoshl(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::acoshl(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::acoshl(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value
    const T expected = T(1.316957896924816795447554795828182);
    assert(is_about(cuda::std::acoshl(T(2.0)), expected));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_asinh(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::asinh(T{})), ret>, "");

  // If  the argument is ±0 or ±∞, it is returned unmodified.
  assert(eq(cuda::std::asinh(val), val));
  assert(eq(cuda::std::asinh(-val), -val));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    // If  the argument is ±0 or ±∞, it is returned unmodified.
    assert(eq(cuda::std::asinh(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::asinh(-cuda::std::numeric_limits<T>::infinity()), -cuda::std::numeric_limits<T>::infinity()));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::asinh(cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::asinh(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::asinh(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::asinh(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // Is symmetric
    assert(eq(cuda::std::asinh(T(0.5)), -cuda::std::asinh(T(-0.5))));
    assert(eq(cuda::std::asinh(T(0.25)), -cuda::std::asinh(T(-0.25))));

    // Some random value
    const T expected = T(0.8813735870195430477380682532384526);
    assert(is_about(cuda::std::asinh(T(1.0)), expected));
    assert(is_about(cuda::std::asinh(T(-1.0)), -expected));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    // If  the argument is ±0 or ±∞, it is returned unmodified.
    assert(eq(cuda::std::asinhf(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::asinhf(-cuda::std::numeric_limits<T>::infinity()), -cuda::std::numeric_limits<T>::infinity()));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::asinhf(cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::asinhf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::asinhf(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::asinhf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // Is symmetric
    assert(eq(cuda::std::asinhf(T(0.5)), -cuda::std::asinhf(T(-0.5))));
    assert(eq(cuda::std::asinhf(T(0.25)), -cuda::std::asinhf(T(-0.25))));

    // Some random value
    const T expected = T(0.8813735870195430477380682532384526);
    assert(is_about(cuda::std::asinhf(T(1.0)), expected));
    assert(is_about(cuda::std::asinhf(T(-1.0)), -expected));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    // If  the argument is ±0 or ±∞, it is returned unmodified.
    assert(eq(cuda::std::asinhl(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::asinhl(-cuda::std::numeric_limits<T>::infinity()), -cuda::std::numeric_limits<T>::infinity()));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::asinhl(cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::asinhl(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::asinhl(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::asinhl(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // Is symmetric
    assert(eq(cuda::std::asinhl(T(0.5)), -cuda::std::asinhl(T(-0.5))));
    assert(eq(cuda::std::asinhl(T(0.25)), -cuda::std::asinhl(T(-0.25))));

    // Some random value
    const T expected = T(0.8813735870195430477380682532384526);
    assert(is_about(cuda::std::asinhl(T(1.0)), expected));
    assert(is_about(cuda::std::asinhl(T(-1.0)), -expected));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_atanh(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::atanh(T{})), ret>, "");

  // 0 is returned unmodified
  assert(eq(cuda::std::atanh(val), val));
  assert(eq(cuda::std::atanh(-val), -val));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::atanh(T(1.0)), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::atanh(T(-1.0)), -cuda::std::numeric_limits<T>::infinity()));
    assert(cuda::std::isnan(cuda::std::atanh(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::atanh(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // If |x| > 1, NaN is returned
    assert(cuda::std::isnan(cuda::std::atanh(T(2.0))));
    assert(cuda::std::isnan(cuda::std::atanh(T(-2.0))));

    // Is symmetric
    assert(eq(cuda::std::atanh(T(0.25)), -cuda::std::atanh(T(-0.25))));
    assert(eq(cuda::std::atanh(T(0.5)), -cuda::std::atanh(T(-0.5))));

    const T expected =
      cuda::std::is_same_v<T, double> ? T(0.5493061443340548910541087934689131) : T(0.5493061542510986328125);
    assert(is_about(cuda::std::atanh(T(0.5)), expected));
    assert(is_about(cuda::std::atanh(T(-0.5)), -expected));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::atanhf(T(1.0)), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::atanhf(T(-1.0)), -cuda::std::numeric_limits<T>::infinity()));
    assert(cuda::std::isnan(cuda::std::atanhf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::atanhf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // If |x| > 1, NaN is returned
    assert(cuda::std::isnan(cuda::std::atanhf(T(2.0))));
    assert(cuda::std::isnan(cuda::std::atanhf(T(-2.0))));

    // Is symmetric
    assert(eq(cuda::std::atanhf(T(0.25)), -cuda::std::atanhf(T(-0.25))));
    assert(eq(cuda::std::atanhf(T(0.5)), -cuda::std::atanhf(T(-0.5))));

    // Some random value
    const T expected = T(0.5493061542510986328125);
    assert(is_about(cuda::std::atanhf(T(0.5)), expected));
    assert(is_about(cuda::std::atanhf(T(-0.5)), -expected));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::atanhl(T(1.0)), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::atanhl(T(-1.0)), -cuda::std::numeric_limits<T>::infinity()));
    assert(cuda::std::isnan(cuda::std::atanhl(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::atanhl(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // If |x| > 1, NaN is returned
    assert(cuda::std::isnan(cuda::std::atanhl(T(2.0))));
    assert(cuda::std::isnan(cuda::std::atanhl(T(-2.0))));

    // Is symmetric
    assert(eq(cuda::std::atanhl(T(0.25)), -cuda::std::atanhl(T(-0.25))));
    assert(eq(cuda::std::atanhl(T(0.5)), -cuda::std::atanhl(T(-0.5))));

    const T expected = T(0.5493061443340548910541087934689131);
    assert(is_about(cuda::std::atanhl(T(0.5)), expected));
    assert(is_about(cuda::std::atanhl(T(-0.5)), -expected));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test(const T val)
{
  test_acosh<T>(val);
  test_asinh<T>(val);
  test_atanh<T>(val);
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
