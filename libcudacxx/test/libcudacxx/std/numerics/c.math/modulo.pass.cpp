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
__host__ __device__ void test_fmod(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::fmod(T{}, T{})), ret>, "");

  const T x = T(13.23456789);
  const T y = T(3.456789123);

  // The result has the same sign as the same sign as x
  assert(eq(cuda::std::fmod(val, x), val));
  assert(eq(cuda::std::fmod(-val, x), -val));

  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::fmod(x, y), T(1.0)));
  }
  else
  {
    // If x is ±0 and y is not zero, ±0 is returned.
    assert(eq(cuda::std::fmod(val, x), val));
    assert(eq(cuda::std::fmod(-val, x), -val));
    assert(eq(cuda::std::fmod(val, -x), val));
    assert(eq(cuda::std::fmod(-val, -x), -val));

    assert(eq(cuda::std::fmod(val, cuda::std::numeric_limits<T>::infinity()), val));
    assert(eq(cuda::std::fmod(-val, cuda::std::numeric_limits<T>::infinity()), -val));

    // If x is ±∞ and y is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::fmod(cuda::std::numeric_limits<T>::infinity(), val)));
    assert(cuda::std::isnan(cuda::std::fmod(-cuda::std::numeric_limits<T>::infinity(), val)));
    assert(cuda::std::isnan(cuda::std::fmod(cuda::std::numeric_limits<T>::infinity(), x)));
    assert(cuda::std::isnan(cuda::std::fmod(-cuda::std::numeric_limits<T>::infinity(), x)));
    assert(cuda::std::isnan(
      cuda::std::fmod(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::fmod(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::fmod(x, val)));
    assert(cuda::std::isnan(cuda::std::fmod(-x, val)));

    // If y is ±∞ and x is finite, x is returned.
    assert(eq(cuda::std::fmod(x, cuda::std::numeric_limits<T>::infinity()), x));
    assert(eq(cuda::std::fmod(x, -cuda::std::numeric_limits<T>::infinity()), x));
    assert(eq(cuda::std::fmod(-x, cuda::std::numeric_limits<T>::infinity()), -x));
    assert(eq(cuda::std::fmod(-x, -cuda::std::numeric_limits<T>::infinity()), -x));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::fmod(x, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::fmod(-x, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::fmod(cuda::std::numeric_limits<T>::quiet_NaN(), x)));
    assert(cuda::std::isnan(cuda::std::fmod(cuda::std::numeric_limits<T>::quiet_NaN(), -x)));
    assert(cuda::std::isnan(cuda::std::fmod(x, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::fmod(-x, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::fmod(cuda::std::numeric_limits<T>::signaling_NaN(), x)));
    assert(cuda::std::isnan(cuda::std::fmod(cuda::std::numeric_limits<T>::signaling_NaN(), -x)));
    assert(cuda::std::isnan(
      cuda::std::fmod(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::fmod(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value
    const T quotient = T(3.0);
    assert(is_about(cuda::std::fmod(x, y), (x - quotient * y)));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    // If x is ±0 and y is not zero, ±0 is returned.
    assert(eq(cuda::std::fmodf(val, x), val));
    assert(eq(cuda::std::fmodf(-val, x), -val));
    assert(eq(cuda::std::fmodf(val, -x), val));
    assert(eq(cuda::std::fmodf(-val, -x), -val));

    assert(eq(cuda::std::fmodf(val, cuda::std::numeric_limits<T>::infinity()), val));
    assert(eq(cuda::std::fmodf(-val, cuda::std::numeric_limits<T>::infinity()), -val));

    // If x is ±∞ and y is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::fmodf(cuda::std::numeric_limits<T>::infinity(), val)));
    assert(cuda::std::isnan(cuda::std::fmodf(-cuda::std::numeric_limits<T>::infinity(), val)));
    assert(cuda::std::isnan(cuda::std::fmodf(cuda::std::numeric_limits<T>::infinity(), x)));
    assert(cuda::std::isnan(cuda::std::fmodf(-cuda::std::numeric_limits<T>::infinity(), x)));
    assert(cuda::std::isnan(
      cuda::std::fmodf(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::fmodf(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::fmodf(x, val)));
    assert(cuda::std::isnan(cuda::std::fmodf(-x, val)));

    // If y is ±∞ and x is finite, x is returned.
    assert(eq(cuda::std::fmodf(x, cuda::std::numeric_limits<T>::infinity()), x));
    assert(eq(cuda::std::fmodf(x, -cuda::std::numeric_limits<T>::infinity()), x));
    assert(eq(cuda::std::fmodf(-x, cuda::std::numeric_limits<T>::infinity()), -x));
    assert(eq(cuda::std::fmodf(-x, -cuda::std::numeric_limits<T>::infinity()), -x));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::fmodf(x, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::fmodf(-x, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::fmodf(cuda::std::numeric_limits<T>::quiet_NaN(), x)));
    assert(cuda::std::isnan(cuda::std::fmodf(cuda::std::numeric_limits<T>::quiet_NaN(), -x)));
    assert(cuda::std::isnan(cuda::std::fmodf(x, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::fmodf(-x, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::fmodf(cuda::std::numeric_limits<T>::signaling_NaN(), x)));
    assert(cuda::std::isnan(cuda::std::fmodf(cuda::std::numeric_limits<T>::signaling_NaN(), -x)));
    assert(cuda::std::isnan(
      cuda::std::fmodf(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::fmodf(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value
    const T quotient = T(3.0);
    assert(is_about(cuda::std::fmodf(x, y), (x - quotient * y)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    // If x is ±0 and y is not zero, ±0 is returned.
    assert(eq(cuda::std::fmodl(val, x), val));
    assert(eq(cuda::std::fmodl(-val, x), -val));
    assert(eq(cuda::std::fmodl(val, -x), val));
    assert(eq(cuda::std::fmodl(-val, -x), -val));

    assert(eq(cuda::std::fmodl(val, cuda::std::numeric_limits<T>::infinity()), val));
    assert(eq(cuda::std::fmodl(-val, cuda::std::numeric_limits<T>::infinity()), -val));

    // If x is ±∞ and y is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::fmodl(cuda::std::numeric_limits<T>::infinity(), val)));
    assert(cuda::std::isnan(cuda::std::fmodl(-cuda::std::numeric_limits<T>::infinity(), val)));
    assert(cuda::std::isnan(cuda::std::fmodl(cuda::std::numeric_limits<T>::infinity(), x)));
    assert(cuda::std::isnan(cuda::std::fmodl(-cuda::std::numeric_limits<T>::infinity(), x)));
    assert(cuda::std::isnan(
      cuda::std::fmodl(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::fmodl(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::fmodl(x, val)));
    assert(cuda::std::isnan(cuda::std::fmodl(-x, val)));

    // If y is ±∞ and x is finite, x is returned.
    assert(eq(cuda::std::fmodl(x, cuda::std::numeric_limits<T>::infinity()), x));
    assert(eq(cuda::std::fmodl(x, -cuda::std::numeric_limits<T>::infinity()), x));
    assert(eq(cuda::std::fmodl(-x, cuda::std::numeric_limits<T>::infinity()), -x));
    assert(eq(cuda::std::fmodl(-x, -cuda::std::numeric_limits<T>::infinity()), -x));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::fmodl(x, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::fmodl(-x, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::fmodl(cuda::std::numeric_limits<T>::quiet_NaN(), x)));
    assert(cuda::std::isnan(cuda::std::fmodl(cuda::std::numeric_limits<T>::quiet_NaN(), -x)));
    assert(cuda::std::isnan(cuda::std::fmodl(x, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::fmodl(-x, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::fmodl(cuda::std::numeric_limits<T>::signaling_NaN(), x)));
    assert(cuda::std::isnan(cuda::std::fmodl(cuda::std::numeric_limits<T>::signaling_NaN(), -x)));
    assert(cuda::std::isnan(
      cuda::std::fmodl(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::fmodl(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value
    const T quotient = T(3.0);
    assert(is_about(cuda::std::fmodl(x, y), (x - quotient * y)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_modf(T val)
{
  static_assert(cuda::std::is_same_v<decltype(cuda::std::modf(T{}, static_cast<T*>(nullptr))), T>, "");
  {
    // If num is ±0, ±0 is returned, and ±0 is stored in *iptr
    T integral = cuda::std::numeric_limits<T>::infinity();
    assert(eq(cuda::std::modf(val, &integral), val));
    assert(eq(integral, val));

    integral = cuda::std::numeric_limits<T>::infinity();
    assert(eq(cuda::std::modf(-val, &integral), -val));
    assert(eq(integral, -val));

    // If num is ±∞, ±0 is returned, and ±∞ is stored in *iptr.
    integral = T(-1.0);
    assert(eq(cuda::std::modf(cuda::std::numeric_limits<T>::infinity(), &integral), val));
    assert(eq(integral, cuda::std::numeric_limits<T>::infinity()));

    // If num is NaN, NaN is returned, and NaN is stored in *iptr.
    integral = T(-1.0);
    assert(cuda::std::isnan(cuda::std::modf(cuda::std::numeric_limits<T>::quiet_NaN(), &integral)));
    assert(cuda::std::isnan(integral));

    integral = T(-1.0);
    assert(cuda::std::isnan(cuda::std::modf(cuda::std::numeric_limits<T>::signaling_NaN(), &integral)));
    assert(cuda::std::isnan(integral));

    // some random value
    integral         = cuda::std::numeric_limits<T>::infinity();
    const T x        = T(12.3456789);
    const T expected = x - T(12.0);
    assert(is_about(cuda::std::modf(x, &integral), expected));
    assert(eq(integral, T(12.0)));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    // If num is ±0, ±0 is returned, and ±0 is stored in *iptr
    T integral = cuda::std::numeric_limits<T>::infinity();
    assert(eq(cuda::std::modff(val, &integral), val));
    assert(eq(integral, val));

    integral = cuda::std::numeric_limits<T>::infinity();
    assert(eq(cuda::std::modff(-val, &integral), -val));
    assert(eq(integral, -val));

    // If num is ±∞, ±0 is returned, and ±∞ is stored in *iptr.
    integral = T(-1.0);
    assert(eq(cuda::std::modff(cuda::std::numeric_limits<T>::infinity(), &integral), val));
    assert(eq(integral, cuda::std::numeric_limits<T>::infinity()));

    // If num is NaN, NaN is returned, and NaN is stored in *iptr.
    integral = T(-1.0);
    assert(cuda::std::isnan(cuda::std::modff(cuda::std::numeric_limits<T>::quiet_NaN(), &integral)));
    assert(cuda::std::isnan(integral));

    integral = T(-1.0);
    assert(cuda::std::isnan(cuda::std::modff(cuda::std::numeric_limits<T>::signaling_NaN(), &integral)));
    assert(cuda::std::isnan(integral));

    // some random value
    integral         = cuda::std::numeric_limits<T>::infinity();
    const T x        = T(12.3456789);
    const T expected = x - T(12.0);
    assert(is_about(cuda::std::modff(x, &integral), expected));
    assert(eq(integral, T(12.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    // If num is ±0, ±0 is returned, and ±0 is stored in *iptr
    T integral = cuda::std::numeric_limits<T>::infinity();
    assert(eq(cuda::std::modfl(val, &integral), val));
    assert(eq(integral, val));

    integral = cuda::std::numeric_limits<T>::infinity();
    assert(eq(cuda::std::modfl(-val, &integral), -val));
    assert(eq(integral, -val));

    // If num is ±∞, ±0 is returned, and ±∞ is stored in *iptr.
    integral = T(-1.0);
    assert(eq(cuda::std::modfl(cuda::std::numeric_limits<T>::infinity(), &integral), val));
    assert(eq(integral, cuda::std::numeric_limits<T>::infinity()));

    // If num is NaN, NaN is returned, and NaN is stored in *iptr.
    integral = T(-1.0);
    assert(cuda::std::isnan(cuda::std::modfl(cuda::std::numeric_limits<T>::quiet_NaN(), &integral)));
    assert(cuda::std::isnan(integral));

    integral = T(-1.0);
    assert(cuda::std::isnan(cuda::std::modfl(cuda::std::numeric_limits<T>::signaling_NaN(), &integral)));
    assert(cuda::std::isnan(integral));

    // some random value
    integral         = cuda::std::numeric_limits<T>::infinity();
    const T x        = T(12.3456789);
    const T expected = x - T(12.0);
    assert(is_about(cuda::std::modfl(x, &integral), expected));
    assert(eq(integral, T(12.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test(const T val)
{
  test_fmod<T>(val);
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    test_modf<T>(val);
  }
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
