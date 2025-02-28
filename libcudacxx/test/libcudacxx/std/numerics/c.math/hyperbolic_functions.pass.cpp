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

#if defined(TEST_COMPILER_MSVC)
#  pragma warning(disable : 4244) // conversion from 'double' to 'float', possible loss of data
#  pragma warning(disable : 4305) // 'argument': truncation from 'T' to 'float'
#  pragma warning(disable : 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif // TEST_COMPILER_MSVC

template <typename T>
__host__ __device__ void test_cosh(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral<T>::value, double, T>;
  static_assert(cuda::std::is_same<decltype(cuda::std::cosh(T{})), ret>::value, "");

  // 0 is returned unmodified
  assert(eq(cuda::std::cosh(val), T(1.0)));
  assert(eq(cuda::std::cosh(-val), T(1.0)));
  if constexpr (!cuda::std::is_integral<T>::value)
  {
    assert(cuda::std::isinf(cuda::std::cosh(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isinf(cuda::std::cosh(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::cosh(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::cosh(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T expected = T(1.543080634815243712409937870688736);
    assert(is_about(cuda::std::cosh(T(1.0)), expected));
    assert(is_about(cuda::std::cosh(T(-1.0)), expected));
  }

  if constexpr (cuda::std::is_same<T, float>::value)
  {
    assert(cuda::std::isinf(cuda::std::coshf(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isinf(cuda::std::coshf(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::coshf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::coshf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T expected = T(1.543080634815243712409937870688736);
    assert(is_about(cuda::std::coshf(T(1.0)), expected));
    assert(is_about(cuda::std::coshf(T(-1.0)), expected));
  }
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  else if constexpr (cuda::std::is_same<T, long double>::value)
  {
    assert(cuda::std::isinf(cuda::std::coshl(cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isinf(cuda::std::coshl(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::coshl(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::coshl(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T expected = T(1.543080634815243712409937870688736);
    assert(is_about(cuda::std::coshl(T(1.0)), expected));
    assert(is_about(cuda::std::coshl(T(-1.0)), expected));
  }
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
}

template <typename T>
__host__ __device__ void test_sinh(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral<T>::value, double, T>;
  static_assert(cuda::std::is_same<decltype(cuda::std::sinh(T{})), ret>::value, "");

  // 0 is returned unmodified
  assert(eq(cuda::std::sinh(val), val));
  assert(eq(cuda::std::sinh(-val), -val));
  if constexpr (!cuda::std::is_integral<T>::value)
  {
    assert(eq(cuda::std::sinh(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::sinh(-cuda::std::numeric_limits<T>::infinity()), -cuda::std::numeric_limits<T>::infinity()));
    assert(cuda::std::isnan(cuda::std::sinh(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::sinh(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T expected = T(1.175201193643801378385660427738912);
    assert(is_about(cuda::std::sinh(T(1.0)), expected));
    assert(is_about(cuda::std::sinh(T(-1.0)), -expected));
  }

  if constexpr (cuda::std::is_same<T, float>::value)
  {
    assert(eq(cuda::std::sinhf(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::sinhf(-cuda::std::numeric_limits<T>::infinity()), -cuda::std::numeric_limits<T>::infinity()));
    assert(cuda::std::isnan(cuda::std::sinhf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::sinhf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T expected = T(1.175201193643801378385660427738912);
    assert(is_about(cuda::std::sinhf(T(1.0)), expected));
    assert(is_about(cuda::std::sinhf(T(-1.0)), -expected));
  }
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  else if constexpr (cuda::std::is_same<T, long double>::value)
  {
    assert(eq(cuda::std::sinhl(cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::sinhl(-cuda::std::numeric_limits<T>::infinity()), -cuda::std::numeric_limits<T>::infinity()));
    assert(cuda::std::isnan(cuda::std::sinhl(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::sinhl(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T expected = T(1.175201193643801378385660427738912);
    assert(is_about(cuda::std::sinhl(T(1.0)), expected));
    assert(is_about(cuda::std::sinhl(T(-1.0)), -expected));
  }
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
}

template <typename T>
__host__ __device__ void test_tanh(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral<T>::value, double, T>;
  static_assert(cuda::std::is_same<decltype(cuda::std::tanh(T{})), ret>::value, "");

  // 0 is returned unmodified
  assert(eq(cuda::std::tanh(val), val));
  assert(eq(cuda::std::tanh(-val), -val));
  if constexpr (!cuda::std::is_integral<T>::value)
  {
    assert(eq(cuda::std::tanh(cuda::std::numeric_limits<T>::infinity()), T(1.0)));
    assert(eq(cuda::std::tanh(-cuda::std::numeric_limits<T>::infinity()), T(-1.0)));
    assert(cuda::std::isnan(cuda::std::tanh(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::tanh(-cuda::std::numeric_limits<T>::quiet_NaN())));

    // We have precision issues with tanh because ... its tanh
    if constexpr (cuda::std::is_same<T, double>::value)
    {
      const T expected = T(0.7615941559557648510292438004398718);
      assert(is_about(cuda::std::tanh(T(1.0)), expected));
      assert(is_about(cuda::std::tanh(T(-1.0)), -expected));
    }
    else
    {
      const T expected = T(0.76159417629241943359375);
      assert(is_about(cuda::std::tanh(T(1.0)), expected));
      assert(is_about(cuda::std::tanh(T(-1.0)), -expected));
    }
  }

  if constexpr (cuda::std::is_same<T, float>::value)
  {
    assert(eq(cuda::std::tanhf(cuda::std::numeric_limits<T>::infinity()), T(1.0)));
    assert(eq(cuda::std::tanhf(-cuda::std::numeric_limits<T>::infinity()), T(-1.0)));
    assert(cuda::std::isnan(cuda::std::tanhf(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::tanhf(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T expected = T(0.76159417629241943359375);
    assert(is_about(cuda::std::tanhf(T(1.0)), expected));
    assert(is_about(cuda::std::tanhf(T(-1.0)), -expected));
  }
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  else if constexpr (cuda::std::is_same<T, long double>::value)
  {
    assert(eq(cuda::std::tanhl(cuda::std::numeric_limits<T>::infinity()), T(1.0)));
    assert(eq(cuda::std::tanhl(-cuda::std::numeric_limits<T>::infinity()), T(-1.0)));
    assert(cuda::std::isnan(cuda::std::tanhl(-cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::tanhl(-cuda::std::numeric_limits<T>::quiet_NaN())));

    const T expected = T(0.7615941559557648510292438004398718);
    assert(is_about(cuda::std::tanhl(T(1.0)), expected));
    assert(is_about(cuda::std::tanhl(T(-1.0)), expected));
  }
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
}

template <typename T>
__host__ __device__ void test(const T val)
{
  test_cosh<T>(val);
  test_sinh<T>(val);
  test_tanh<T>(val);
}

__host__ __device__ void test(const float val)
{
  test<float>(val);
  test<double>(val);
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test<long double>();
#endif //!_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#ifdef _LIBCUDACXX_HAS_NVFP16
  test<__half>(val);
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  test<__nv_bfloat16>(val);
#endif // _LIBCUDACXX_HAS_NVBF16

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
