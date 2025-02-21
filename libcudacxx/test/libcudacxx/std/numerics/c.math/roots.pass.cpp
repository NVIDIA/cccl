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

#include "fp_compare.h"
#include "test_macros.h"

#if defined(TEST_COMPILER_MSVC)
#  pragma warning(disable : 4244) // conversion from 'double' to 'float', possible loss of data
#  pragma warning(disable : 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif // TEST_COMPILER_MSVC

template <typename T>
__host__ __device__ bool eq(T lhs, T rhs) noexcept
{
  return lhs == rhs;
}

template <typename T, typename U, cuda::std::enable_if_t<cuda::std::is_arithmetic<U>::value, int> = 0>
__host__ __device__ bool eq(T lhs, U rhs) noexcept
{
  return eq(lhs, T(rhs));
}

#ifdef _LIBCUDACXX_HAS_NVFP16
__host__ __device__ bool eq(__half lhs, __half rhs) noexcept
{
  return ::__heq(lhs, rhs);
}
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
__host__ __device__ bool eq(__nv_bfloat16 lhs, __nv_bfloat16 rhs) noexcept
{
  return ::__heq(lhs, rhs);
}
#endif // _LIBCUDACXX_HAS_NVBF16

template <typename T>
__host__ __device__ void test_sqrt(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral<T>::value, double, T>;
  static_assert(cuda::std::is_same<decltype(cuda::std::sqrt(T{})), ret>::value, "");

  assert(eq(cuda::std::sqrt(val), T(8.0)));
  assert(eq(cuda::std::sqrt(T(0.0)), T(0.0)));
  assert(eq(cuda::std::sqrt(T(cuda::std::numeric_limits<T>::infinity())), cuda::std::numeric_limits<T>::infinity()));
  if (cuda::std::is_same<T, float>::value)
  {
    assert(eq(cuda::std::sqrtf(val), T(8.0)));
    assert(eq(cuda::std::sqrtf(T(0.0)), T(0.0)));
  }
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  else if (cuda::std::is_same<T, long double>::value)
  {
    assert(eq(cuda::std::sqrtl(val), T(8)));
    assert(eq(cuda::std::sqrtl(T(0.0)), T(0.0)));
  }
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
}

template <typename T>
__host__ __device__ void test_cbrt(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral<T>::value, double, T>;
  static_assert(cuda::std::is_same<decltype(cuda::std::cbrt(T{})), ret>::value, "");

  assert(eq(cuda::std::cbrt(val), T(2)));
  assert(eq(cuda::std::cbrt(T(0.0)), T(0.0)));
  assert(eq(cuda::std::cbrt(-T(0.0)), -T(0.0)));
  assert(eq(cuda::std::cbrt(T(cuda::std::numeric_limits<T>::infinity())), cuda::std::numeric_limits<T>::infinity()));
  if (cuda::std::is_same<T, float>::value)
  {
    assert(eq(cuda::std::cbrtf(val), T(2)));
    assert(eq(cuda::std::cbrtf(T(0.0)), T(0.0)));
  }
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  else if (cuda::std::is_same<T, long double>::value)
  {
    assert(eq(cuda::std::cbrtl(val), T(2)));
    assert(eq(cuda::std::cbrtl(T(0.0)), T(0.0)));
  }
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
}

template <typename T>
__host__ __device__ void test(const T val)
{
  test_sqrt<T>(val);
  test_cbrt<T>(val / T(8));
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
  volatile float val = 64.f;
  test(val);
  return 0;
}
