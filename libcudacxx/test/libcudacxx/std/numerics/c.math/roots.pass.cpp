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

TEST_DIAG_SUPPRESS_MSVC(4244) // conversion from 'const double' to 'int', possible loss of data
TEST_DIAG_SUPPRESS_MSVC(4146) // unary minus operator applied to unsigned type, result still unsigned

template <typename T>
__host__ __device__ void test_sqrt(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::sqrt(T{})), ret>, "");

  assert(eq(cuda::std::sqrt(val), T(8.0)));
  assert(eq(cuda::std::sqrt(T(0.0)), T(0.0)));
  assert(eq(cuda::std::sqrt(T(cuda::std::numeric_limits<T>::infinity())), cuda::std::numeric_limits<T>::infinity()));
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::sqrtf(val), T(8.0)));
    assert(eq(cuda::std::sqrtf(T(0.0)), T(0.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::sqrtl(val), T(8)));
    assert(eq(cuda::std::sqrtl(T(0.0)), T(0.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_cbrt(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::cbrt(T{})), ret>, "");

  assert(eq(cuda::std::cbrt(val), T(2)));
  assert(eq(cuda::std::cbrt(T(0.0)), T(0.0)));
  assert(eq(cuda::std::cbrt(-T(0.0)), -T(0.0)));
  assert(eq(cuda::std::cbrt(T(cuda::std::numeric_limits<T>::infinity())), cuda::std::numeric_limits<T>::infinity()));
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::cbrtf(val), T(2)));
    assert(eq(cuda::std::cbrtf(T(0.0)), T(0.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::cbrtl(val), T(2)));
    assert(eq(cuda::std::cbrtl(T(0.0)), T(0.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
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
  volatile float val = 64.f;
  test(val);
  return 0;
}
