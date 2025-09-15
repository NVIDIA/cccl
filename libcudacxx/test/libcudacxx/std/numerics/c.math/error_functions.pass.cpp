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
__host__ __device__ void test_erf(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::erf(T{})), ret>, "");

  // If the argument is ±0, ±0 is returned.
  assert(eq(cuda::std::erf(val), val));
  assert(eq(cuda::std::erf(-val), -val));

  if constexpr (!::cuda::std::is_integral_v<T>)
  {
    // If the argument is ±∞, ±1 is returned.
    assert(eq(cuda::std::erf(cuda::std::numeric_limits<T>::infinity()), T(1.0)));
    assert(eq(cuda::std::erf(-cuda::std::numeric_limits<T>::infinity()), -T(1.0)));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::erf(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::erf(cuda::std::numeric_limits<T>::signaling_NaN())));
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::erff(val), val));
    assert(eq(cuda::std::erff(-val), -val));

    // If the argument is ±∞, ±1 is returned.
    assert(eq(cuda::std::erff(cuda::std::numeric_limits<T>::infinity()), T(1.0)));
    assert(eq(cuda::std::erff(-cuda::std::numeric_limits<T>::infinity()), -T(1.0)));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::erff(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::erff(cuda::std::numeric_limits<T>::signaling_NaN())));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::erfl(val), val));
    assert(eq(cuda::std::erfl(-val), -val));

    // If the argument is ±∞, ±1 is returned.
    assert(eq(cuda::std::erfl(cuda::std::numeric_limits<T>::infinity()), T(1.0)));
    assert(eq(cuda::std::erfl(-cuda::std::numeric_limits<T>::infinity()), -T(1.0)));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::erfl(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::erfl(cuda::std::numeric_limits<T>::signaling_NaN())));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_erfc(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::erfc(T{})), ret>, "");

  // If the argument is ±0, ±0 is returned.
  assert(eq(cuda::std::erfc(val), 1));
  assert(eq(cuda::std::erfc(-val), 1));

  if constexpr (!::cuda::std::is_integral_v<T>)
  {
    // If the argument is +∞, 0 is returned.
    assert(eq(cuda::std::erfc(cuda::std::numeric_limits<T>::infinity()), T(0.0)));
    // If the argument is -∞, 2 is returned.
    assert(eq(cuda::std::erfc(-cuda::std::numeric_limits<T>::infinity()), T(2.0)));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::erfc(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::erfc(cuda::std::numeric_limits<T>::signaling_NaN())));
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    // If the argument is ±0, ±0 is returned.
    assert(eq(cuda::std::erfcf(val), 1));
    assert(eq(cuda::std::erfcf(-val), 1));

    // If the argument is +∞, 0 is returned.
    assert(eq(cuda::std::erfcf(cuda::std::numeric_limits<T>::infinity()), T(0.0)));
    // If the argument is -∞, 2 is returned.
    assert(eq(cuda::std::erfcf(-cuda::std::numeric_limits<T>::infinity()), T(2.0)));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::erfcf(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::erfcf(cuda::std::numeric_limits<T>::signaling_NaN())));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    // If the argument is ±0, ±0 is returned.
    assert(eq(cuda::std::erfcl(val), 1));
    assert(eq(cuda::std::erfcl(-val), 1));

    // If the argument is +∞, 0 is returned.
    assert(eq(cuda::std::erfcl(cuda::std::numeric_limits<T>::infinity()), T(0.0)));
    // If the argument is -∞, 2 is returned.
    assert(eq(cuda::std::erfcl(-cuda::std::numeric_limits<T>::infinity()), T(2.0)));

    // If the argument is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::erfcl(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::erfcl(cuda::std::numeric_limits<T>::signaling_NaN())));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test(const T val)
{
  test_erf<T>(val);
  test_erfc<T>(val);
}

__host__ __device__ void test(const float val)
{
  test<float>(val);
  test<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>(::__float2half(val));
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>(__float2bfloat16(val));
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
