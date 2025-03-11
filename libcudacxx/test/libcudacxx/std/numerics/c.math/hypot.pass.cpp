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
__host__ __device__ void test_hypot(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::hypot(T{}, T{})), ret>, "");

  // hypot(x, y), hypot(y, x), and hypot(x, -y) are equivalent.
  assert(eq(cuda::std::hypot(T(1.0), T(0.25)), cuda::std::hypot(T(0.25), T(1.0))));
  if constexpr (cuda::std::is_signed_v<T>)
  {
    assert(eq(cuda::std::hypot(T(1.0), T(0.25)), cuda::std::hypot(T(1.0), T(-0.25))));
  }

  // If one of the arguments is ±0, std::hypot(x, y) is equivalent to std::fabs called with the non-zero argument.
  assert(eq(cuda::std::hypot(val, T(0.25)), T(0.25)));
  assert(eq(cuda::std::hypot(-val, T(0.25)), T(0.25)));

  assert(is_about(cuda::std::hypot(T(3.0), T(4.0)), T(5.0)));
  if constexpr (cuda::std::is_signed_v<T>)
  {
    assert(eq(cuda::std::hypot(val, T(-0.25)), T(0.25)));
    assert(eq(cuda::std::hypot(-val, T(-0.25)), T(0.25)));
  }

  if constexpr (!cuda::std::is_integral_v<T>)
  {
    // If one of the arguments is ±∞, std::hypot(x, y) returns +∞ even if the other argument is NaN.
    assert(eq(cuda::std::hypot(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::quiet_NaN()),
              cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::hypot(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::quiet_NaN()),
              cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::hypot(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::infinity()),
              cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::hypot(cuda::std::numeric_limits<T>::quiet_NaN(), -cuda::std::numeric_limits<T>::infinity()),
              cuda::std::numeric_limits<T>::infinity()));

    // Otherwise, if any of the arguments is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::hypot(T(3.0), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::hypot(T(3.0), -cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::hypot(T(3.0), cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::hypot(T(3.0), -cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::hypot(cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypot(-cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypot(cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypot(-cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));

    // Some random value
    const T expected = T(2.236067977499789805051477742381394);
    assert(is_about(cuda::std::hypot(T(1.0), T(2.0)), expected));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    // If one of the arguments is ±∞, std::hypot(x, y) returns +∞ even if the other argument is NaN.
    assert(eq(cuda::std::hypotf(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::quiet_NaN()),
              cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::hypotf(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::quiet_NaN()),
              cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::hypotf(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::infinity()),
              cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::hypotf(cuda::std::numeric_limits<T>::quiet_NaN(), -cuda::std::numeric_limits<T>::infinity()),
              cuda::std::numeric_limits<T>::infinity()));

    // Otherwise, if any of the arguments is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::hypotf(T(3.0), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotf(T(3.0), -cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotf(T(3.0), cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotf(T(3.0), -cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotf(cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotf(-cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotf(cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotf(-cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));

    // Some random value
    const T expected = T(2.236067977499789805051477742381394);
    assert(is_about(cuda::std::hypotf(T(1.0), T(2.0)), expected));
    assert(is_about(cuda::std::hypotf(T(3.0), T(4.0)), T(5.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    // If one of the arguments is ±∞, std::hypot(x, y) returns +∞ even if the other argument is NaN.
    assert(eq(cuda::std::hypotl(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::quiet_NaN()),
              cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::hypotl(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::quiet_NaN()),
              cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::hypotl(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::infinity()),
              cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::hypotl(cuda::std::numeric_limits<T>::quiet_NaN(), -cuda::std::numeric_limits<T>::infinity()),
              cuda::std::numeric_limits<T>::infinity()));

    // Otherwise, if any of the arguments is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::hypotl(T(3.0), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotl(T(3.0), -cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotl(T(3.0), cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotl(T(3.0), -cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotl(cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotl(-cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotl(cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotl(-cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));

    // Some random value
    const T expected = T(2.236067977499789805051477742381394);
    assert(is_about(cuda::std::hypotl(T(1.0), T(2.0)), expected));
    assert(is_about(cuda::std::hypotl(T(3.0), T(4.0)), T(5.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_hypot3(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::hypot(T{}, T{}, T{})), ret>, "");

  // hypot(x, y), hypot(y, x), and hypot(x, -y) are equivalent.
  assert(eq(cuda::std::hypot(T(1.0), T(0.25), T(2.0)), cuda::std::hypot(T(0.25), T(1.0), T(2.0))));
  if constexpr (cuda::std::is_signed_v<T>)
  {
    assert(eq(cuda::std::hypot(T(1.0), T(0.25), T(2.0)), cuda::std::hypot(T(1.0), T(-0.25), T(2.0))));
  }

  // If one of the arguments is ±0, std::hypot(x, y) is equivalent to std::fabs called with the non-zero argument.
  assert(eq(cuda::std::hypot(val, T(0.25), T(2.0)), cuda::std::hypot(T(0.25), T(2.0))));
  assert(eq(cuda::std::hypot(-val, T(0.25), T(2.0)), cuda::std::hypot(T(0.25), T(2.0))));
  if constexpr (cuda::std::is_signed_v<T>)
  {
    assert(eq(cuda::std::hypot(val, T(-0.25), T(2.0)), cuda::std::hypot(T(0.25), T(2.0))));
    assert(eq(cuda::std::hypot(-val, T(-0.25), T(2.0)), cuda::std::hypot(T(0.25), T(2.0))));
  }

  if constexpr (!cuda::std::is_integral_v<T>)
  {
    // If any of the arguments is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::hypot(cuda::std::numeric_limits<T>::quiet_NaN(), T(0.25), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypot(-cuda::std::numeric_limits<T>::quiet_NaN(), T(0.25), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypot(cuda::std::numeric_limits<T>::signaling_NaN(), T(0.25), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypot(-cuda::std::numeric_limits<T>::signaling_NaN(), T(0.25), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypot(T(0.25), cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypot(T(0.25), -cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypot(T(0.25), cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypot(T(0.25), -cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypot(T(0.25), T(3.0), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::hypot(T(0.25), T(3.0), -cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::hypot(T(0.25), T(3.0), cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::hypot(T(0.25), T(3.0), -cuda::std::numeric_limits<T>::signaling_NaN())));

    // Some random value
    assert(is_about(cuda::std::hypot(T(2.0), T(3.0), T(6.0)), T(7.0)));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    // If any of the arguments is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::hypotf(cuda::std::numeric_limits<T>::quiet_NaN(), T(0.25), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotf(-cuda::std::numeric_limits<T>::quiet_NaN(), T(0.25), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotf(cuda::std::numeric_limits<T>::signaling_NaN(), T(0.25), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotf(-cuda::std::numeric_limits<T>::signaling_NaN(), T(0.25), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotf(T(0.25), cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotf(T(0.25), -cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotf(T(0.25), cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotf(T(0.25), -cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotf(T(0.25), T(3.0), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotf(T(0.25), T(3.0), -cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotf(T(0.25), T(3.0), cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotf(T(0.25), T(3.0), -cuda::std::numeric_limits<T>::signaling_NaN())));

    // Some random value
    assert(is_about(cuda::std::hypotf(T(2.0), T(3.0), T(6.0)), T(7.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    // If any of the arguments is NaN, NaN is returned.
    assert(cuda::std::isnan(cuda::std::hypotl(cuda::std::numeric_limits<T>::quiet_NaN(), T(0.25), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotl(-cuda::std::numeric_limits<T>::quiet_NaN(), T(0.25), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotl(cuda::std::numeric_limits<T>::signaling_NaN(), T(0.25), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotl(-cuda::std::numeric_limits<T>::signaling_NaN(), T(0.25), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotl(T(0.25), cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotl(T(0.25), -cuda::std::numeric_limits<T>::quiet_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotl(T(0.25), cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotl(T(0.25), -cuda::std::numeric_limits<T>::signaling_NaN(), T(3.0))));
    assert(cuda::std::isnan(cuda::std::hypotl(T(0.25), T(3.0), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotl(T(0.25), T(3.0), -cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotl(T(0.25), T(3.0), cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::hypotl(T(0.25), T(3.0), -cuda::std::numeric_limits<T>::signaling_NaN())));

    // Some random value
    assert(is_about(cuda::std::hypotl(T(2.0), T(3.0), T(6.0)), T(7.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test(const T val)
{
  test_hypot<T>(val);
  test_hypot3<T>(val);
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
