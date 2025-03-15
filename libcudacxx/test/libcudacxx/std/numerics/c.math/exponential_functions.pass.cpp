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
__host__ __device__ void test_exp(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::exp(T{})), ret>, "");

  [[maybe_unused]] const T euler = T(2.718281828459045);
  assert(eq(cuda::std::exp(T(-0.0)), T(1.0)));
  unused(val);
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::exp(val), euler));
    assert(eq(cuda::std::exp(T(-cuda::std::numeric_limits<T>::infinity())), T(0.0)));
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::expf(val), euler));
    assert(eq(cuda::std::expf(T(-0.0)), T(1.0)));
    assert(eq(cuda::std::expf(T(-cuda::std::numeric_limits<T>::infinity())), T(0.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if (cuda::std::is_same<T, long double>::value)
  {
    assert(eq(cuda::std::expl(val), euler));
    assert(eq(cuda::std::expl(T(-0.0)), T(1.0)));
    assert(eq(cuda::std::expl(T(-cuda::std::numeric_limits<T>::infinity())), T(0.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_exp2(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::exp2(T{})), ret>, "");

  assert(eq(cuda::std::exp2(val), T(2.0)));
  assert(eq(cuda::std::exp2(val * T(4)), T(16.0)));
  assert(eq(cuda::std::exp2(T(-0.0)), T(1.0)));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::exp2(val * T(-4)), T(0.0625)));
    assert(eq(cuda::std::exp2(T(-cuda::std::numeric_limits<T>::infinity())), T(0.0)));
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::exp2f(val), T(2.0)));
    assert(eq(cuda::std::exp2f(val * T(4)), T(16.0)));
    assert(eq(cuda::std::exp2f(val * T(-4)), T(0.0625)));
    assert(eq(cuda::std::exp2f(T(-0.0)), T(1.0)));
    assert(eq(cuda::std::exp2f(T(-cuda::std::numeric_limits<T>::infinity())), T(0.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::exp2l(val), T(2.0)));
    assert(eq(cuda::std::exp2l(val * T(4)), T(16.0)));
    assert(eq(cuda::std::exp2l(val * T(-4)), T(0.0625)));
    assert(eq(cuda::std::exp2l(T(-0.0)), T(1.0)));
    assert(eq(cuda::std::exp2l(T(-cuda::std::numeric_limits<T>::infinity())), T(0.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_expm1(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::expm1(T{})), ret>, "");

  const T eulerm1 = T(1.718281828459045);
  assert(is_about(T(cuda::std::expm1(val)), eulerm1));
  assert(eq(cuda::std::expm1(T(-0.0)), T(-0.0)));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::expm1(800), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::expm1(T(-cuda::std::numeric_limits<T>::infinity())), T(-1)));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(is_about(T(cuda::std::expm1f(val)), eulerm1));
    assert(eq(cuda::std::expm1f(T(-0.0)), T(-0.0)));
    assert(eq(cuda::std::expm1f(800), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::expm1f(T(-cuda::std::numeric_limits<T>::infinity())), T(-1)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(is_about(T(cuda::std::expm1l(val)), eulerm1));
    assert(eq(cuda::std::expm1l(T(-0.0)), T(-0.0)));
    assert(eq(cuda::std::expm1l(800), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::expm1l(T(-cuda::std::numeric_limits<T>::infinity())), T(-1)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_frexp(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::frexp(T{}, nullptr)), ret>, "");

  int exponent = -1;
  assert(eq(cuda::std::frexp(T(0.0), &exponent), T(0.0)));
  assert(exponent == 0);
  exponent = -1;
  assert(eq(cuda::std::frexp(T(-0.0), &exponent), T(0.0)));
  assert(exponent == 0);
  unused(val);
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    exponent = -1;
    assert(eq(cuda::std::frexp(val, &exponent), T(0.5)));
    assert(exponent == 1);
    // exponent is undefined here
    assert(eq(cuda::std::frexp(T(cuda::std::numeric_limits<T>::infinity()), &exponent),
              T(cuda::std::numeric_limits<T>::infinity())));
    assert(eq(cuda::std::frexp(T(-cuda::std::numeric_limits<T>::infinity()), &exponent),
              T(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::frexp(T(cuda::std::numeric_limits<T>::quiet_NaN()), &exponent)));
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    exponent = -1;
    assert(eq(cuda::std::frexpf(val, &exponent), T(0.5)));
    assert(exponent == 1);

    exponent = -1;
    assert(eq(cuda::std::frexpf(T(0.0), &exponent), T(0.0)));
    assert(exponent == 0);
    exponent = -1;
    assert(eq(cuda::std::frexpf(T(-0.0), &exponent), T(0.0)));
    assert(exponent == 0);

    // exponent is undefined here
    assert(eq(cuda::std::frexpf(T(cuda::std::numeric_limits<T>::infinity()), &exponent),
              T(cuda::std::numeric_limits<T>::infinity())));
    assert(eq(cuda::std::frexpf(T(-cuda::std::numeric_limits<T>::infinity()), &exponent),
              T(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::frexpf(T(cuda::std::numeric_limits<T>::quiet_NaN()), &exponent)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    exponent = -1;
    assert(eq(cuda::std::frexpl(val, &exponent), T(0.5)));
    assert(exponent == 1);
    exponent = -1;
    assert(eq(cuda::std::frexpl(T(0.0), &exponent), T(0.0)));
    assert(exponent == 0);
    exponent = -1;
    assert(eq(cuda::std::frexpl(T(-0.0), &exponent), T(0.0)));
    assert(exponent == 0);

    // exponent is undefined here
    assert(eq(cuda::std::frexpl(T(cuda::std::numeric_limits<T>::infinity()), &exponent),
              T(cuda::std::numeric_limits<T>::infinity())));
    assert(eq(cuda::std::frexpl(T(-cuda::std::numeric_limits<T>::infinity()), &exponent),
              T(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::frexpl(T(cuda::std::numeric_limits<T>::quiet_NaN()), &exponent)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_ldexp(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ldexp(T{}, int{})), ret>, "");

  assert(eq(cuda::std::ldexp(T(0.0), 800), T(0.0)));
  assert(eq(cuda::std::ldexp(T(-0.0), 800), T(0.0)));
  assert(eq(cuda::std::ldexp(val, 5), T(32.0)));
  assert(eq(cuda::std::ldexp(val, 0), val));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::ldexp(T(cuda::std::numeric_limits<T>::infinity()), 1),
              T(cuda::std::numeric_limits<T>::infinity())));
    assert(eq(cuda::std::ldexp(T(-cuda::std::numeric_limits<T>::infinity()), 1),
              T(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::ldexp(T(cuda::std::numeric_limits<T>::quiet_NaN()), 1)));
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::ldexpf(T(0.0), 800), T(0.0)));
    assert(eq(cuda::std::ldexpf(T(-0.0), 800), T(0.0)));
    assert(eq(cuda::std::ldexpf(val, 5), T(32.0)));
    assert(eq(cuda::std::ldexpf(val, 0), val));
    assert(eq(cuda::std::ldexpf(T(cuda::std::numeric_limits<T>::infinity()), 1),
              T(cuda::std::numeric_limits<T>::infinity())));
    assert(eq(cuda::std::ldexpf(T(-cuda::std::numeric_limits<T>::infinity()), 1),
              T(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::ldexpf(T(cuda::std::numeric_limits<T>::quiet_NaN()), 1)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::ldexpl(T(0.0), 800), T(0.0)));
    assert(eq(cuda::std::ldexpl(T(-0.0), 800), T(0.0)));
    assert(eq(cuda::std::ldexpl(val, 5), T(32.0)));
    assert(eq(cuda::std::ldexpl(val, 0), val));
    assert(eq(cuda::std::ldexpl(T(cuda::std::numeric_limits<T>::infinity()), 1),
              T(cuda::std::numeric_limits<T>::infinity())));
    assert(eq(cuda::std::ldexpl(T(-cuda::std::numeric_limits<T>::infinity()), 1),
              T(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::ldexpl(T(cuda::std::numeric_limits<T>::quiet_NaN()), 1)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_scalbln(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::scalbln(T{}, long{})), ret>, "");

  assert(eq(cuda::std::scalbln(T(0.0), 800), T(0.0)));
  assert(eq(cuda::std::scalbln(T(-0.0), 800), T(0.0)));
  assert(eq(cuda::std::scalbln(val, 5), T(32.0)));
  assert(eq(cuda::std::scalbln(val, 0), val));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::scalbln(T(cuda::std::numeric_limits<T>::infinity()), 1),
              T(cuda::std::numeric_limits<T>::infinity())));
    assert(eq(cuda::std::scalbln(T(-cuda::std::numeric_limits<T>::infinity()), 1),
              T(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::scalbln(T(cuda::std::numeric_limits<T>::quiet_NaN()), 1)));
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::scalblnf(T(0.0), 800), T(0.0)));
    assert(eq(cuda::std::scalblnf(T(-0.0), 800), T(0.0)));
    assert(eq(cuda::std::scalblnf(val, 5), T(32.0)));
    assert(eq(cuda::std::scalblnf(val, 0), val));
    assert(eq(cuda::std::scalblnf(T(cuda::std::numeric_limits<T>::infinity()), 1),
              T(cuda::std::numeric_limits<T>::infinity())));
    assert(eq(cuda::std::scalblnf(T(-cuda::std::numeric_limits<T>::infinity()), 1),
              T(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::scalblnf(T(cuda::std::numeric_limits<T>::quiet_NaN()), 1)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::scalblnl(T(0.0), 800), T(0.0)));
    assert(eq(cuda::std::scalblnl(T(-0.0), 800), T(0.0)));
    assert(eq(cuda::std::scalblnl(val, 5), T(32.0)));
    assert(eq(cuda::std::scalblnl(val, 0), val));
    assert(eq(cuda::std::scalblnl(T(cuda::std::numeric_limits<T>::infinity()), 1),
              T(cuda::std::numeric_limits<T>::infinity())));
    assert(eq(cuda::std::scalblnl(T(-cuda::std::numeric_limits<T>::infinity()), 1),
              T(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::scalblnl(T(cuda::std::numeric_limits<T>::quiet_NaN()), 1)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_scalbn(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::scalbn(T{}, int{})), ret>, "");

  assert(eq(cuda::std::scalbn(T(0.0), 800), T(0.0)));
  assert(eq(cuda::std::scalbn(T(-0.0), 800), T(0.0)));
  assert(eq(cuda::std::scalbn(val, 5), T(32.0)));
  assert(eq(cuda::std::scalbn(val, 0), val));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::scalbn(T(cuda::std::numeric_limits<T>::infinity()), 1),
              T(cuda::std::numeric_limits<T>::infinity())));
    assert(eq(cuda::std::scalbn(T(-cuda::std::numeric_limits<T>::infinity()), 1),
              T(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::scalbn(T(cuda::std::numeric_limits<T>::quiet_NaN()), 1)));
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::scalbnf(T(0.0), 800), T(0.0)));
    assert(eq(cuda::std::scalbnf(T(-0.0), 800), T(0.0)));
    assert(eq(cuda::std::scalbnf(val, 5), T(32.0)));
    assert(eq(cuda::std::scalbnf(val, 0), val));
    assert(eq(cuda::std::scalbnf(T(cuda::std::numeric_limits<T>::infinity()), 1),
              T(cuda::std::numeric_limits<T>::infinity())));
    assert(eq(cuda::std::scalbnf(T(-cuda::std::numeric_limits<T>::infinity()), 1),
              T(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::scalbnf(T(cuda::std::numeric_limits<T>::quiet_NaN()), 1)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::scalbnl(T(0.0), 800), T(0.0)));
    assert(eq(cuda::std::scalbnl(T(-0.0), 800), T(0.0)));
    assert(eq(cuda::std::scalbnl(val, 5), T(32.0)));
    assert(eq(cuda::std::scalbnl(val, 0), val));
    assert(eq(cuda::std::scalbnl(T(cuda::std::numeric_limits<T>::infinity()), 1),
              T(cuda::std::numeric_limits<T>::infinity())));
    assert(eq(cuda::std::scalbnl(T(-cuda::std::numeric_limits<T>::infinity()), 1),
              T(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::std::scalbnl(T(cuda::std::numeric_limits<T>::quiet_NaN()), 1)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_pow(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::pow(T{}, T{})), ret>, "");

  assert(eq(cuda::std::pow(T(2.0), T(10.0)), T(1024.0)));
  assert(eq(cuda::std::pow(val, cuda::std::numeric_limits<T>::infinity()), val));
  assert(eq(cuda::std::pow(-val, cuda::std::numeric_limits<T>::infinity()), val));
  assert(eq(cuda::std::pow(T(0.0), val), T(0.0)));
  assert(eq(cuda::std::pow(T(-0.0), val), T(-0.0)));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::pow(T(2), T(-3)), T(0.125)));

    // Returns always 1 even for NaN
    assert(eq(cuda::std::pow(val, cuda::std::numeric_limits<T>::quiet_NaN()), val));

    assert(
      eq(cuda::std::pow(T(2.0), cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::pow(T(2.0), -cuda::std::numeric_limits<T>::infinity()), T(0.0)));

    assert(eq(cuda::std::pow(T(0.5), cuda::std::numeric_limits<T>::infinity()), T(0.0)));
    assert(
      eq(cuda::std::pow(T(0.5), -cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::powf(T(2), T(-3)), T(0.125)));

    // Returns always 1 even for NaN
    assert(eq(cuda::std::powf(val, cuda::std::numeric_limits<T>::quiet_NaN()), val));

    assert(
      eq(cuda::std::powf(T(2.0), cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::powf(T(2.0), -cuda::std::numeric_limits<T>::infinity()), T(0.0)));

    assert(eq(cuda::std::powf(T(0.5), cuda::std::numeric_limits<T>::infinity()), T(0.0)));
    assert(
      eq(cuda::std::powf(T(0.5), -cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::powl(T(2), T(-3)), T(0.125)));

    // Returns always 1 even for NaN
    assert(eq(cuda::std::powl(val, cuda::std::numeric_limits<T>::quiet_NaN()), val));

    assert(
      eq(cuda::std::powl(T(2.0), cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(eq(cuda::std::powl(T(2.0), -cuda::std::numeric_limits<T>::infinity()), T(0.0)));

    assert(eq(cuda::std::powl(T(0.5), cuda::std::numeric_limits<T>::infinity()), T(0.0)));
    assert(
      eq(cuda::std::powl(T(0.5), -cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test(const T val)
{
  test_exp<T>(val);
  test_exp2<T>(val);
  test_expm1<T>(val);
  test_frexp<T>(val);
  test_ldexp<T>(val);
  test_scalbln<T>(val);
  test_scalbn<T>(val);
  test_pow<T>(val);
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
  volatile float val = 1.0f;
  test(val);
  return 0;
}
