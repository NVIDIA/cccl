//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

// UNSUPPORTED: force-tile
// error calling a __host__ __device__ function from a __host__ __device__ __tile__ function is not allowed

// <cmath>

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "comparison.h"
#include "test_macros.h"

#if _CCCL_HOSTED()
#  include <fenv.h>
#endif // _CCCL_HOSTED()

TEST_DIAG_SUPPRESS_MSVC(4244) // conversion from 'double' to 'float', possible loss of data
TEST_DIAG_SUPPRESS_MSVC(4305) // 'argument': truncation from 'T' to 'float'
TEST_DIAG_SUPPRESS_MSVC(4146) // unary minus operator applied to unsigned type, result still unsigned

template <typename T>
TEST_HOST_DEVICE_FUNC void test_exp(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::exp(T{})), ret>);

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
TEST_HOST_DEVICE_FUNC void test_exp2(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::exp2(T{})), ret>);

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
TEST_HOST_DEVICE_FUNC void test_expm1(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::expm1(T{})), ret>);

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
TEST_HOST_DEVICE_FUNC void test_frexp(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::frexp(T{}, nullptr)), ret>);

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
TEST_HOST_DEVICE_FUNC void test_ldexp(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ldexp(T{}, int{})), ret>);

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
TEST_HOST_DEVICE_FUNC void test_scalbln(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::scalbln(T{}, long{})), ret>);

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
TEST_HOST_DEVICE_FUNC void test_scalbn(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::scalbn(T{}, int{})), ret>);

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
TEST_HOST_DEVICE_FUNC void test_pow(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::pow(T{}, T{})), ret>);

  // Fast float pow uses exp2(y * log2(x)), which does not preserve all IEEE special cases.
  // Small floating-point types also use the float implementation; integral overloads use double.
  bool test_special_values = true;
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (test_special_values = !TEST_HAS_CUDA_FAST_MATH || cuda::std::is_integral_v<T> || sizeof(T) > sizeof(float);))

  assert(eq(cuda::std::pow(T(2.0), T(10.0)), T(1024.0)));
  if (test_special_values)
  {
    assert(eq(cuda::std::pow(val, cuda::std::numeric_limits<T>::infinity()), val));
    assert(eq(cuda::std::pow(-val, cuda::std::numeric_limits<T>::infinity()), val));
  }
  assert(eq(cuda::std::pow(T(0.0), val), T(0.0)));
  assert(eq(cuda::std::pow(T(-0.0), val), T(-0.0)));
  if constexpr (!cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::pow(T(2), T(-3)), T(0.125)));

    if (test_special_values)
    {
      // Returns always 1 even for NaN
      assert(eq(cuda::std::pow(val, cuda::std::numeric_limits<T>::quiet_NaN()), val));
    }

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

    if (test_special_values)
    {
      // Returns always 1 even for NaN
      assert(eq(cuda::std::powf(val, cuda::std::numeric_limits<T>::quiet_NaN()), val));
    }

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
TEST_HOST_DEVICE_FUNC void test(const T val)
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

#if _CCCL_HAS_FLOAT128()
TEST_HOST_DEVICE_FUNC void test_float128_ldexp_edge_cases(float val)
{
  using T = __float128;

  constexpr int int_min = cuda::std::numeric_limits<int>::min();
  constexpr int int_max = cuda::std::numeric_limits<int>::max();

  // val is only known at run time, which keeps the calls from being evaluated by the compiler
  T one           = T(val);
  T min           = one * cuda::std::numeric_limits<T>::min(); // 2^-16382
  T max           = one * cuda::std::numeric_limits<T>::max(); // (2 - 2^-112) * 2^16383
  T dmin          = one * cuda::std::numeric_limits<T>::denorm_min(); // 2^-16494
  T eps           = one * cuda::std::numeric_limits<T>::epsilon(); // 2^-112
  T inf           = one * cuda::std::numeric_limits<T>::infinity();
  T nan           = one * cuda::std::numeric_limits<T>::quiet_NaN();
  T two_pow_112   = one * T(1ull << 56) * T(1ull << 56);
  T two_pow_16383 = T(2) / min;

  // positive and negative values and exponents
  assert(eq(cuda::std::ldexp(T(3) * one, 4), T(48)));
  assert(eq(cuda::std::ldexp(T(-3) * one, 4), T(-48)));
  assert(eq(cuda::std::ldexp(T(3) * one, -2), T(0.75)));
  assert(eq(cuda::std::ldexp(T(-3) * one, -2), T(-0.75)));
  assert(eq(cuda::std::ldexp(max, 0), max));
  assert(eq(cuda::std::ldexp(-dmin, 0), -dmin));

  // binary128 keeps mantissa bits well beyond the double precision
  assert(eq(cuda::std::ldexp(one + eps, 112), two_pow_112 + T(1)));
  assert(eq(cuda::std::ldexp(two_pow_112 + T(1), -112), one + eps));
  assert(eq(cuda::std::ldexp(max, -16383), T(2) - eps));

  // zero keeps its sign for every exponent
  assert(eq(cuda::std::ldexp(T(0.0) * one, int_max), T(0.0)));
  assert(!cuda::std::signbit(cuda::std::ldexp(T(0.0) * one, int_max)));
  assert(!cuda::std::signbit(cuda::std::ldexp(T(0.0) * one, int_min)));
  assert(eq(cuda::std::ldexp(T(-0.0) * one, int_max), T(0.0)));
  assert(cuda::std::signbit(cuda::std::ldexp(T(-0.0) * one, int_max)));
  assert(cuda::std::signbit(cuda::std::ldexp(T(-0.0) * one, int_min)));
  assert(cuda::std::signbit(cuda::std::ldexp(T(-0.0) * one, 1)));

  // infinity and NaN are returned for every exponent
  assert(eq(cuda::std::ldexp(inf, int_min), inf));
  assert(eq(cuda::std::ldexp(inf, int_max), inf));
  assert(eq(cuda::std::ldexp(-inf, int_min), -inf));
  assert(eq(cuda::std::ldexp(-inf, int_max), -inf));
  assert(cuda::std::isnan(cuda::std::ldexp(nan, 0)));
  assert(cuda::std::isnan(cuda::std::ldexp(nan, int_min)));
  assert(cuda::std::isnan(cuda::std::ldexp(nan, int_max)));
  assert(cuda::std::isnan(cuda::std::ldexp(-nan, 1)));

  // overflow
  assert(eq(cuda::std::ldexp(one, 16383), two_pow_16383));
  assert(eq(cuda::std::ldexp(one, 16384), inf));
  assert(eq(cuda::std::ldexp(-one, 16384), -inf));
  assert(eq(cuda::std::ldexp(max / T(2), 1), max));
  assert(eq(cuda::std::ldexp(max, 1), inf));
  assert(eq(cuda::std::ldexp(-max, 1), -inf));

  // subnormals scale up to normal values exactly
  assert(eq(cuda::std::ldexp(dmin, 112), min));
  assert(eq(cuda::std::ldexp(dmin, 16494), one));
  assert(eq(cuda::std::ldexp(dmin * T(5), 16494), T(5)));
  assert(eq(cuda::std::ldexp(-dmin, 16494 + 16383), -two_pow_16383));
  assert(eq(cuda::std::ldexp(dmin, 16494 + 16384), inf));
  assert(eq(cuda::std::ldexp(min - dmin, 1), (min - dmin) * T(2)));

  // the boundary between normal and subnormal values
  assert(eq(cuda::std::ldexp(one, -16382), min));
  assert(eq(cuda::std::ldexp(one, -16383), min / T(2)));
  assert(eq(cuda::std::ldexp(min, -1), min / T(2)));
  assert(eq(cuda::std::ldexp(min / T(2), 1), min));
  assert(eq(cuda::std::ldexp(-min, -112), -dmin));
  assert(eq(cuda::std::ldexp(one + eps, -16382), min + dmin));
  assert(eq(cuda::std::ldexp(two_pow_16383, -16383 - 16382), min));
  assert(eq(cuda::std::ldexp(two_pow_16383, -16383 - 16494), dmin));

  // subnormal results are rounded to nearest, ties to even
  assert(eq(cuda::std::ldexp(one + eps, -16383), min / T(2)));
  assert(eq(cuda::std::ldexp(one + eps * T(2), -16383), min / T(2) + dmin));
  assert(eq(cuda::std::ldexp(one + eps * T(3), -16383), min / T(2) + dmin * T(2)));
  assert(eq(cuda::std::ldexp(T(1.5) * one, -16495), dmin));
  assert(eq(cuda::std::ldexp(T(3) * one, -16495), dmin * T(2)));
  assert(eq(cuda::std::ldexp(T(-5) * one, -16495), -dmin * T(2)));

  // underflow
  assert(eq(cuda::std::ldexp(one, -16494), dmin));
  assert(eq(cuda::std::ldexp(one, -16495), T(0.0)));
  assert(!cuda::std::signbit(cuda::std::ldexp(one, -16495)));
  assert(eq(cuda::std::ldexp(-one, -16495), T(0.0)));
  assert(cuda::std::signbit(cuda::std::ldexp(-one, -16495)));
  // just above half of the smallest subnormal, which rounding in two steps would turn into zero
  assert(eq(cuda::std::ldexp(one + eps, -16495), dmin));
  assert(eq(cuda::std::ldexp(max, -16383 - 16494), dmin * T(2)));
  assert(eq(cuda::std::ldexp(max, -16383 - 16495), dmin));
  assert(eq(cuda::std::ldexp(-max, -16383 - 16495), -dmin));
  assert(eq(cuda::std::ldexp(max, -16383 - 16496), T(0.0)));
  assert(cuda::std::signbit(cuda::std::ldexp(-max, -16383 - 16496)));

  // extreme exponents
  assert(eq(cuda::std::ldexp(one, int_max), inf));
  assert(eq(cuda::std::ldexp(-one, int_max), -inf));
  assert(eq(cuda::std::ldexp(dmin, int_max), inf));
  assert(eq(cuda::std::ldexp(-dmin, int_max), -inf));
  assert(eq(cuda::std::ldexp(dmin, int_max - 1), inf));
  assert(eq(cuda::std::ldexp(one, int_min), T(0.0)));
  assert(eq(cuda::std::ldexp(max, int_min), T(0.0)));
  assert(!cuda::std::signbit(cuda::std::ldexp(max, int_min)));
  assert(cuda::std::signbit(cuda::std::ldexp(-max, int_min)));
  assert(eq(cuda::std::ldexp(max, int_min + 1), T(0.0)));
}

#  if _CCCL_HOSTED()
// The device has neither a floating point environment nor floating point exceptions
TEST_HOST_DEVICE_FUNC void test_float128_ldexp_fenv(float val)
{
  unused(val);
  NV_IF_TARGET(NV_IS_HOST, ({
                 using T = __float128;

                 constexpr int int_min    = cuda::std::numeric_limits<int>::min();
                 constexpr int int_max    = cuda::std::numeric_limits<int>::max();
                 constexpr int exceptions = FE_INEXACT | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW;

                 // volatile keeps the calls from being moved across the changes of the rounding mode
                 volatile T one  = T(val);
                 volatile T max  = cuda::std::numeric_limits<T>::max();
                 volatile T dmin = cuda::std::numeric_limits<T>::denorm_min();
                 volatile T inf  = cuda::std::numeric_limits<T>::infinity();
                 volatile T nan  = cuda::std::numeric_limits<T>::quiet_NaN();
                 volatile T res  = T(0.0);

                 // exact results raise no exceptions
                 feclearexcept(FE_ALL_EXCEPT);
                 res = cuda::std::ldexp(T(3) * one, 4);
                 assert(eq(T(res), T(48)));
                 res = cuda::std::ldexp(one, -16494);
                 assert(eq(T(res), T(dmin)));
                 res = cuda::std::ldexp(dmin, 16494);
                 assert(eq(T(res), T(1)));
                 res = cuda::std::ldexp(T(0.0) * one, int_max);
                 assert(eq(T(res), T(0.0)));
                 res = cuda::std::ldexp(inf, int_min);
                 assert(eq(T(res), T(inf)));
                 res = cuda::std::ldexp(nan, 1);
                 assert(cuda::std::isnan(T(res)));
                 assert(!fetestexcept(exceptions));

                 // overflow
                 res = cuda::std::ldexp(max, 1);
                 assert(eq(T(res), T(inf)));
                 assert(fetestexcept(exceptions) == (FE_INEXACT | FE_OVERFLOW));
                 feclearexcept(FE_ALL_EXCEPT);

                 // underflow
                 res = cuda::std::ldexp(one, -16495);
                 assert(eq(T(res), T(0.0)));
                 assert(fetestexcept(exceptions) == (FE_INEXACT | FE_UNDERFLOW));
                 feclearexcept(FE_ALL_EXCEPT);

                 // results that overflow or underflow are rounded in the current rounding mode
                 assert(fesetround(FE_UPWARD) == 0);
                 res = cuda::std::ldexp(one, -16495);
                 assert(eq(T(res), T(dmin)));
                 res = cuda::std::ldexp(-one, -16495);
                 assert(eq(T(res), T(0.0)) && cuda::std::signbit(T(res)));
                 res = cuda::std::ldexp(max, int_min);
                 assert(eq(T(res), T(dmin)));
                 res = cuda::std::ldexp(max, 1);
                 assert(eq(T(res), T(inf)));
                 res = cuda::std::ldexp(-max, 1);
                 assert(eq(T(res), -T(max)));

                 assert(fesetround(FE_DOWNWARD) == 0);
                 res = cuda::std::ldexp(one, -16495);
                 assert(eq(T(res), T(0.0)) && !cuda::std::signbit(T(res)));
                 res = cuda::std::ldexp(-one, -16495);
                 assert(eq(T(res), -T(dmin)));
                 res = cuda::std::ldexp(max, 1);
                 assert(eq(T(res), T(max)));
                 res = cuda::std::ldexp(-max, 1);
                 assert(eq(T(res), -T(inf)));

                 assert(fesetround(FE_TOWARDZERO) == 0);
                 res = cuda::std::ldexp(T(1.5) * one, -16495);
                 assert(eq(T(res), T(0.0)));
                 res = cuda::std::ldexp(T(-3) * one, -16495);
                 assert(eq(T(res), -T(dmin)));
                 res = cuda::std::ldexp(max, int_max);
                 assert(eq(T(res), T(max)));
                 res = cuda::std::ldexp(-dmin, int_max);
                 assert(eq(T(res), -T(max)));

                 assert(fesetround(FE_TONEAREST) == 0);
                 feclearexcept(FE_ALL_EXCEPT);
               }))
}
#  endif // _CCCL_HOSTED()

// Only ldexp supports __float128 so far, so the generic driver can not be used
TEST_HOST_DEVICE_FUNC void test_float128(const float val)
{
  test_ldexp<__float128>(val);
  test_float128_ldexp_edge_cases(val);
#  if _CCCL_HOSTED()
  test_float128_ldexp_fenv(val);
#  endif // _CCCL_HOSTED()
}
#endif // _CCCL_HAS_FLOAT128()

TEST_HOST_DEVICE_FUNC void test(const float val)
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
#if _CCCL_HAS_FLOAT128()
  test_float128(val);
#endif // _CCCL_HAS_FLOAT128()

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
