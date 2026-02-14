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
__host__ __device__ void test_remainder(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::remainder(T{}, T{})), ret>, "");

  const T x = T(13.23456789);

  // The result has the same sign as x, if the returned value is zero
  assert(eq(cuda::std::remainder(val, x), val));
  assert(eq(cuda::std::remainder(-val, x), -val));

  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::remainder(x, T(3.0)), T(1.0)));
  }
  else
  {
    // If x is ±0 and y is not zero, ±0 is returned.
    assert(eq(cuda::std::remainder(val, x), val));
    assert(eq(cuda::std::remainder(val, -x), val));

    assert(eq(cuda::std::remainder(val, cuda::std::numeric_limits<T>::infinity()), val));

    // If x is ±∞ and y is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remainder(cuda::std::numeric_limits<T>::infinity(), val)));
    assert(cuda::std::isnan(cuda::std::remainder(-cuda::std::numeric_limits<T>::infinity(), val)));
    assert(cuda::std::isnan(cuda::std::remainder(cuda::std::numeric_limits<T>::infinity(), x)));
    assert(cuda::std::isnan(cuda::std::remainder(-cuda::std::numeric_limits<T>::infinity(), x)));
    assert(cuda::std::isnan(
      cuda::std::remainder(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::remainder(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remainder(x, val)));
    assert(cuda::std::isnan(cuda::std::remainder(-x, val)));

    // If y is ±∞ and x is finite, x is returned.
    assert(eq(cuda::std::remainder(x, cuda::std::numeric_limits<T>::infinity()), x));
    assert(eq(cuda::std::remainder(x, -cuda::std::numeric_limits<T>::infinity()), x));
    assert(eq(cuda::std::remainder(-x, cuda::std::numeric_limits<T>::infinity()), -x));
    assert(eq(cuda::std::remainder(-x, -cuda::std::numeric_limits<T>::infinity()), -x));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remainder(x, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::remainder(-x, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::remainder(cuda::std::numeric_limits<T>::quiet_NaN(), x)));
    assert(cuda::std::isnan(cuda::std::remainder(cuda::std::numeric_limits<T>::quiet_NaN(), -x)));
    assert(cuda::std::isnan(cuda::std::remainder(x, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::remainder(-x, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::remainder(cuda::std::numeric_limits<T>::signaling_NaN(), x)));
    assert(cuda::std::isnan(cuda::std::remainder(cuda::std::numeric_limits<T>::signaling_NaN(), -x)));
    assert(cuda::std::isnan(
      cuda::std::remainder(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::remainder(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value, where the remainder is negative
    {
      const T y        = T(3.456789123);
      const T quotient = T(4.0);
      assert(is_about(cuda::std::remainder(x, y), (x - quotient * y)));
    }

    // Some random value, where the remainder is positive
    {
      const T y        = T(5.678912345);
      const T quotient = T(2.0);
      assert(is_about(cuda::std::remainder(x, y), (x - quotient * y)));
    }
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    // If x is ±0 and y is not zero, ±0 is returned.
    assert(eq(cuda::std::remainderf(val, x), val));
    assert(eq(cuda::std::remainderf(-val, x), -val));
    assert(eq(cuda::std::remainderf(val, -x), val));
    assert(eq(cuda::std::remainderf(-val, -x), -val));

    assert(eq(cuda::std::remainderf(val, cuda::std::numeric_limits<T>::infinity()), val));
    assert(eq(cuda::std::remainderf(-val, cuda::std::numeric_limits<T>::infinity()), -val));

    // If x is ±∞ and y is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remainderf(cuda::std::numeric_limits<T>::infinity(), val)));
    assert(cuda::std::isnan(cuda::std::remainderf(-cuda::std::numeric_limits<T>::infinity(), val)));
    assert(cuda::std::isnan(cuda::std::remainderf(cuda::std::numeric_limits<T>::infinity(), x)));
    assert(cuda::std::isnan(cuda::std::remainderf(-cuda::std::numeric_limits<T>::infinity(), x)));
    assert(cuda::std::isnan(
      cuda::std::remainderf(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::remainderf(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remainderf(x, val)));
    assert(cuda::std::isnan(cuda::std::remainderf(-x, val)));

    // If y is ±∞ and x is finite, x is returned.
    assert(eq(cuda::std::remainderf(x, cuda::std::numeric_limits<T>::infinity()), x));
    assert(eq(cuda::std::remainderf(x, -cuda::std::numeric_limits<T>::infinity()), x));
    assert(eq(cuda::std::remainderf(-x, cuda::std::numeric_limits<T>::infinity()), -x));
    assert(eq(cuda::std::remainderf(-x, -cuda::std::numeric_limits<T>::infinity()), -x));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remainderf(x, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::remainderf(-x, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::remainderf(cuda::std::numeric_limits<T>::quiet_NaN(), x)));
    assert(cuda::std::isnan(cuda::std::remainderf(cuda::std::numeric_limits<T>::quiet_NaN(), -x)));
    assert(cuda::std::isnan(cuda::std::remainderf(x, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::remainderf(-x, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::remainderf(cuda::std::numeric_limits<T>::signaling_NaN(), x)));
    assert(cuda::std::isnan(cuda::std::remainderf(cuda::std::numeric_limits<T>::signaling_NaN(), -x)));
    assert(cuda::std::isnan(
      cuda::std::remainderf(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::remainderf(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value, where the remainder is negative
    {
      const T y        = T(3.456789123);
      const T quotient = T(4.0);
      assert(is_about(cuda::std::remainder(x, y), (x - quotient * y)));
    }

    // Some random value, where the remainder is positive
    {
      const T y        = T(5.678912345);
      const T quotient = T(2.0);
      assert(is_about(cuda::std::remainder(x, y), (x - quotient * y)));
    }
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    // If x is ±0 and y is not zero, ±0 is returned.
    assert(eq(cuda::std::remainderl(val, x), val));
    assert(eq(cuda::std::remainderl(-val, x), -val));
    assert(eq(cuda::std::remainderl(val, -x), val));
    assert(eq(cuda::std::remainderl(-val, -x), -val));

    assert(eq(cuda::std::remainderl(val, cuda::std::numeric_limits<T>::infinity()), val));
    assert(eq(cuda::std::remainderl(-val, cuda::std::numeric_limits<T>::infinity()), -val));

    // If x is ±∞ and y is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remainderl(cuda::std::numeric_limits<T>::infinity(), val)));
    assert(cuda::std::isnan(cuda::std::remainderl(-cuda::std::numeric_limits<T>::infinity(), val)));
    assert(cuda::std::isnan(cuda::std::remainderl(cuda::std::numeric_limits<T>::infinity(), x)));
    assert(cuda::std::isnan(cuda::std::remainderl(-cuda::std::numeric_limits<T>::infinity(), x)));
    assert(cuda::std::isnan(
      cuda::std::remainderl(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(
      cuda::std::remainderl(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity())));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remainderl(x, val)));
    assert(cuda::std::isnan(cuda::std::remainderl(-x, val)));

    // If y is ±∞ and x is finite, x is returned.
    assert(eq(cuda::std::remainderl(x, cuda::std::numeric_limits<T>::infinity()), x));
    assert(eq(cuda::std::remainderl(x, -cuda::std::numeric_limits<T>::infinity()), x));
    assert(eq(cuda::std::remainderl(-x, cuda::std::numeric_limits<T>::infinity()), -x));
    assert(eq(cuda::std::remainderl(-x, -cuda::std::numeric_limits<T>::infinity()), -x));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remainderl(x, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::remainderl(-x, cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::remainderl(cuda::std::numeric_limits<T>::quiet_NaN(), x)));
    assert(cuda::std::isnan(cuda::std::remainderl(cuda::std::numeric_limits<T>::quiet_NaN(), -x)));
    assert(cuda::std::isnan(cuda::std::remainderl(x, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::remainderl(-x, cuda::std::numeric_limits<T>::signaling_NaN())));
    assert(cuda::std::isnan(cuda::std::remainderl(cuda::std::numeric_limits<T>::signaling_NaN(), x)));
    assert(cuda::std::isnan(cuda::std::remainderl(cuda::std::numeric_limits<T>::signaling_NaN(), -x)));
    assert(cuda::std::isnan(
      cuda::std::remainderl(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(
      cuda::std::remainderl(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN())));

    // Some random value, where the remainder is negative
    {
      const T y        = T(3.456789123);
      const T quotient = T(4.0);
      assert(is_about(cuda::std::remainder(x, y), (x - quotient * y)));
    }

    // Some random value, where the remainder is positive
    {
      const T y        = T(5.678912345);
      const T quotient = T(2.0);
      assert(is_about(cuda::std::remainder(x, y), (x - quotient * y)));
    }
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test_remquo(const T x, const T y, const bool expected_octant)
{
  const T quotient = cuda::std::remainder(x, y);
  int octant       = 0;
  assert(eq(cuda::std::remquo(x, y, &octant), quotient));
  assert(cuda::std::signbit(octant) == expected_octant);
}

template <typename T>
__host__ __device__ void test_remquo(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::remquo(T{}, T{}, static_cast<int*>(nullptr))), ret>, "");

  const T x = T(13.23456789);

  // The result has the same sign as x, if the returned value is zero
  int octant = -1;
  assert(eq(cuda::std::remquo(val, x, &octant), cuda::std::remainder(val, x)));
  assert(!cuda::std::signbit(octant));

  octant = -1;
  assert(eq(cuda::std::remquo(-val, x, &octant), cuda::std::remainder(-val, x)));
  assert(!cuda::std::signbit(octant));

  if constexpr (cuda::std::is_integral_v<T>)
  {
    octant = -1;
    assert(eq(cuda::std::remquo(x, T(3.0), &octant), cuda::std::remainder(x, T(3.0))));
    assert(!cuda::std::signbit(octant));
  }
  else
  {
    // If x is ±0 and y is not zero, ±0 is returned.
    octant = -1;
    assert(eq(cuda::std::remquo(val, cuda::std::numeric_limits<T>::infinity(), &octant), val));
    assert(!cuda::std::signbit(octant));

    octant = -1;
    assert(eq(cuda::std::remquo(-val, cuda::std::numeric_limits<T>::infinity(), &octant), -val));
    assert(!cuda::std::signbit(octant));

    // If x is ±∞ and y is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remquo(cuda::std::numeric_limits<T>::infinity(), val, &octant)));
    assert(cuda::std::isnan(cuda::std::remquo(-cuda::std::numeric_limits<T>::infinity(), val, &octant)));
    assert(cuda::std::isnan(cuda::std::remquo(cuda::std::numeric_limits<T>::infinity(), x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquo(-cuda::std::numeric_limits<T>::infinity(), x, &octant)));
    assert(cuda::std::isnan(
      cuda::std::remquo(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity(), &octant)));
    assert(cuda::std::isnan(
      cuda::std::remquo(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity(), &octant)));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remquo(x, val, &octant)));
    assert(cuda::std::isnan(cuda::std::remquo(-x, val, &octant)));

    // If y is ±∞ and x is finite, x is returned.
    octant = -1;
    assert(eq(cuda::std::remquo(x, cuda::std::numeric_limits<T>::infinity(), &octant), x));
    assert(!cuda::std::signbit(octant));

    octant = -1;
    assert(eq(cuda::std::remquo(x, -cuda::std::numeric_limits<T>::infinity(), &octant), x));
    assert(!cuda::std::signbit(octant));

    octant = -1;
    assert(eq(cuda::std::remquo(-x, cuda::std::numeric_limits<T>::infinity(), &octant), -x));
    assert(!cuda::std::signbit(octant));

    octant = -1;
    assert(eq(cuda::std::remquo(-x, -cuda::std::numeric_limits<T>::infinity(), &octant), -x));
    assert(!cuda::std::signbit(octant));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remquo(x, cuda::std::numeric_limits<T>::quiet_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquo(-x, cuda::std::numeric_limits<T>::quiet_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquo(cuda::std::numeric_limits<T>::quiet_NaN(), x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquo(cuda::std::numeric_limits<T>::quiet_NaN(), -x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquo(x, cuda::std::numeric_limits<T>::signaling_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquo(-x, cuda::std::numeric_limits<T>::signaling_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquo(cuda::std::numeric_limits<T>::signaling_NaN(), x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquo(cuda::std::numeric_limits<T>::signaling_NaN(), -x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquo(
      cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquo(
      cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), &octant)));

    // Some random value, where the remquo is negative but octant positive
    {
      const T y = T(3.456789123);
      octant    = -1;
      assert(is_about(cuda::std::remquo(x, y, &octant), cuda::std::remainder(x, y)));
      assert(!cuda::std::signbit(octant));
    }

    // Some random value, where remquo and octant are negativew
    {
      const T y = T(3.456789123);
      octant    = -1;
      assert(is_about(cuda::std::remquo(-x, y, &octant), cuda::std::remainder(-x, y)));
      assert(cuda::std::signbit(octant));
    }

    // Some random value, where the remquo and octant are positive
    {
      const T y = T(5.678912345);
      octant    = -1;
      assert(is_about(cuda::std::remquo(x, y, &octant), cuda::std::remainder(x, y)));
      assert(!cuda::std::signbit(octant));
    }

    // Some random value, where the remquo is positive but negative octant
    {
      const T y = T(-5.678912345);
      octant    = -1;
      assert(is_about(cuda::std::remquo(x, y, &octant), cuda::std::remainder(x, y)));
      assert(cuda::std::signbit(octant));
    }
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    // If x is ±0 and y is not zero, ±0 is returned.
    octant = -1;
    assert(eq(cuda::std::remquof(val, cuda::std::numeric_limits<T>::infinity(), &octant), val));
    assert(!cuda::std::signbit(octant));

    octant = -1;
    assert(eq(cuda::std::remquof(-val, cuda::std::numeric_limits<T>::infinity(), &octant), -val));
    assert(!cuda::std::signbit(octant));

    // If x is ±∞ and y is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remquof(cuda::std::numeric_limits<T>::infinity(), val, &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(-cuda::std::numeric_limits<T>::infinity(), val, &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(cuda::std::numeric_limits<T>::infinity(), x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(-cuda::std::numeric_limits<T>::infinity(), x, &octant)));
    assert(cuda::std::isnan(
      cuda::std::remquof(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(
      -cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity(), &octant)));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remquof(x, val, &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(-x, val, &octant)));

    // If y is ±∞ and x is finite, x is returned.
    octant = -1;
    assert(eq(cuda::std::remquof(x, cuda::std::numeric_limits<T>::infinity(), &octant), x));
    assert(!cuda::std::signbit(octant));

    octant = -1;
    assert(eq(cuda::std::remquof(x, -cuda::std::numeric_limits<T>::infinity(), &octant), x));
    assert(!cuda::std::signbit(octant));

    octant = -1;
    assert(eq(cuda::std::remquof(-x, cuda::std::numeric_limits<T>::infinity(), &octant), -x));
    assert(!cuda::std::signbit(octant));

    octant = -1;
    assert(eq(cuda::std::remquof(-x, -cuda::std::numeric_limits<T>::infinity(), &octant), -x));
    assert(!cuda::std::signbit(octant));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remquof(x, cuda::std::numeric_limits<T>::quiet_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(-x, cuda::std::numeric_limits<T>::quiet_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(cuda::std::numeric_limits<T>::quiet_NaN(), x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(cuda::std::numeric_limits<T>::quiet_NaN(), -x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(x, cuda::std::numeric_limits<T>::signaling_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(-x, cuda::std::numeric_limits<T>::signaling_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(cuda::std::numeric_limits<T>::signaling_NaN(), x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(cuda::std::numeric_limits<T>::signaling_NaN(), -x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(
      cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquof(
      cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), &octant)));

    // Some random value, where the remquof is negative but octant positive
    {
      const T y = T(3.456789123);
      octant    = -1;
      assert(is_about(cuda::std::remquof(x, y, &octant), cuda::std::remainder(x, y)));
      assert(!cuda::std::signbit(octant));
    }

    // Some random value, where remquof and octant are negativew
    {
      const T y = T(3.456789123);
      octant    = -1;
      assert(is_about(cuda::std::remquof(-x, y, &octant), cuda::std::remainder(-x, y)));
      assert(cuda::std::signbit(octant));
    }

    // Some random value, where the remquof and octant are positive
    {
      const T y = T(5.678912345);
      octant    = -1;
      assert(is_about(cuda::std::remquof(x, y, &octant), cuda::std::remainder(x, y)));
      assert(!cuda::std::signbit(octant));
    }

    // Some random value, where the remquof is positive but negative octant
    {
      const T y = T(-5.678912345);
      octant    = -1;
      assert(is_about(cuda::std::remquof(x, y, &octant), cuda::std::remainder(x, y)));
      assert(cuda::std::signbit(octant));
    }
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    // If x is ±0 and y is not zero, ±0 is returned.
    octant = -1;
    assert(eq(cuda::std::remquol(val, cuda::std::numeric_limits<T>::infinity(), &octant), val));
    assert(!cuda::std::signbit(octant));

    octant = -1;
    assert(eq(cuda::std::remquol(-val, cuda::std::numeric_limits<T>::infinity(), &octant), -val));
    assert(!cuda::std::signbit(octant));

    // If x is ±∞ and y is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remquol(cuda::std::numeric_limits<T>::infinity(), val, &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(-cuda::std::numeric_limits<T>::infinity(), val, &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(cuda::std::numeric_limits<T>::infinity(), x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(-cuda::std::numeric_limits<T>::infinity(), x, &octant)));
    assert(cuda::std::isnan(
      cuda::std::remquol(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(
      -cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity(), &octant)));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remquol(x, val, &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(-x, val, &octant)));

    // If y is ±∞ and x is finite, x is returned.
    octant = -1;
    assert(eq(cuda::std::remquol(x, cuda::std::numeric_limits<T>::infinity(), &octant), x));
    assert(!cuda::std::signbit(octant));

    octant = -1;
    assert(eq(cuda::std::remquol(x, -cuda::std::numeric_limits<T>::infinity(), &octant), x));
    assert(!cuda::std::signbit(octant));

    octant = -1;
    assert(eq(cuda::std::remquol(-x, cuda::std::numeric_limits<T>::infinity(), &octant), -x));
    assert(!cuda::std::signbit(octant));

    octant = -1;
    assert(eq(cuda::std::remquol(-x, -cuda::std::numeric_limits<T>::infinity(), &octant), -x));
    assert(!cuda::std::signbit(octant));

    // If y is ±0 and x is not NaN, NaN is returned and FE_INVALID is raised.
    assert(cuda::std::isnan(cuda::std::remquol(x, cuda::std::numeric_limits<T>::quiet_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(-x, cuda::std::numeric_limits<T>::quiet_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(cuda::std::numeric_limits<T>::quiet_NaN(), x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(cuda::std::numeric_limits<T>::quiet_NaN(), -x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(x, cuda::std::numeric_limits<T>::signaling_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(-x, cuda::std::numeric_limits<T>::signaling_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(cuda::std::numeric_limits<T>::signaling_NaN(), x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(cuda::std::numeric_limits<T>::signaling_NaN(), -x, &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(
      cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), &octant)));
    assert(cuda::std::isnan(cuda::std::remquol(
      cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN(), &octant)));

    // Some random value, where the remquol is negative but octant positive
    {
      const T y = T(3.456789123);
      octant    = -1;
      assert(is_about(cuda::std::remquol(x, y, &octant), cuda::std::remainder(x, y)));
      assert(!cuda::std::signbit(octant));
    }

    // Some random value, where remquol and octant are negativew
    {
      const T y = T(3.456789123);
      octant    = -1;
      assert(is_about(cuda::std::remquol(-x, y, &octant), cuda::std::remainder(-x, y)));
      assert(cuda::std::signbit(octant));
    }

    // Some random value, where the remquol and octant are positive
    {
      const T y = T(5.678912345);
      octant    = -1;
      assert(is_about(cuda::std::remquol(x, y, &octant), cuda::std::remainder(x, y)));
      assert(!cuda::std::signbit(octant));
    }

    // Some random value, where the remquol is positive but negative octant
    {
      const T y = T(-5.678912345);
      octant    = -1;
      assert(is_about(cuda::std::remquol(x, y, &octant), cuda::std::remainder(x, y)));
      assert(cuda::std::signbit(octant));
    }
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
__host__ __device__ void test(const T val)
{
  test_remainder<T>(val);
  test_remquo<T>(val);
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
