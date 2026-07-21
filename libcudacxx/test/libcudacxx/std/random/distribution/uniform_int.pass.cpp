//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: dynamic memory allocation is unsupported in tile code
//
// REQUIRES: long_tests

// <random>

// template<class IntType = int>
// class uniform_int_distribution

#include <cuda/std/cassert>
#include <cuda/std/random>

#include "random_utilities/test_distribution.h"
#include "test_macros.h"

template <class T>
struct uniform_int_cdf
{
  using P = typename cuda::std::uniform_int_distribution<T>::param_type;

  TEST_FUNC double operator()(double x, const P& p) const
  {
    // CDF: F(x; a, b) = (floor(x) - a + 1) / (b - a + 1) for a <= x <= b
    //                 = 0 for x < a
    //                 = 1 for x > b
    double a = static_cast<double>(p.a());
    double b = static_cast<double>(p.b());
    if (x < a)
    {
      return 0.0;
    }
    if (x > b)
    {
      return 1.0;
    }
    return (cuda::std::floor(x) - a + 1.0) / (b - a + 1.0);
  }
};

template <class T>
TEST_FUNC void test()
{
  [[maybe_unused]] const bool test_constexpr = true;
  using D                                    = cuda::std::uniform_int_distribution<T>;
  using P                                    = typename D::param_type;
  using G                                    = cuda::std::philox4x64;
  cuda::std::array<P, 5> params              = {
    P(0, 10),
    P(1, 100),
    P(-50, 50),
    P(cuda::std::numeric_limits<T>::min(), cuda::std::numeric_limits<T>::max()),
    P(100, 1000)};
  test_distribution<D, false, G, test_constexpr>(params, uniform_int_cdf<T>{});
}

// P4037R1: made signed char / unsigned char valid IntType template arguments.

TEST_FUNC void test_p4037r1_small_types()
{
  using D_s = cuda::std::uniform_int_distribution<signed char>;
  using D_u = cuda::std::uniform_int_distribution<unsigned char>;
  static_assert(cuda::std::is_same_v<D_s::result_type, signed char>);
  static_assert(cuda::std::is_same_v<D_u::result_type, unsigned char>);
  static_assert(D_s().min() == 0);
  static_assert(D_s().max() == cuda::std::numeric_limits<signed char>::max());
  static_assert(D_u().min() == 0);
  static_assert(D_u().max() == cuda::std::numeric_limits<unsigned char>::max());
}

int main(int, char**)
{
  test<int>();
  test<long>();
  test<short>();
  test_p4037r1_small_types();
  return 0;
}
