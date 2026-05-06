//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: dynamic memory allocation is unsupported in tile code

// <random>
//
// Verify that distributions and engines accept signed char / unsigned char as
// template arguments (C++26, P4037R1) and produce correct values at runtime.

#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/random>
#include <cuda/std/type_traits>

#include "test_macros.h"

// Compile-only smoke check: every IntType distribution instantiates cleanly
// with signed char and unsigned char (P4037R1). Pure type-level checks, so we
// do not require the default constructor to be constexpr (binomial/poisson
// default ctors perform FP math that is not constexpr pre-C++23).

template <template <class> class Dist, class IntType>
struct smoke_int_distribution
{
  using D = Dist<IntType>;
  using P = typename D::param_type;
  static_assert(cuda::std::is_same_v<typename D::result_type, IntType>);
  static_assert(cuda::std::is_same_v<typename P::distribution_type, D>);
};

template struct smoke_int_distribution<cuda::std::uniform_int_distribution, signed char>;
template struct smoke_int_distribution<cuda::std::uniform_int_distribution, unsigned char>;
template struct smoke_int_distribution<cuda::std::binomial_distribution, signed char>;
template struct smoke_int_distribution<cuda::std::binomial_distribution, unsigned char>;
template struct smoke_int_distribution<cuda::std::geometric_distribution, signed char>;
template struct smoke_int_distribution<cuda::std::geometric_distribution, unsigned char>;
template struct smoke_int_distribution<cuda::std::negative_binomial_distribution, signed char>;
template struct smoke_int_distribution<cuda::std::negative_binomial_distribution, unsigned char>;
template struct smoke_int_distribution<cuda::std::poisson_distribution, signed char>;
template struct smoke_int_distribution<cuda::std::poisson_distribution, unsigned char>;

// End-to-end instantiation: distributions and engines compile and run for
// signed char / unsigned char.

template <class IntType>
TEST_FUNC TEST_CONSTEXPR_CXX20 bool test_uniform_int_small_type()
{
  using D = cuda::std::uniform_int_distribution<IntType>;
  using P = typename D::param_type;
  using G = cuda::std::philox4x64;

  // Construct over the full domain of the small type.
  IntType lo = cuda::std::numeric_limits<IntType>::min();
  IntType hi = cuda::std::numeric_limits<IntType>::max();
  D d(lo, hi);
  assert(d.a() == lo);
  assert(d.b() == hi);
  assert(d.min() == lo);
  assert(d.max() == hi);

  G g(42);
  for (int i = 0; i < 32; ++i)
  {
    IntType v = d(g);
    assert(static_cast<int>(v) >= static_cast<int>(lo));
    assert(static_cast<int>(v) <= static_cast<int>(hi));
  }

  // Sub-range also works.
  D d2(P(IntType{0}, IntType{10}));
  for (int i = 0; i < 32; ++i)
  {
    IntType v = d2(g);
    assert(static_cast<int>(v) >= 0);
    assert(static_cast<int>(v) <= 10);
  }
  return true;
}

TEST_FUNC TEST_CONSTEXPR_CXX20 bool test_small_lce()
{
  // A trivially small linear_congruential_engine with an 8-bit result_type.
  // Parameters chosen so 0 <= c < m and 0 <= a < m, and _Min (=0) < _Max (=12).
  using Engine = cuda::std::linear_congruential_engine<unsigned char, 5u, 1u, 13u>;
  static_assert(cuda::std::is_same_v<Engine::result_type, unsigned char>);
  static_assert(Engine::min() == 0u);
  static_assert(Engine::max() == 12u);

  // From seed 1: x -> (5x + 1) mod 13 produces the cycle 6, 5, 0, 1, 6, 5, 0, 1, ...
  Engine e(1);
  assert(e() == 6u);
  assert(e() == 5u);
  assert(e() == 0u);
  assert(e() == 1u);

  // All generated values stay in [min(), max()].
  for (int i = 0; i < 32; ++i)
  {
    unsigned char v = e();
    assert(v <= Engine::max());
  }
  return true;
}

int main(int, char**)
{
  test_uniform_int_small_type<signed char>();
  test_uniform_int_small_type<unsigned char>();
  test_small_lce();

#if TEST_STD_VER >= 2020
  static_assert(test_uniform_int_small_type<signed char>());
  static_assert(test_uniform_int_small_type<unsigned char>());
  static_assert(test_small_lce());
#endif // TEST_STD_VER >= 2020

  return 0;
}
