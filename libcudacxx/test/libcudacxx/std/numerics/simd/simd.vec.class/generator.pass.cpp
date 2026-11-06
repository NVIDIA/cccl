//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a host device function in tile mode

// <cuda/std/__simd_>
// [simd.ctor], [simd.mask.ctor], generator constraints and invocation

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <typename T>
struct generator
{
  template <typename I>
  TEST_HOST_DEVICE_FUNC T operator()(I) const;
};

struct rvalue_generator
{
  template <typename I>
  TEST_HOST_DEVICE_FUNC int operator()(I) &&;
};

struct mixed_generator
{
  template <typename I>
  TEST_HOST_DEVICE_FUNC cuda::std::conditional_t<I::value == 2, double, int> operator()(I) const;
};

struct mixed_mask_generator
{
  template <typename I>
  TEST_HOST_DEVICE_FUNC cuda::std::conditional_t<I::value == 2, int, bool> operator()(I) const;
};

struct conversion_result
{
  TEST_HOST_DEVICE_FUNC constexpr operator short() const
  {
    return 2;
  }
  TEST_HOST_DEVICE_FUNC explicit constexpr operator int() const
  {
    return 3;
  }
};

struct conversion_generator
{
  template <typename I>
  TEST_HOST_DEVICE_FUNC constexpr conversion_result operator()(I) const
  {
    return {};
  }
};

template <typename T>
struct lvalue_generator
{
  // test Lvalue invocation
  template <typename I>
  TEST_HOST_DEVICE_FUNC constexpr T operator()(I) &
  {
    return static_cast<T>(I::value % 2);
  }
};

template <int N>
TEST_HOST_DEVICE_FUNC constexpr void test_vec()
{
  using cuda::std::is_constructible_v;
  using cuda::std::is_convertible_v;
  using Vec = simd::vec<int, N>;

  static_assert(is_constructible_v<Vec, generator<short>>);
  static_assert(is_constructible_v<simd::vec<double, N>, generator<int>>);
  static_assert(is_constructible_v<simd::vec<double, N>, generator<float>>);
  static_assert(is_constructible_v<simd::vec<cuda::std::complex<float>, N>, generator<float>>);
  static_assert(is_constructible_v<Vec, mixed_generator> == (N <= 2));

  static_assert(!is_constructible_v<Vec, generator<double>>);
  static_assert(!is_constructible_v<simd::vec<float, N>, generator<double>>);
  static_assert(!is_constructible_v<simd::vec<short, N>, generator<int>>);
  static_assert(!is_constructible_v<Vec, generator<void>>);
  static_assert(!is_constructible_v<Vec, rvalue_generator>);

  // From_i is the expression type, so references need only be convertible
  static_assert(is_constructible_v<Vec, generator<double&>>);
  static_assert(is_constructible_v<Vec, conversion_generator>);
  static_assert(!is_convertible_v<conversion_generator, Vec>); // Vec v = conversion_generator{}; is rejected

  lvalue_generator<short> gen{};
  static_assert(is_constructible_v<Vec, decltype(gen)>);
  static_assert(is_constructible_v<Vec, decltype(gen)&>);
  static_assert(!is_constructible_v<Vec, const decltype(gen)&>);

  const Vec values(gen);
  const simd::vec<double, N> widening(lvalue_generator<int>{});
  const Vec converted(conversion_generator{});
  for (int i = 0; i < N; ++i)
  {
    assert(values[i] == i % 2);
    assert(widening[i] == double(i % 2));
    assert(converted[i] == 3);
  }
}

template <int N>
TEST_HOST_DEVICE_FUNC constexpr void test_mask()
{
  using cuda::std::is_constructible_v;
  using Mask = simd::mask<int, N>;

  static_assert(is_constructible_v<Mask, generator<bool>>);
  static_assert(is_constructible_v<Mask, mixed_mask_generator> == (N <= 2));

  static_assert(!is_constructible_v<Mask, generator<int>>);
  static_assert(!is_constructible_v<Mask, generator<bool&>>);
  static_assert(!is_constructible_v<Mask, generator<const bool&>>);
  static_assert(!is_constructible_v<Mask, conversion_generator>);
  static_assert(!is_constructible_v<Mask, generator<void>>);

  const Mask mask(lvalue_generator<bool>{});
  for (int i = 0; i < N; ++i)
  {
    assert(mask[i] == (i % 2 != 0));
  }
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  test_vec<1>();
  test_vec<4>();
  test_mask<1>();
  test_mask<4>();
  return true;
}

int main(int, char**)
{
  assert(test()); // NOLINT(misc-static-assert)
  static_assert(test());
  return 0;
}
