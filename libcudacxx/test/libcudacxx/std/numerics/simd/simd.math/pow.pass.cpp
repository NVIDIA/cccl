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

// [simd.math], pow

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "math_test_utils.h"

struct power_function
{
  template <typename X, typename Y>
  TEST_HOST_DEVICE_FUNC auto operator()(const X& x, const Y& y) const -> decltype(simd::pow(x, y));
};

struct unrelated_type
{};

template <int N>
TEST_HOST_DEVICE_FUNC void test_mixed_arguments()
{
  using F              = simd::vec<float, 3>;
  using D              = simd::vec<double, 3>;
  using Other          = simd::vec<double, 4>;
  using I              = simd::vec<int, 3>;
  using math_wrapper_t = explicit_math_wrapper<D>;

  // Overload participation.
  static_assert(cuda::std::is_invocable_v<power_function, math_wrapper_t, math_wrapper_t>);
  static_assert(cuda::std::is_invocable_v<power_function, math_wrapper_t, F>);
  static_assert(cuda::std::is_invocable_v<power_function, F, D>);
  static_assert(cuda::std::is_invocable_v<power_function, D, F>);
  static_assert(cuda::std::is_invocable_v<power_function, D, I>);
  static_assert(cuda::std::is_invocable_v<power_function, F, float>);
  static_assert(!cuda::std::is_invocable_v<power_function, F, double>);
  static_assert(!cuda::std::is_invocable_v<power_function, F, I>);
  static_assert(!cuda::std::is_invocable_v<power_function, F, Other>);
  static_assert(!cuda::std::is_invocable_v<power_function, Other, F>);
  static_assert(!cuda::std::is_invocable_v<power_function, float, double>);
  static_assert(!cuda::std::is_invocable_v<power_function, F, unrelated_type>);

  // Mixed argument types.
  const F x(4.0f);
  const D y(2.0);
  static_assert(cuda::std::is_same_v<decltype(simd::pow(x, y)), D>);
  static_assert(cuda::std::is_same_v<decltype(simd::pow(y, x)), D>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::pow(x, y)), D>);
  static_assert(noexcept(simd::pow(x, y)));
  const D power    = simd::pow(x, y);
  const D reversed = simd::pow(y, x);
  check_math_values(power, 16.0);
  check_math_values(reversed, 16.0);

  // Mixed wrapper and vector arguments.
  const math_wrapper<D> y1{D(2.0)};
  static_assert(cuda::std::is_same_v<decltype(simd::pow(x, y)), D>);
  static_assert(cuda::std::is_same_v<decltype(simd::pow(y1, x)), D>);
  check_math_values(simd::pow(x, y1), 16.0);
  check_math_values(simd::pow(y1, x), 16.0);

  const explicit_math_wrapper<D> explicit_x{D(4.0)};
  check_math_values(simd::pow(explicit_x, x), 256.0);
}

template <typename T, int N>
TEST_HOST_DEVICE_FUNC void test_wrapper()
{
  using V = simd::vec<T, N>;
  const explicit_math_wrapper<V> x{V(T{4})};
  const explicit_math_wrapper<V> y{V(T{2})};
  static_assert(noexcept(simd::pow(x, y)));

  check_math_values(simd::pow(x, y), T{16});
}

template <typename T, int N>
TEST_HOST_DEVICE_FUNC void test_non_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec x(positive_math_values<T>{});
  Vec y(T{0.5});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::pow(x, y)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::pow(x, y)), Vec>);
  static_assert(noexcept(cuda::std::simd::pow(x, y)));

  Vec pow_result = cuda::std::simd::pow(x, y);
  T tolerance    = T{1e-5};
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(pow_result[i], cuda::std::pow(x[i], y[i]), tolerance));
  }
}

template <typename T, int N>
TEST_HOST_DEVICE_FUNC void test_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec x(positive_math_values<T>{});
  T scalar_y{0.5};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::pow(x, scalar_y)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::pow(scalar_y, x)), Vec>);

  static_assert(noexcept(cuda::std::simd::pow(x, scalar_y)));
  static_assert(noexcept(cuda::std::simd::pow(scalar_y, x)));

  Vec pow_vec_scalar = cuda::std::simd::pow(x, scalar_y);
  Vec pow_scalar_vec = cuda::std::simd::pow(scalar_y, x);
  T tolerance        = T{1e-5};
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(pow_vec_scalar[i], cuda::std::pow(x[i], scalar_y), tolerance));
    assert(almost_equal(pow_scalar_vec[i], cuda::std::pow(scalar_y, x[i]), tolerance));
  }
}

template <typename T, int N>
TEST_HOST_DEVICE_FUNC void test_type()
{
  test_mixed_arguments<N>();
  test_wrapper<T, N>();
  test_non_scalar<T, N>();
  test_scalar<T, N>();
}

DEFINE_SIMD_MATH_FLOATING_TEST()
DEFINE_SIMD_MATH_FLOATING_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  assert(test_runtime());
  return 0;
}
