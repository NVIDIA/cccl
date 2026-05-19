//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__simd_>

// [simd.math], pow

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_non_scalar()
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
TEST_FUNC void test_scalar()
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
TEST_FUNC void test_type()
{
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
