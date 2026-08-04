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

// [simd.math], hypot

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
  Vec z(T{0.25});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, y)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, y, z)), Vec>);

  static_assert(cuda::std::is_same_v<decltype(cuda::std::hypot(x, y)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::hypot(x, y, z)), Vec>);

  static_assert(noexcept(cuda::std::simd::hypot(x, y)));
  static_assert(noexcept(cuda::std::simd::hypot(x, y, z)));

  Vec hypot2_result = cuda::std::simd::hypot(x, y);
  Vec hypot3_result = cuda::std::simd::hypot(x, y, z);
  T tolerance       = T{1e-5};
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(hypot2_result[i], cuda::std::hypot(x[i], y[i]), tolerance));
    assert(almost_equal(hypot3_result[i], cuda::std::hypot(x[i], y[i], z[i]), tolerance));
  }
}

template <typename T, int N>
TEST_FUNC void test_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec x(positive_math_values<T>{});
  Vec y(T{0.5});
  Vec z(T{0.25});
  T scalar_y{0.5};
  T scalar_x{0.25};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, scalar_y)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(scalar_y, x)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, y, scalar_x)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, scalar_y, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(scalar_y, y, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(scalar_y, scalar_x, x)), Vec>);

  static_assert(noexcept(cuda::std::simd::hypot(x, scalar_y)));
  static_assert(noexcept(cuda::std::simd::hypot(scalar_y, x)));
  static_assert(noexcept(cuda::std::simd::hypot(x, y, scalar_x)));
  static_assert(noexcept(cuda::std::simd::hypot(x, scalar_y, z)));
  static_assert(noexcept(cuda::std::simd::hypot(scalar_y, y, z)));
  static_assert(noexcept(cuda::std::simd::hypot(scalar_y, scalar_x, x)));

  Vec hypot_vec_scalar        = cuda::std::simd::hypot(x, scalar_y);
  Vec hypot_scalar_vec        = cuda::std::simd::hypot(scalar_y, x);
  Vec hypot_vec_scalar_vec    = cuda::std::simd::hypot(x, y, scalar_x);
  Vec hypot_vec_scalar_scalar = cuda::std::simd::hypot(x, scalar_y, z);
  Vec hypot_scalar_vec_scalar = cuda::std::simd::hypot(scalar_y, y, z);
  Vec hypot_scalar_scalar_vec = cuda::std::simd::hypot(scalar_y, scalar_x, x);
  T tolerance                 = T{1e-5};
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(hypot_vec_scalar[i], cuda::std::hypot(x[i], scalar_y), tolerance));
    assert(almost_equal(hypot_scalar_vec[i], cuda::std::hypot(scalar_y, x[i]), tolerance));
    assert(almost_equal(hypot_vec_scalar_vec[i], cuda::std::hypot(x[i], y[i], scalar_x), tolerance));
    assert(almost_equal(hypot_vec_scalar_scalar[i], cuda::std::hypot(x[i], scalar_y, z[i]), tolerance));
    assert(almost_equal(hypot_scalar_vec_scalar[i], cuda::std::hypot(scalar_y, y[i], z[i]), tolerance));
    assert(almost_equal(hypot_scalar_scalar_vec[i], cuda::std::hypot(scalar_y, scalar_x, x[i]), tolerance));
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
