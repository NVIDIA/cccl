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

// [simd.math], fmin

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_non_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec lhs(positive_math_values<T>{});
  Vec rhs(T{0.5});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmin(lhs, rhs)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::fmin(lhs, rhs)), Vec>);
  static_assert(noexcept(cuda::std::simd::fmin(lhs, rhs)));

  Vec fmin_result = cuda::std::simd::fmin(lhs, rhs);
  for (int i = 0; i < N; ++i)
  {
    assert(fmin_result[i] == cuda::std::fmin(lhs[i], rhs[i]));
  }
}

template <typename T, int N>
TEST_FUNC void test_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec lhs(positive_math_values<T>{});
  Vec rhs(T{0.5});
  T scalar{0.5};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmin(lhs, scalar)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmin(scalar, rhs)), Vec>);
  static_assert(noexcept(cuda::std::simd::fmin(lhs, scalar)));
  static_assert(noexcept(cuda::std::simd::fmin(scalar, rhs)));

  Vec fmin_vec_scalar = cuda::std::simd::fmin(lhs, scalar);
  Vec fmin_scalar_vec = cuda::std::simd::fmin(scalar, rhs);
  for (int i = 0; i < N; ++i)
  {
    assert(fmin_vec_scalar[i] == cuda::std::fmin(lhs[i], scalar));
    assert(fmin_scalar_vec[i] == cuda::std::fmin(scalar, rhs[i]));
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
