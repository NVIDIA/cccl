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

// [simd.math], copysign

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_non_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(positive_math_values<T>{});
  Vec signs(T{2});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::copysign(vec, signs)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::copysign(vec, signs)), Vec>);
  static_assert(noexcept(cuda::std::simd::copysign(vec, signs)));

  Vec copysign_result = cuda::std::simd::copysign(vec, signs);
  for (int i = 0; i < N; ++i)
  {
    assert(copysign_result[i] == cuda::std::copysign(vec[i], signs[i]));
  }
}

template <typename T, int N>
TEST_FUNC void test_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec vec(positive_math_values<T>{});
  Vec signs(T{2});
  T scalar_sign{2};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::copysign(vec, scalar_sign)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::copysign(scalar_sign, signs)), Vec>);

  Vec copysign_vec_scalar = cuda::std::simd::copysign(vec, scalar_sign);
  Vec copysign_scalar_vec = cuda::std::simd::copysign(scalar_sign, signs);
  for (int i = 0; i < N; ++i)
  {
    assert(copysign_vec_scalar[i] == cuda::std::copysign(vec[i], scalar_sign));
    assert(copysign_scalar_vec[i] == cuda::std::copysign(scalar_sign, signs[i]));
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
