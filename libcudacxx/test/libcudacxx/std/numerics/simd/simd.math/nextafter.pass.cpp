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

// [simd.math], nextafter

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_non_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(positive_math_values<T>{});
  Vec next(T{2});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::nextafter(vec, next)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::nextafter(vec, next)), Vec>);
  static_assert(noexcept(cuda::std::simd::nextafter(vec, next)));

  Vec nextafter_result = cuda::std::simd::nextafter(vec, next);
  for (int i = 0; i < N; ++i)
  {
    assert(nextafter_result[i] == cuda::std::nextafter(vec[i], next[i]));
  }
}

template <typename T, int N>
TEST_FUNC void test_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec vec(positive_math_values<T>{});
  Vec next(T{2});
  T scalar_next{2};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::nextafter(vec, scalar_next)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::nextafter(scalar_next, next)), Vec>);
  static_assert(noexcept(cuda::std::simd::nextafter(vec, scalar_next)));
  static_assert(noexcept(cuda::std::simd::nextafter(scalar_next, next)));

  Vec nextafter_vec_scalar = cuda::std::simd::nextafter(vec, scalar_next);
  Vec nextafter_scalar_vec = cuda::std::simd::nextafter(scalar_next, next);
  for (int i = 0; i < N; ++i)
  {
    assert(nextafter_vec_scalar[i] == cuda::std::nextafter(vec[i], scalar_next));
    assert(nextafter_scalar_vec[i] == cuda::std::nextafter(scalar_next, next[i]));
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
