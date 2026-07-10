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

// [simd.math], cbrt, sqrt

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec x(positive_math_values<T>{});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::sqrt(x)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::cbrt(x)), Vec>);

  static_assert(cuda::std::is_same_v<decltype(cuda::std::sqrt(x)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::cbrt(x)), Vec>);

  static_assert(noexcept(cuda::std::simd::cbrt(x)));
  static_assert(noexcept(cuda::std::simd::sqrt(x)));

  Vec sqrt_result = cuda::std::simd::sqrt(x);
  Vec cbrt_result = cuda::std::simd::cbrt(x);
  T tolerance     = T{1e-5};
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(sqrt_result[i], cuda::std::sqrt(x[i]), tolerance));
    assert(almost_equal(cbrt_result[i], cuda::std::cbrt(x[i]), tolerance));
  }
}

DEFINE_SIMD_MATH_FLOATING_TEST()
DEFINE_SIMD_MATH_FLOATING_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  assert(test_runtime());
  return 0;
}
