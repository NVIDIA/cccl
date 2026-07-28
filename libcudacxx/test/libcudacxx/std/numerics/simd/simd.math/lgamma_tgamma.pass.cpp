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

// [simd.math], gamma functions

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(positive_math_values<T>{});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lgamma(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::tgamma(vec)), Vec>);

  static_assert(cuda::std::is_same_v<decltype(cuda::std::lgamma(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::tgamma(vec)), Vec>);

  static_assert(noexcept(cuda::std::simd::lgamma(vec)));
  static_assert(noexcept(cuda::std::simd::tgamma(vec)));

  Vec lgamma_result = cuda::std::simd::lgamma(vec);
  Vec tgamma_result = cuda::std::simd::tgamma(vec);
  T tolerance       = T{1e-5};
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(lgamma_result[i], cuda::std::lgamma(vec[i]), tolerance));
    assert(almost_equal(tgamma_result[i], cuda::std::tgamma(vec[i]), tolerance));
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
