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

// [simd.math], sin, asin, cos, acos

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(math_values<T>{});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::acos(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::asin(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::cos(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::sin(vec)), Vec>);

  static_assert(cuda::std::is_same_v<decltype(cuda::std::acos(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::asin(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::cos(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::sin(vec)), Vec>);

  static_assert(noexcept(cuda::std::simd::acos(vec)));
  static_assert(noexcept(cuda::std::simd::asin(vec)));
  static_assert(noexcept(cuda::std::simd::cos(vec)));
  static_assert(noexcept(cuda::std::simd::sin(vec)));

  Vec acos_result = cuda::std::simd::acos(vec);
  Vec asin_result = cuda::std::simd::asin(vec);
  Vec cos_result  = cuda::std::simd::cos(vec);
  Vec sin_result  = cuda::std::simd::sin(vec);
  T tolerance     = T{1e-5};
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(acos_result[i], cuda::std::acos(vec[i]), tolerance));
    assert(almost_equal(asin_result[i], cuda::std::asin(vec[i]), tolerance));
    assert(almost_equal(cos_result[i], cuda::std::cos(vec[i]), tolerance));
    assert(almost_equal(sin_result[i], cuda::std::sin(vec[i]), tolerance));
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
