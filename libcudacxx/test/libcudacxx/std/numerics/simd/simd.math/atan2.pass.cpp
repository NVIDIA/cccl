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

// [simd.math], atan2

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_non_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(math_values<T>{});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::atan2(vec, vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::atan2(vec, vec)), Vec>);
  static_assert(noexcept(cuda::std::simd::atan2(vec, vec)));

  Vec atan2_result = cuda::std::simd::atan2(vec, vec);
  T tolerance      = T{1e-5};
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(atan2_result[i], cuda::std::atan2(vec[i], vec[i]), tolerance));
  }
}

template <typename T, int N>
TEST_FUNC void test_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(math_values<T>{});
  T scalar{0.5};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::atan2(vec, scalar)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::atan2(scalar, vec)), Vec>);
  static_assert(noexcept(cuda::std::simd::atan2(vec, scalar)));
  static_assert(noexcept(cuda::std::simd::atan2(scalar, vec)));

  Vec atan2_vec_scalar = cuda::std::simd::atan2(vec, scalar);
  Vec atan2_scalar_vec = cuda::std::simd::atan2(scalar, vec);
  T tolerance          = T{1e-5};
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(atan2_vec_scalar[i], cuda::std::atan2(vec[i], scalar), tolerance));
    assert(almost_equal(atan2_scalar_vec[i], cuda::std::atan2(scalar, vec[i]), tolerance));
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
