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

// [simd.math], lerp

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_non_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec x(positive_math_values<T>{});
  Vec y(T{2});
  Vec z(T{0.25});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(x, y, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::lerp(x, y, z)), Vec>);
  static_assert(noexcept(cuda::std::simd::lerp(x, y, z)));

  Vec lerp_result = cuda::std::simd::lerp(x, y, z);
  for (int i = 0; i < N; ++i)
  {
    assert(lerp_result[i] == cuda::std::lerp(x[i], y[i], z[i]));
  }
}

template <typename T, int N>
TEST_FUNC void test_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec x(positive_math_values<T>{});
  Vec y(T{2});
  Vec z(T{0.25});
  T scalar_y{2};
  T scalar_z{0.25};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(scalar_y, y, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(x, scalar_y, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(x, y, scalar_z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(scalar_y, scalar_z, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(scalar_y, y, scalar_z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(x, scalar_y, scalar_z)), Vec>);

  static_assert(noexcept(cuda::std::simd::lerp(scalar_y, y, z)));
  static_assert(noexcept(cuda::std::simd::lerp(x, scalar_y, z)));
  static_assert(noexcept(cuda::std::simd::lerp(x, y, scalar_z)));
  static_assert(noexcept(cuda::std::simd::lerp(scalar_y, scalar_z, z)));
  static_assert(noexcept(cuda::std::simd::lerp(scalar_y, y, scalar_z)));
  static_assert(noexcept(cuda::std::simd::lerp(x, scalar_y, scalar_z)));

  Vec lerp_mixed1 = cuda::std::simd::lerp(scalar_y, y, z);
  Vec lerp_mixed2 = cuda::std::simd::lerp(x, scalar_y, z);
  Vec lerp_mixed3 = cuda::std::simd::lerp(x, y, scalar_z);
  Vec lerp_mixed4 = cuda::std::simd::lerp(scalar_y, scalar_z, z);
  Vec lerp_mixed5 = cuda::std::simd::lerp(scalar_y, y, scalar_z);
  Vec lerp_mixed6 = cuda::std::simd::lerp(x, scalar_y, scalar_z);
  for (int i = 0; i < N; ++i)
  {
    assert(lerp_mixed1[i] == cuda::std::lerp(scalar_y, y[i], z[i]));
    assert(lerp_mixed2[i] == cuda::std::lerp(x[i], scalar_y, z[i]));
    assert(lerp_mixed3[i] == cuda::std::lerp(x[i], y[i], scalar_z));
    assert(lerp_mixed4[i] == cuda::std::lerp(scalar_y, scalar_z, z[i]));
    assert(lerp_mixed5[i] == cuda::std::lerp(scalar_y, y[i], scalar_z));
    assert(lerp_mixed6[i] == cuda::std::lerp(x[i], scalar_y, scalar_z));
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
