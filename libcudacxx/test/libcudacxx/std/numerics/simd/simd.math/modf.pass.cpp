//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a host device function in tile mode

// <cuda/std/__simd_>

// [simd.math], modf

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "math_test_utils.h"

template <typename T, int N>
TEST_HOST_DEVICE_FUNC void test_wrapper()
{
  using V = simd::vec<T, N>;
  // modf takes a vector reference, so the wrapper must be implicitly convertible
  const math_wrapper<V> x{V(T{4})};
  V integrals;
  static_assert(!noexcept(simd::modf(x, &integrals)));

  check_math_values(simd::modf(x, &integrals), T{0});
  check_math_values(integrals, T{4});
}

template <typename T, int N>
TEST_HOST_DEVICE_FUNC void test_type()
{
  test_wrapper<T, N>();
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec lhs(positive_math_values<T>{});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::modf(lhs, cuda::std::declval<Vec*>())), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::modf(lhs, cuda::std::declval<Vec*>())), Vec>);
  static_assert(noexcept(cuda::std::simd::modf(lhs, cuda::std::declval<Vec*>())));

  Vec integrals;
  Vec modf_result = cuda::std::simd::modf(lhs, &integrals);
  for (int i = 0; i < N; ++i)
  {
    T integral = 0;
    assert(modf_result[i] == cuda::std::modf(lhs[i], &integral));
    assert(integrals[i] == integral);
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
