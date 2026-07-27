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

// [simd.math], remquo

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_non_scalar()
{
  using Vec    = simd::basic_vec<T, simd::fixed_size<N>>;
  using IntVec = simd::rebind_t<int, Vec>;

  Vec lhs(positive_math_values<T>{});
  Vec rhs(T{0.5});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::remquo(lhs, rhs, cuda::std::declval<IntVec*>())), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::remquo(lhs, rhs, cuda::std::declval<IntVec*>())), Vec>);
  static_assert(noexcept(cuda::std::simd::remquo(lhs, rhs, cuda::std::declval<IntVec*>())));

  IntVec quotients;
  Vec remquo_result = cuda::std::simd::remquo(lhs, rhs, &quotients);
  for (int i = 0; i < N; ++i)
  {
    int quotient = 0;
    assert(remquo_result[i] == cuda::std::remquo(lhs[i], rhs[i], &quotient));
    assert(quotients[i] == quotient);
  }
}

template <typename T, int N>
TEST_FUNC void test_scalar()
{
  using Vec    = simd::basic_vec<T, simd::fixed_size<N>>;
  using IntVec = simd::rebind_t<int, Vec>;

  Vec lhs(positive_math_values<T>{});
  T scalar{0.5};

  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::simd::remquo(lhs, scalar, cuda::std::declval<IntVec*>())), Vec>);
  static_assert(noexcept(cuda::std::simd::remquo(lhs, scalar, cuda::std::declval<IntVec*>())));

  IntVec mixed_quotients;
  Vec remquo_mixed = cuda::std::simd::remquo(lhs, scalar, &mixed_quotients);
  for (int i = 0; i < N; ++i)
  {
    int quotient = 0;
    assert(remquo_mixed[i] == cuda::std::remquo(lhs[i], scalar, &quotient));
    assert(mixed_quotients[i] == quotient);
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
