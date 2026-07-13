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

// [simd.math], ordered comparison functions

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_non_scalar()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;

  Vec lhs(math_values<T>{});
  Vec rhs(positive_math_values<T>{});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isgreater(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isgreaterequal(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isless(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::islessequal(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::islessgreater(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isunordered(lhs, rhs)), Mask>);

  static_assert(cuda::std::is_same_v<decltype(cuda::std::isgreater(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::isgreaterequal(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::isless(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::islessequal(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::islessgreater(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::isunordered(lhs, rhs)), Mask>);

  static_assert(noexcept(cuda::std::simd::isgreater(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::isgreaterequal(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::isless(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::islessequal(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::islessgreater(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::isunordered(lhs, rhs)));

  Mask greater       = cuda::std::simd::isgreater(lhs, rhs);
  Mask greater_equal = cuda::std::simd::isgreaterequal(lhs, rhs);
  Mask less          = cuda::std::simd::isless(lhs, rhs);
  Mask less_equal    = cuda::std::simd::islessequal(lhs, rhs);
  Mask less_greater  = cuda::std::simd::islessgreater(lhs, rhs);
  Mask unordered     = cuda::std::simd::isunordered(lhs, rhs);
  for (int i = 0; i < N; ++i)
  {
    assert(greater[i] == cuda::std::isgreater(lhs[i], rhs[i]));
    assert(greater_equal[i] == cuda::std::isgreaterequal(lhs[i], rhs[i]));
    assert(less[i] == cuda::std::isless(lhs[i], rhs[i]));
    assert(less_equal[i] == cuda::std::islessequal(lhs[i], rhs[i]));
    assert(less_greater[i] == cuda::std::islessgreater(lhs[i], rhs[i]));
    assert(unordered[i] == cuda::std::isunordered(lhs[i], rhs[i]));
  }
}

template <typename T, int N>
TEST_FUNC void test_scalar()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;

  Vec lhs(math_values<T>{});
  Vec rhs(positive_math_values<T>{});
  T scalar{0.5};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isgreater(scalar, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isgreater(lhs, scalar)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isgreaterequal(scalar, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isgreaterequal(lhs, scalar)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isless(scalar, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isless(lhs, scalar)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::islessequal(scalar, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::islessequal(lhs, scalar)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::islessgreater(scalar, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::islessgreater(lhs, scalar)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isunordered(scalar, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isunordered(lhs, scalar)), Mask>);

  static_assert(noexcept(cuda::std::simd::isgreater(scalar, rhs)));
  static_assert(noexcept(cuda::std::simd::isgreater(lhs, scalar)));
  static_assert(noexcept(cuda::std::simd::isgreaterequal(scalar, rhs)));
  static_assert(noexcept(cuda::std::simd::isgreaterequal(lhs, scalar)));
  static_assert(noexcept(cuda::std::simd::isless(scalar, rhs)));
  static_assert(noexcept(cuda::std::simd::isless(lhs, scalar)));
  static_assert(noexcept(cuda::std::simd::islessequal(scalar, rhs)));
  static_assert(noexcept(cuda::std::simd::islessequal(lhs, scalar)));
  static_assert(noexcept(cuda::std::simd::islessgreater(scalar, rhs)));
  static_assert(noexcept(cuda::std::simd::islessgreater(rhs, scalar)));
  static_assert(noexcept(cuda::std::simd::isunordered(scalar, rhs)));
  static_assert(noexcept(cuda::std::simd::isunordered(lhs, scalar)));

  Mask greater_vec_scalar       = cuda::std::simd::isgreater(lhs, scalar);
  Mask greater_scalar_vec       = cuda::std::simd::isgreater(scalar, rhs);
  Mask greater_equal_vec_scalar = cuda::std::simd::isgreaterequal(lhs, scalar);
  Mask greater_equal_scalar_vec = cuda::std::simd::isgreaterequal(scalar, rhs);

  Mask less_vec_scalar       = cuda::std::simd::isless(lhs, scalar);
  Mask less_scalar_vec       = cuda::std::simd::isless(scalar, rhs);
  Mask less_equal_vec_scalar = cuda::std::simd::islessequal(lhs, scalar);
  Mask less_equal_scalar_vec = cuda::std::simd::islessequal(scalar, rhs);

  Mask less_greater_vec_scalar = cuda::std::simd::islessgreater(lhs, scalar);
  Mask less_greater_scalar_vec = cuda::std::simd::islessgreater(scalar, rhs);
  Mask unordered_vec_scalar    = cuda::std::simd::isunordered(lhs, scalar);
  Mask unordered_scalar_vec    = cuda::std::simd::isunordered(scalar, rhs);
  for (int i = 0; i < N; ++i)
  {
    assert(greater_vec_scalar[i] == cuda::std::isgreater(lhs[i], scalar));
    assert(greater_scalar_vec[i] == cuda::std::isgreater(scalar, rhs[i]));
    assert(greater_equal_vec_scalar[i] == cuda::std::isgreaterequal(lhs[i], scalar));
    assert(greater_equal_scalar_vec[i] == cuda::std::isgreaterequal(scalar, rhs[i]));

    assert(less_vec_scalar[i] == cuda::std::isless(lhs[i], scalar));
    assert(less_scalar_vec[i] == cuda::std::isless(scalar, rhs[i]));
    assert(less_equal_vec_scalar[i] == cuda::std::islessequal(lhs[i], scalar));
    assert(less_equal_scalar_vec[i] == cuda::std::islessequal(scalar, rhs[i]));

    assert(less_greater_vec_scalar[i] == cuda::std::islessgreater(lhs[i], scalar));
    assert(less_greater_scalar_vec[i] == cuda::std::islessgreater(scalar, rhs[i]));
    assert(unordered_vec_scalar[i] == cuda::std::isunordered(lhs[i], scalar));
    assert(unordered_scalar_vec[i] == cuda::std::isunordered(scalar, rhs[i]));
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
