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

// [simd.math], ldexp

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec    = simd::basic_vec<T, simd::fixed_size<N>>;
  using IntVec = simd::rebind_t<int, Vec>;

  Vec vec(positive_math_values<T>{});
  IntVec exponents(1);
  int scalar_exp = 1;

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::ldexp(vec, exponents)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::ldexp(vec, scalar_exp)), Vec>);
  static_assert(noexcept(cuda::std::simd::ldexp(vec, exponents)));
  static_assert(noexcept(cuda::std::simd::ldexp(vec, scalar_exp)));

  Vec ldexp_result = cuda::std::simd::ldexp(vec, exponents);
  for (int i = 0; i < N; ++i)
  {
    assert(ldexp_result[i] == cuda::std::ldexp(vec[i], exponents[i]));
  }

  Vec ldexp_scalar = cuda::std::simd::ldexp(vec, scalar_exp);
  for (int i = 0; i < N; ++i)
  {
    assert(ldexp_scalar[i] == cuda::std::ldexp(vec[i], scalar_exp));
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
