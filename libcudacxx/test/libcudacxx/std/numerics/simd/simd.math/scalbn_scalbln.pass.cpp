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

// [simd.math], scalbn, scalbln

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_non_scalar()
{
  using Vec     = simd::basic_vec<T, simd::fixed_size<N>>;
  using IntVec  = simd::rebind_t<int, Vec>;
  using LongVec = simd::rebind_t<long, Vec>;

  Vec vec(positive_math_values<T>{});
  IntVec exponents(1);
  LongVec long_exponents(1);

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::scalbn(vec, exponents)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::scalbln(vec, long_exponents)), Vec>);

  static_assert(cuda::std::is_same_v<decltype(cuda::std::scalbn(vec, exponents)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::scalbln(vec, long_exponents)), Vec>);

  static_assert(noexcept(cuda::std::simd::scalbn(vec, exponents)));
  static_assert(noexcept(cuda::std::simd::scalbln(vec, long_exponents)));

  Vec scalbn_result  = cuda::std::simd::scalbn(vec, exponents);
  Vec scalbln_result = cuda::std::simd::scalbln(vec, long_exponents);
  for (int i = 0; i < N; ++i)
  {
    assert(scalbn_result[i] == cuda::std::scalbn(vec[i], exponents[i]));
    assert(scalbln_result[i] == cuda::std::scalbln(vec[i], long_exponents[i]));
  }
}

template <typename T, int N>
TEST_FUNC void test_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(positive_math_values<T>{});
  int scalar_exp       = 1;
  long scalar_long_exp = 1;

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::scalbn(vec, scalar_exp)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::scalbln(vec, scalar_long_exp)), Vec>);

  Vec scalbn_scalar  = cuda::std::simd::scalbn(vec, scalar_exp);
  Vec scalbln_scalar = cuda::std::simd::scalbln(vec, scalar_long_exp);
  for (int i = 0; i < N; ++i)
  {
    assert(scalbn_scalar[i] == cuda::std::scalbn(vec[i], scalar_exp));
    assert(scalbln_scalar[i] == cuda::std::scalbln(vec[i], scalar_long_exp));
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
