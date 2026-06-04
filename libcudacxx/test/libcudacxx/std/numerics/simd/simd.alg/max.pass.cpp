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

// [simd.alg], algorithms
//
// template<class T, class Abi> constexpr basic_vec<T, Abi> max(
//   const basic_vec<T, Abi>& a, const basic_vec<T, Abi>& b) noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec a(T{6});
  Vec b(T{3});
  unused(a, b);

  static_assert(cuda::std::is_same_v<decltype(simd::max(a, b)), Vec>);
  static_assert(noexcept(simd::max(a, b)));

  Vec ia = make_iota_vec<T, N>();
  Vec ib = make_iota_reverse_vec<T, N>();
  Vec mx = simd::max(ia, ib);
  for (int i = 0; i < N; ++i)
  {
    assert(mx[i] == cuda::std::max(ia[i], ib[i]));
  }
}

DEFINE_BASIC_VEC_TEST()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  static_assert(test());
  assert(test_runtime());
  return 0;
}
