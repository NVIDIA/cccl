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
// template<class T, class Abi> constexpr pair<basic_vec<T, Abi>, basic_vec<T, Abi>>
// minmax(const basic_vec<T, Abi>& a, const basic_vec<T, Abi>& b) noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Pair = cuda::std::pair<Vec, Vec>;
  Vec a(T{6});
  Vec b(T{3});
  unused(a, b);

  static_assert(cuda::std::is_same_v<decltype(simd::minmax(a, b)), Pair>);
  static_assert(noexcept(simd::minmax(a, b)));

  Vec ia = make_iota_vec<T, N>();
  Vec ib = make_iota_reverse_vec<T, N>();
  Pair q = simd::minmax(ia, ib);
  for (int i = 0; i < N; ++i)
  {
    assert(q.first[i] == cuda::std::min(ia[i], ib[i]));
    assert(q.second[i] == cuda::std::max(ia[i], ib[i]));
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
