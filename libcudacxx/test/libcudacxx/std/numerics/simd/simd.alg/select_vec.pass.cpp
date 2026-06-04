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
// template<size_t Bytes, class Abi, class T, class U> constexpr auto
//   select(const basic_mask<Bytes, Abi>& c, const T& a, const U& b) noexcept
//   -> decltype(__simd-select-impl(c, a, b));
//
// Covers the basic_vec overload: select(mask, vec, vec) -> vec.

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;

  Vec a(T{1});
  Vec b(T{2});
  Mask m(is_even{});

  static_assert(cuda::std::is_same_v<decltype(simd::select(m, a, b)), Vec>);
  static_assert(noexcept(simd::select(m, a, b)));

  Vec r = simd::select(m, a, b);
  for (int i = 0; i < N; ++i)
  {
    assert(r[i] == (i % 2 == 0 ? T{1} : T{2}));
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
