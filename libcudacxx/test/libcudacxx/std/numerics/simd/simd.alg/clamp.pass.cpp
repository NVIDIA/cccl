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
// template<class T, class Abi> constexpr basic_vec<T, Abi> clamp(
//   const basic_vec<T, Abi>& v,
//   const basic_vec<T, Abi>& lo,
//   const basic_vec<T, Abi>& hi);

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec v(T{2});
  Vec lo(T{1});
  Vec hi(T{3});
  unused(v, lo, hi);

  static_assert(cuda::std::is_same_v<decltype(simd::clamp(v, lo, hi)), Vec>);

  Vec iv      = make_iota_vec<T, N>();
  Vec clamped = simd::clamp(iv, lo, hi);
  for (int i = 0; i < N; ++i)
  {
    assert(clamped[i] == cuda::std::clamp(iv[i], lo[i], hi[i]));
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
