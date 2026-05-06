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

// [simd.iterator], dereference and subscript for __simd_iterator.

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <typename T, int N>
TEST_FUNC constexpr void test_dereference()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Iter = typename Vec::iterator;

  Vec vec = make_iota_vec<T, N>();

  auto it = vec.begin();
  static_assert(cuda::std::is_same_v<decltype(*it), typename Iter::value_type>);
  static_assert(cuda::std::is_same_v<decltype(it[0]), typename Iter::value_type>);
  static_assert(noexcept(*it));
  static_assert(noexcept(it[0]));

  for (int i = 0; i < N; ++i)
  {
    assert(*it == static_cast<T>(i));
    assert(it[0] == static_cast<T>(i));
    ++it;
  }

  auto it2 = vec.begin();
  for (int i = 0; i < N; ++i)
  {
    assert(it2[i] == static_cast<T>(i));
  }
}

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_dereference<T, N>();
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
