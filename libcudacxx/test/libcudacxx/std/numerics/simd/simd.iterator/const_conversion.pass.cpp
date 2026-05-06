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

// [simd.iterator], non-const to const iterator conversion for __simd_iterator.

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <typename T, int N>
TEST_FUNC constexpr void test_const_conversion()
{
  using Vec       = simd::basic_vec<T, simd::fixed_size<N>>;
  using ConstIter = typename Vec::const_iterator;

  Vec vec = make_iota_vec<T, N>();

  auto it            = vec.begin();
  ConstIter const_it = it;
  assert(*const_it == *it);
  assert(*const_it == static_cast<T>(0));

  const Vec const_vec{};
  ConstIter const_it2 = const_vec.begin();
  ConstIter const_it3 = const_vec.cbegin();
  assert(const_it2 == const_it3);
}

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_const_conversion<T, N>();
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
