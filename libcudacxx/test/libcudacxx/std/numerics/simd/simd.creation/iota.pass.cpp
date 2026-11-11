//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a host device function in tile mode

// [simd.creation], iota

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_HOST_DEVICE_FUNC constexpr void test_type()
{
  using V = simd::vec<T, N>;

  static_assert(cuda::std::is_same_v<decltype(simd::iota<T>), const T>);
  static_assert(cuda::std::is_same_v<decltype(simd::iota<V>), const V>);
  static_assert(simd::iota<T> == T{});

  constexpr V indices     = simd::iota<V>;
  const V runtime_indices = simd::iota<V>;
  for (int i = 0; i < N; ++i)
  {
    assert(indices[i] == static_cast<T>(i));
    assert(runtime_indices[i] == static_cast<T>(i));
  }
  assert(simd::iota<T> == T{});
}

DEFINE_BASIC_VEC_TEST()

int main(int, char**)
{
  test(); // NOLINT(misc-static-assert)
  static_assert(test());
  return 0;
}
