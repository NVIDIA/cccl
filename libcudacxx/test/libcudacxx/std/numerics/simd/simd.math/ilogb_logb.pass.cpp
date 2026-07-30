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

// [simd.math], ilogb, logb

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

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::ilogb(vec)), IntVec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::logb(vec)), Vec>);

  static_assert(cuda::std::is_same_v<decltype(cuda::std::ilogb(vec)), IntVec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::logb(vec)), Vec>);

  static_assert(noexcept(cuda::std::simd::ilogb(vec)));
  static_assert(noexcept(cuda::std::simd::logb(vec)));

  IntVec ilogb_result = cuda::std::simd::ilogb(vec);
  Vec logb_result     = cuda::std::simd::logb(vec);
  for (int i = 0; i < N; ++i)
  {
    assert(ilogb_result[i] == cuda::std::ilogb(vec[i]));
    assert(logb_result[i] == cuda::std::logb(vec[i]));
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
