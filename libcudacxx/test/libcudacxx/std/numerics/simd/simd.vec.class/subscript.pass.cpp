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

// [simd.subscr], basic_vec subscript operators
//
// constexpr value_type operator[](simd-size-type) const;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// subscript read-back

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(T{7});

  static_assert(cuda::std::is_same_v<decltype(vec[0]), typename Vec::value_type>);
  static_assert(noexcept(vec[0]));
  static_assert(is_const_member_function_v<decltype(&Vec::operator[])>);
  unused(vec);

  Vec iota = make_iota_vec<T, N>();
  for (int i = 0; i < N; ++i)
  {
    assert(iota[i] == static_cast<T>(i));
  }
}

//----------------------------------------------------------------------------------------------------------------------

DEFINE_BASIC_VEC_TEST()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  static_assert(test());
  assert(test_runtime());
  return 0;
}
