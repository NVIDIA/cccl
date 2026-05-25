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

// [simd.comparison], basic_vec compare operators
//
// friend constexpr mask_type operator==(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr mask_type operator!=(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr mask_type operator>=(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr mask_type operator<=(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr mask_type operator>(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr mask_type operator<(const basic_vec&, const basic_vec&) noexcept;

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
  Vec a      = make_iota_vec<T, N>();
  Vec b(T{3});

  static_assert(cuda::std::is_same_v<decltype(a == b), Mask>);
  static_assert(cuda::std::is_same_v<decltype(a != b), Mask>);
  static_assert(cuda::std::is_same_v<decltype(a >= b), Mask>);
  static_assert(cuda::std::is_same_v<decltype(a <= b), Mask>);
  static_assert(cuda::std::is_same_v<decltype(a > b), Mask>);
  static_assert(cuda::std::is_same_v<decltype(a < b), Mask>);
  static_assert(noexcept(a == b));
  static_assert(noexcept(a != b));
  static_assert(noexcept(a >= b));
  static_assert(noexcept(a <= b));
  static_assert(noexcept(a > b));
  static_assert(noexcept(a < b));

  Mask eq = a == b;
  Mask ne = a != b;
  Mask ge = a >= b;
  Mask le = a <= b;
  Mask gt = a > b;
  Mask lt = a < b;
  for (int i = 0; i < N; ++i)
  {
    T val = static_cast<T>(i);
    assert(eq[i] == (val == T{3}));
    assert(ne[i] == (val != T{3}));
    assert(ge[i] == (val >= T{3}));
    assert(le[i] == (val <= T{3}));
    assert(gt[i] == (val > T{3}));
    assert(lt[i] == (val < T{3}));
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
