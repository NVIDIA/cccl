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

// [simd.mask.subscr], basic_mask subscript operators
//
// constexpr value_type operator[](simd-size-type) const noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// subscript read-back

template <int Bytes, int N>
TEST_FUNC constexpr void test_subscript()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Mask mask(true);

  static_assert(cuda::std::is_same_v<decltype(mask[0]), typename Mask::value_type>);
  static_assert(noexcept(mask[0]));
  static_assert(is_const_member_function_v<decltype(&Mask::operator[])>);
  unused(mask);

  Mask all_true(true);
  Mask all_false(false);
  Mask alternating(is_even{});
  for (int i = 0; i < N; ++i)
  {
    assert(all_true[i] == true);
    assert(all_false[i] == false);
    assert(alternating[i] == (i % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <int Bytes>
TEST_FUNC constexpr void test_bytes()
{
  test_subscript<Bytes, 1>();
  test_subscript<Bytes, 4>();
}

TEST_FUNC constexpr bool test()
{
  test_bytes<1>();
  test_bytes<2>();
  test_bytes<4>();
  test_bytes<8>();
#if _CCCL_HAS_INT128()
  test_bytes<16>();
#endif
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
