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

// [simd.creation], cat (basic_mask)
//
// template<size_t Bytes, class A0, class... Abis> constexpr auto cat(const basic_mask<Bytes,A0>&,
//                                                                    const basic_mask<Bytes,Abis>&...);

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// cat(basic_mask) - single argument

template <typename T>
TEST_FUNC constexpr void test_cat_one_mask()
{
  constexpr cuda::std::size_t Bytes = sizeof(T);
  using Mask4                       = simd::basic_mask<Bytes, simd::fixed_size<4>>;

  Mask4 a(is_even{}); // T,F,T,F
  auto result = simd::cat(a);
  static_assert(cuda::std::is_same_v<decltype(result), Mask4>);
  for (int i = 0; i < 4; ++i)
  {
    assert(result[i] == (i % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// cat(basic_mask, basic_mask, basic_mask)

template <typename T>
TEST_FUNC constexpr void test_cat_three_mask()
{
  constexpr cuda::std::size_t Bytes = sizeof(T);
  using Mask2                       = simd::basic_mask<Bytes, simd::fixed_size<2>>;
  using Mask6                       = simd::basic_mask<Bytes, simd::fixed_size<6>>;

  Mask2 a(false);
  Mask2 b(true);
  Mask2 c(false);
  auto result = simd::cat(a, b, c);
  static_assert(cuda::std::is_same_v<decltype(result), Mask6>);
  for (int i = 0; i < 2; ++i)
  {
    assert(result[i] == false);
    assert(result[2 + i] == true);
    assert(result[4 + i] == false);
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
TEST_FUNC constexpr void test_type()
{
  test_cat_one_mask<T>();
  test_cat_three_mask<T>();
}

TEST_FUNC constexpr bool test()
{
  test_type<cuda::std::int16_t>();
  test_type<float>();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
