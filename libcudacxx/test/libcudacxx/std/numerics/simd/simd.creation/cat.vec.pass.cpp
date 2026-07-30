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

// [simd.creation], cat (basic_vec)
//
// template<class T, class A0, class... Abis> constexpr auto cat(const basic_vec<T,A0>&, const basic_vec<T,Abis>&...);

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// cat(basic_vec) - single argument

template <typename T>
TEST_FUNC constexpr void test_cat_one_vec()
{
  using Vec4 = simd::basic_vec<T, simd::fixed_size<4>>;

  Vec4 a(iota_generator<T, 0>{});

  auto result = simd::cat(a);
  static_assert(cuda::std::is_same_v<decltype(result), Vec4>);
  for (int i = 0; i < 4; ++i)
  {
    assert(result[i] == static_cast<T>(i));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// cat(basic_vec, basic_vec, basic_vec)

template <typename T>
TEST_FUNC constexpr void test_cat_three_vec()
{
  using Vec2 = simd::basic_vec<T, simd::fixed_size<2>>;
  using Vec6 = simd::basic_vec<T, simd::fixed_size<6>>;

  Vec2 a(iota_generator<T, 0>{});
  Vec2 b(iota_generator<T, 10>{});
  Vec2 c(iota_generator<T, 20>{});

  auto result = simd::cat(a, b, c);
  static_assert(cuda::std::is_same_v<decltype(result), Vec6>);
  for (int i = 0; i < 2; ++i)
  {
    assert(result[i] == static_cast<T>(i));
    assert(result[2 + i] == static_cast<T>(i + 10));
    assert(result[4 + i] == static_cast<T>(i + 20));
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
TEST_FUNC constexpr void test_type()
{
  test_cat_one_vec<T>();
  test_cat_three_vec<T>();
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
