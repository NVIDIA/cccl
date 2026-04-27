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

// [simd.mask.cassign], basic_mask compound assignment
//
// friend constexpr basic_mask& operator&=(basic_mask&, const basic_mask&) noexcept;
// friend constexpr basic_mask& operator|=(basic_mask&, const basic_mask&) noexcept;
// friend constexpr basic_mask& operator^=(basic_mask&, const basic_mask&) noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <int Bytes>
TEST_FUNC constexpr void test_and()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<4>>;
  Mask a(is_even{});
  Mask b(is_first_half{});

  static_assert(cuda::std::is_same_v<decltype(a &= b), Mask&>);
  static_assert(noexcept(a &= b));

  a &= b;
  assert(a[0] == true);
  assert(a[1] == false);
  assert(a[2] == false);
  assert(a[3] == false);
}

template <int Bytes>
TEST_FUNC constexpr void test_or()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<4>>;
  Mask a(is_even{});
  Mask b(is_first_half{});

  static_assert(cuda::std::is_same_v<decltype(a |= b), Mask&>);
  static_assert(noexcept(a |= b));

  a |= b;
  assert(a[0] == true);
  assert(a[1] == true);
  assert(a[2] == true);
  assert(a[3] == false);
}

template <int Bytes>
TEST_FUNC constexpr void test_xor()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<4>>;
  Mask a(is_even{});
  Mask b(is_first_half{});

  static_assert(cuda::std::is_same_v<decltype(a ^= b), Mask&>);
  static_assert(noexcept(a ^= b));

  a ^= b;
  assert(a[0] == false);
  assert(a[1] == true);
  assert(a[2] == true);
  assert(a[3] == false);
}

//----------------------------------------------------------------------------------------------------------------------

TEST_FUNC constexpr bool test()
{
  test_and<1>();
  test_and<2>();
  test_and<4>();
  test_and<8>();
#if _CCCL_HAS_INT128()
  test_and<16>();
#endif

  test_or<1>();
  test_or<2>();
  test_or<4>();
  test_or<8>();
#if _CCCL_HAS_INT128()
  test_or<16>();
#endif

  test_xor<1>();
  test_xor<2>();
  test_xor<4>();
  test_xor<8>();
#if _CCCL_HAS_INT128()
  test_xor<16>();
#endif
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
