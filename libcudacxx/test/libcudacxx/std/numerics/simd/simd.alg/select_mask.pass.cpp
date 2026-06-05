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

// [simd.alg], algorithms
//
// template<class T, class U> constexpr auto select(bool c, const T& a, const U& b)
//   -> remove_cvref_t<decltype(c ? a : b)>;
// template<size_t Bytes, class Abi, class T, class U> constexpr auto
//   select(const basic_mask<Bytes, Abi>& c, const T& a, const U& b) noexcept
//   -> decltype(__simd-select-impl(c, a, b));
//
// Covers the scalar overload and the basic_mask overloads that produce a mask
// (select(mask, mask, mask) and select(mask, bool, bool)).

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

TEST_FUNC constexpr void test_scalar_select()
{
  static_assert(cuda::std::is_same_v<decltype(simd::select(true, 1, 2)), int>);
  static_assert(cuda::std::is_same_v<decltype(simd::select(true, 1, 2.0)), double>);
  assert(simd::select(true, 1, 2) == 1);
  assert(simd::select(false, 1, 2) == 2);
}

template <cuda::std::size_t Bytes, int N>
TEST_FUNC constexpr void test_mask_bool_select()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Mask m(is_even{});

  static_assert(cuda::std::is_same_v<decltype(simd::select(m, true, false)), Mask>);
  static_assert(noexcept(simd::select(m, true, false)));

  Mask r1 = simd::select(m, true, false);
  Mask r2 = simd::select(m, false, true);
  for (int i = 0; i < N; ++i)
  {
    assert(r1[i] == (i % 2 == 0));
    assert(r2[i] == (i % 2 == 1));
  }
}

template <cuda::std::size_t Bytes, int N>
TEST_FUNC constexpr void test_mask_mask_select()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Mask m(is_even{});
  Mask all_true(true);
  Mask all_false(false);
  unused(all_true, all_false);

  static_assert(cuda::std::is_same_v<decltype(simd::select(m, all_true, all_false)), Mask>);
  static_assert(noexcept(simd::select(m, all_true, all_false)));

  Mask alt_a(is_even{});
  Mask r2 = simd::select(m, alt_a, all_false);
  for (int i = 0; i < N; ++i)
  {
    const bool expected = (i % 2 == 0);
    assert(r2[i] == expected);
  }
}

template <cuda::std::size_t Bytes>
TEST_FUNC constexpr void test_bytes()
{
  test_mask_mask_select<Bytes, 1>();
  test_mask_mask_select<Bytes, 4>();
  test_mask_bool_select<Bytes, 1>();
  test_mask_bool_select<Bytes, 4>();
}

TEST_FUNC constexpr bool test()
{
  test_scalar_select();
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
