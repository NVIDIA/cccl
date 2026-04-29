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

// [simd.mask.reductions], none_of
//
// template<size_t Bytes, class Abi>
//   constexpr bool none_of(const basic_mask<Bytes, Abi>&) noexcept;
//
// constexpr bool none_of(same_as<bool> auto) noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <int Bytes, int N>
TEST_FUNC constexpr void test_none_of()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;

  static_assert(cuda::std::is_same_v<decltype(simd::none_of(Mask(true))), bool>);
  static_assert(noexcept(simd::none_of(Mask(true))));

  Mask all_true(true);
  Mask all_false(false);
  assert(simd::none_of(all_true) == false);
  assert(simd::none_of(all_false) == true);
  assert(simd::none_of(all_true) == !simd::any_of(all_true));
  assert(simd::none_of(all_false) == !simd::any_of(all_false));

  if constexpr (N > 1)
  {
    Mask even(is_even{});
    assert(simd::none_of(even) == false);
    assert(simd::none_of(even) == !simd::any_of(even));
  }
}

TEST_FUNC constexpr void test_none_of_scalar_bool()
{
  static_assert(cuda::std::is_same_v<decltype(simd::none_of(true)), bool>);
  static_assert(noexcept(simd::none_of(true)));

  assert(simd::none_of(true) == false);
  assert(simd::none_of(false) == true);
}

template <int Bytes>
TEST_FUNC constexpr void test_bytes()
{
  test_none_of<Bytes, 1>();
  test_none_of<Bytes, 4>();
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
  test_none_of_scalar_bool();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
