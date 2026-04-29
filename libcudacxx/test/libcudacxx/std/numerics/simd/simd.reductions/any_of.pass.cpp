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

// [simd.mask.reductions], any_of
//
// template<size_t Bytes, class Abi>
//   constexpr bool any_of(const basic_mask<Bytes, Abi>&) noexcept;
//
// constexpr bool any_of(same_as<bool> auto) noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <int Bytes, int N>
TEST_FUNC constexpr void test_any_of()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;

  static_assert(cuda::std::is_same_v<decltype(simd::any_of(Mask(true))), bool>);
  static_assert(noexcept(simd::any_of(Mask(true))));

  assert(simd::any_of(Mask(true)) == true);
  assert(simd::any_of(Mask(false)) == false);

  if constexpr (N > 1)
  {
    Mask even(is_even{});
    assert(simd::any_of(even) == true);
  }
}

TEST_FUNC constexpr void test_any_of_scalar_bool()
{
  static_assert(cuda::std::is_same_v<decltype(simd::any_of(true)), bool>);
  static_assert(noexcept(simd::any_of(true)));

  assert(simd::any_of(true) == true);
  assert(simd::any_of(false) == false);
}

template <int Bytes>
TEST_FUNC constexpr void test_bytes()
{
  test_any_of<Bytes, 1>();
  test_any_of<Bytes, 4>();
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
  test_any_of_scalar_bool();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
