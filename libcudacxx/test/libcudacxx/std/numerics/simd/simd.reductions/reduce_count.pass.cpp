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

// [simd.mask.reductions], reduce_count
//
// template<size_t Bytes, class Abi>
//   constexpr simd-size-type reduce_count(const basic_mask<Bytes, Abi>&) noexcept;
//
// constexpr simd-size-type reduce_count(same_as<bool> auto) noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <int Bytes, int N>
TEST_FUNC constexpr void test_reduce_count()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;

  static_assert(noexcept(simd::reduce_count(Mask(true))));

  assert(simd::reduce_count(Mask(true)) == N);
  assert(simd::reduce_count(Mask(false)) == 0);

  if constexpr (N > 1)
  {
    Mask even(is_even{});
    int expected = N / 2;
    assert(simd::reduce_count(even) == expected);
  }
}

TEST_FUNC constexpr void test_reduce_count_scalar_bool()
{
  static_assert(noexcept(simd::reduce_count(true)));

  assert(simd::reduce_count(true) == 1);
  assert(simd::reduce_count(false) == 0);
}

template <int Bytes>
TEST_FUNC constexpr void test_bytes()
{
  test_reduce_count<Bytes, 1>();
  test_reduce_count<Bytes, 4>();
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
  test_reduce_count_scalar_bool();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
