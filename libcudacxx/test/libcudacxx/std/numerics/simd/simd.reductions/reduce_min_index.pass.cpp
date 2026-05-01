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

// [simd.mask.reductions], reduce_min_index
//
// template<size_t Bytes, class Abi>
//   constexpr simd-size-type reduce_min_index(const basic_mask<Bytes, Abi>&);
//
// constexpr simd-size-type reduce_min_index(same_as<bool> auto);

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <int Bytes, int N>
TEST_FUNC constexpr void test_reduce_min_index()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  assert(simd::reduce_min_index(Mask(true)) == 0);

  if constexpr (N > 1)
  {
    Mask last_only(is_index<N - 1>{});
    assert(simd::reduce_min_index(last_only) == N - 1);

    Mask even(is_even{});
    assert(simd::reduce_min_index(even) == 0);
  }
  if constexpr (N >= 4)
  {
    Mask upper_half(is_greater_equal_than_index<N / 2>{});
    assert(simd::reduce_min_index(upper_half) == N / 2);
  }
}

TEST_FUNC constexpr void test_reduce_min_index_scalar_bool()
{
  assert(simd::reduce_min_index(true) == 0);
}

template <int Bytes>
TEST_FUNC constexpr void test_bytes()
{
  test_reduce_min_index<Bytes, 1>();
  test_reduce_min_index<Bytes, 4>();
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
  test_reduce_min_index_scalar_bool();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
