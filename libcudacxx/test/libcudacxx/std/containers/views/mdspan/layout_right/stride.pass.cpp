//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// constexpr index_type stride(rank_type i) const noexcept;
//
//   Constraints: extents_type::rank() > 0 is true.
//
//   Preconditions: i < extents_type::rank() is true.
//
//   Returns: extents().rev-prod-of-extents(i).

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/mdspan>

#include "test_macros.h"

template <class E, class... Args>
__host__ __device__ constexpr void test_stride(cuda::std::array<typename E::index_type, E::rank()> strides, Args... args)
{
  using M = cuda::std::layout_right::mapping<E>;
  M m(E(args...));

  ASSERT_NOEXCEPT(m.stride(0));
  for (size_t r = 0; r < E::rank(); r++)
  {
    assert(strides[r] == m.stride(r));
  }
}

__host__ __device__ constexpr bool test()
{
  constexpr size_t D = cuda::std::dynamic_extent;
  test_stride<cuda::std::extents<unsigned, D>>(cuda::std::array<unsigned, 1>{1}, 7);
  test_stride<cuda::std::extents<unsigned, 7>>(cuda::std::array<unsigned, 1>{1});
  test_stride<cuda::std::extents<unsigned, 7, 8>>(cuda::std::array<unsigned, 2>{8, 1});
  test_stride<cuda::std::extents<int64_t, D, 8, D, D>>(cuda::std::array<int64_t, 4>{720, 90, 10, 1}, 7, 9, 10);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
