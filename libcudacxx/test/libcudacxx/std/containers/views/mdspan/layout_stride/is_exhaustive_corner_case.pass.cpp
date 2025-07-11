//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// constexpr bool is_exhaustive() const noexcept;
//
// Returns:
//   - true if rank_ is 0.
//   - Otherwise, true if there is a permutation P of the integers in the range [0, rank_) such that
//     stride(p0) equals 1, and stride(pi) equals stride(pi_1) * extents().extent(pi_1) for i in the
//     range [1, rank_), where pi is the ith element of P.
//   - Otherwise, false.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class E>
__host__ __device__ constexpr void
test_layout_mapping_stride(E ext, cuda::std::array<typename E::index_type, E::rank()> strides, bool exhaustive)
{
  using M = cuda::std::layout_stride::template mapping<E>;
  M m(ext, strides);
  assert(m.is_exhaustive() == exhaustive);
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_layout_mapping_stride(cuda::std::extents<int, 0>(), cuda::std::array<int, 1>{1}, true);
  test_layout_mapping_stride(cuda::std::extents<unsigned, D>(0), cuda::std::array<unsigned, 1>{3}, false);
  test_layout_mapping_stride(cuda::std::extents<int, 0, 3>(), cuda::std::array<int, 2>{6, 2}, true);
  test_layout_mapping_stride(cuda::std::extents<int, D, D>(3, 0), cuda::std::array<int, 2>{6, 2}, false);
  test_layout_mapping_stride(cuda::std::extents<int, D, D>(0, 0), cuda::std::array<int, 2>{6, 2}, false);
  test_layout_mapping_stride(
    cuda::std::extents<unsigned, D, D, D, D>(3, 3, 0, 3), cuda::std::array<unsigned, 4>{3, 1, 27, 9}, true);
  test_layout_mapping_stride(
    cuda::std::extents<int, D, D, D, D>(0, 3, 3, 3), cuda::std::array<int, 4>{3, 1, 27, 9}, false);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
