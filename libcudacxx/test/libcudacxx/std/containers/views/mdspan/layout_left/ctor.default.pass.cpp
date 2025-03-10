//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// Test default construction:
//
// constexpr mapping() noexcept = default;

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/mdspan>

#include "test_macros.h"

template <class E, cuda::std::enable_if_t<E::rank() != 0, int> = 0>
__host__ __device__ constexpr void test_construction()
{
  using M = cuda::std::layout_left::mapping<E>;
  static_assert(noexcept(M{}));
  M m;
  E e;

  // check correct extents are returned
  static_assert(noexcept(m.extents()));
  assert(m.extents() == e);

  // check required_span_size()
  typename E::index_type expected_size = 1;
  for (typename E::rank_type r = 0; r < E::rank(); r++)
  {
    expected_size *= e.extent(r);
  }
  assert(m.required_span_size() == expected_size);
}

template <class E, cuda::std::enable_if_t<E::rank() == 0, int> = 0>
__host__ __device__ constexpr void test_construction()
{
  using M = cuda::std::layout_left::mapping<E>;
  static_assert(noexcept(M{}));
  M m;
  E e;

  // check correct extents are returned
  static_assert(noexcept(m.extents()));
  assert(m.extents() == e);
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_construction<cuda::std::extents<int>>();
  test_construction<cuda::std::extents<unsigned, D>>();
  test_construction<cuda::std::extents<unsigned, 7>>();
  test_construction<cuda::std::extents<unsigned, 7, 8>>();
  test_construction<cuda::std::extents<int64_t, D, 8, D, D>>();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
