//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/mdspan>

// Test offset functionality specific to layout_stride_relaxed:
//
// constexpr intptr_t offset() const noexcept;
//
// The offset allows accommodating negative strides, where the logical index 0
// maps to a position other than the beginning of the underlying storage.

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"

using cuda::std::intptr_t;

template <class E>
__host__ __device__ constexpr void
test_offset(E e, cuda::std::array<intptr_t, E::rank()> strides, intptr_t expected_offset)
{
  using M           = cuda::layout_stride_relaxed::mapping<E>;
  using offset_type = typename M::offset_type;
  M m(e, strides, static_cast<offset_type>(expected_offset));
  const M c_m = m;

  static_assert(noexcept(m.offset()));
  static_assert(noexcept(c_m.offset()));

  assert(m.offset() == expected_offset);
  assert(c_m.offset() == expected_offset);
}

// Test that offset is correctly used in index computation
template <class E, class... Indices>
__host__ __device__ constexpr void test_offset_in_indexing(
  E e,
  cuda::std::array<intptr_t, E::rank()> strides,
  intptr_t offset,
  typename E::index_type expected_at_zero,
  Indices... zero_indices)
{
  using M           = cuda::layout_stride_relaxed::mapping<E>;
  using offset_type = typename M::offset_type;
  M m(e, strides, static_cast<offset_type>(offset));

  // The result at indices (0, 0, ...) should equal the offset
  assert(m(zero_indices...) == expected_at_zero);
}

// Test reverse array pattern using negative stride
__host__ __device__ constexpr void test_reverse_array_pattern()
{
  // For a 1D array of size N with negative stride -1 and offset N-1,
  // we get a reverse iteration pattern:
  // logical index 0 -> physical N-1
  // logical index 1 -> physical N-2
  // ...
  // logical index N-1 -> physical 0

  using E           = cuda::std::extents<int, 5>;
  using M           = cuda::layout_stride_relaxed::mapping<E>;
  using offset_type = typename M::offset_type;

  cuda::std::array<intptr_t, 1> strides{-1};
  offset_type offset = 4; // N - 1

  M m(E{}, strides, offset);

  assert(m(0) == 4); // logical 0 -> physical 4
  assert(m(1) == 3); // logical 1 -> physical 3
  assert(m(2) == 2); // logical 2 -> physical 2
  assert(m(3) == 1); // logical 3 -> physical 1
  assert(m(4) == 0); // logical 4 -> physical 0
}

// Test 2D array with one reversed dimension
__host__ __device__ constexpr void test_2d_partial_reverse()
{
  // 2D array 3x4 where the first dimension is reversed
  // Layout: physical[offset - i*stride0 + j*stride1]
  //
  // With strides = {-4, 1} and offset = 8:
  // (0,0) -> 8 - 0*4 + 0*1 = 8
  // (1,0) -> 8 - 1*4 + 0*1 = 4
  // (2,0) -> 8 - 2*4 + 0*1 = 0
  // (0,1) -> 8 - 0*4 + 1*1 = 9

  using E           = cuda::std::extents<int, 3, 4>;
  using M           = cuda::layout_stride_relaxed::mapping<E>;
  using offset_type = typename M::offset_type;

  cuda::std::array<intptr_t, 2> strides{-4, 1};
  intptr_t offset = 8; // (rows-1) * |stride0|

  M m(E{}, strides, static_cast<offset_type>(offset));

  assert(m(0, 0) == 8);
  assert(m(1, 0) == 4);
  assert(m(2, 0) == 0);
  assert(m(0, 1) == 9);
  assert(m(2, 3) == 3);
}

// Test that default constructor has zero offset
__host__ __device__ constexpr void test_default_zero_offset()
{
  using E = cuda::std::extents<int, 4, 5>;
  using M = cuda::layout_stride_relaxed::mapping<E>;

  M m{};
  assert(m.offset() == 0);
}

// Test that copy constructor preserves offset
__host__ __device__ constexpr void test_copy_preserves_offset()
{
  using E           = cuda::std::extents<int, 4, 5>;
  using M           = cuda::layout_stride_relaxed::mapping<E>;
  using offset_type = typename M::offset_type;

  cuda::std::array<intptr_t, 2> strides{5, 1};
  offset_type offset = 42;

  M m1(E{}, strides, offset);
  M m2 = m1;

  assert(m2.offset() == offset);
  assert(m1.offset() == m2.offset());
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // Basic offset tests
  test_offset(cuda::std::extents<int>(), cuda::std::array<intptr_t, 0>{}, 0);
  test_offset(cuda::std::extents<int>(), cuda::std::array<intptr_t, 0>{}, 10);
  test_offset(cuda::std::extents<int, 5>(), cuda::std::array<intptr_t, 1>{1}, 0);
  test_offset(cuda::std::extents<int, 5>(), cuda::std::array<intptr_t, 1>{1}, 100);
  test_offset(cuda::std::extents<int, D>(7), cuda::std::array<intptr_t, 1>{2}, 50);
  test_offset(cuda::std::extents<int, 4, 5>(), cuda::std::array<intptr_t, 2>{5, 1}, 25);

  // Test offset in indexing
  test_offset_in_indexing(cuda::std::extents<int>(), cuda::std::array<intptr_t, 0>{}, 5, 5);
  test_offset_in_indexing(cuda::std::extents<int, 4>(), cuda::std::array<intptr_t, 1>{1}, 10, 10, 0);
  test_offset_in_indexing(cuda::std::extents<int, 3, 4>(), cuda::std::array<intptr_t, 2>{4, 1}, 20, 20, 0, 0);

  // Pattern tests
  test_reverse_array_pattern();
  test_2d_partial_reverse();
  test_default_zero_offset();
  test_copy_preserves_offset();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
