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

// Test properties of layout_stride_relaxed::mapping:
//
//     static constexpr bool is_always_unique() noexcept { return false; }
//     static constexpr bool is_always_exhaustive() noexcept { return false; }
//     static constexpr bool is_always_strided() noexcept { return false; }
//
//     constexpr bool is_unique() noexcept;  // true for rank 0, empty mappings, and positive unique strides
//     constexpr bool is_exhaustive() noexcept { return false; }  // conservative
//     constexpr bool is_strided() noexcept { return offset_ == 0; }

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::intptr_t;

template <class E>
TEST_FUNC constexpr void test_layout_mapping_stride_relaxed(
  E ext,
  [[maybe_unused]] cuda::std::array<intptr_t, E::rank()> input_strides,
  intptr_t offset,
  bool expected_is_strided,
  bool expected_is_unique)
{
  using M            = cuda::layout_stride_relaxed::template mapping<E>;
  using strides_type = typename M::strides_type;
  using offset_type  = typename M::offset_type;
  M m(ext, strides_type(input_strides), static_cast<offset_type>(offset));
  const M c_m = m;

  assert(m.strides() == strides_type(input_strides));
  assert(c_m.strides() == strides_type(input_strides));
  assert(m.extents() == ext);
  assert(c_m.extents() == ext);
  assert(cuda::std::cmp_equal(m.offset(), offset));
  assert(cuda::std::cmp_equal(c_m.offset(), offset));

  // layout_stride_relaxed: is_always_* are all false (conservative)
  assert(M::is_always_unique() == false);
  assert(M::is_always_exhaustive() == false);
  assert(M::is_always_strided() == false);

  // is_unique is true for rank 0, empty mappings, and positive unique strides
  assert(m.is_unique() == expected_is_unique);
  assert(c_m.is_unique() == expected_is_unique);
  assert(m.is_exhaustive() == false);
  assert(c_m.is_exhaustive() == false);

  // is_strided is true only if offset is zero
  assert(m.is_strided() == expected_is_strided);
  assert(c_m.is_strided() == expected_is_strided);

  static_assert(noexcept(m.strides()));
  static_assert(noexcept(c_m.strides()));
  static_assert(noexcept(m.extents()));
  static_assert(noexcept(c_m.extents()));
  static_assert(noexcept(m.offset()));
  static_assert(noexcept(c_m.offset()));
  static_assert(noexcept(M::is_always_unique()));
  static_assert(noexcept(M::is_always_exhaustive()));
  static_assert(noexcept(M::is_always_strided()));
  static_assert(noexcept(m.is_unique()));
  static_assert(noexcept(c_m.is_unique()));
  static_assert(noexcept(m.is_exhaustive()));
  static_assert(noexcept(c_m.is_exhaustive()));
  static_assert(noexcept(m.is_strided()));
  static_assert(noexcept(c_m.is_strided()));

  if constexpr (E::rank() > 0)
  {
    for (typename E::rank_type r = 0; r < E::rank(); r++)
    {
      assert(cuda::std::cmp_equal(m.stride(r), input_strides[r]));
      assert(cuda::std::cmp_equal(c_m.stride(r), input_strides[r]));
      static_assert(noexcept(m.stride(r)));
      static_assert(noexcept(c_m.stride(r)));
    }
  }

  static_assert(cuda::std::is_trivially_copyable_v<M>);
}

TEST_FUNC constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // Rank-0 cases
  test_layout_mapping_stride_relaxed(cuda::std::extents<int>(), cuda::std::array<intptr_t, 0>{}, 0, true, true);
  test_layout_mapping_stride_relaxed(cuda::std::extents<int>(), cuda::std::array<intptr_t, 0>{}, 5, false, true);

  // Basic cases with zero offset (is_strided = true)
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, 2, 3>(), cuda::std::array<intptr_t, 2>{1, 2}, 0, true, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<signed char, 4, 5>(), cuda::std::array<intptr_t, 2>{1, 4}, 0, true, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<unsigned, D, 4>(7), cuda::std::array<intptr_t, 2>{20, 2}, 0, true, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<size_t, D, D, D, D>(3, 3, 3, 3), cuda::std::array<intptr_t, 4>{3, 1, 9, 27}, 0, true, true);

  // Cases with non-zero offset (is_strided = false)
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<signed char, 4, 5>(), cuda::std::array<intptr_t, 2>{1, 4}, 10, false, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<unsigned, D, 4>(7), cuda::std::array<intptr_t, 2>{20, 2}, 5, false, true);

  // Cases with negative strides
  test_layout_mapping_stride_relaxed(cuda::std::extents<int, 4>(), cuda::std::array<intptr_t, 1>{-1}, 3, false, false);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, 4, 5>(), cuda::std::array<intptr_t, 2>{-1, 4}, 3, false, false);

  // Cases with zero strides (broadcasting)
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, 4, 5>(), cuda::std::array<intptr_t, 2>{0, 1}, 0, true, false);

  // ============================================================================
  // Edge cases with zero extents
  // ============================================================================

  // Single zero extent (dynamic)
  test_layout_mapping_stride_relaxed(cuda::std::extents<int, D>(0), cuda::std::array<intptr_t, 1>{1}, 0, true, true);
  test_layout_mapping_stride_relaxed(cuda::std::extents<int, D>(0), cuda::std::array<intptr_t, 1>{1}, 5, false, true);

  // Single zero extent (static)
  test_layout_mapping_stride_relaxed(cuda::std::extents<int, 0>(), cuda::std::array<intptr_t, 1>{1}, 0, true, true);
  test_layout_mapping_stride_relaxed(cuda::std::extents<int, 0>(), cuda::std::array<intptr_t, 1>{1}, 10, false, true);

  // All extents zero (multiple dimensions)
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, D, D>(0, 0), cuda::std::array<intptr_t, 2>{1, 1}, 0, true, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, 0, 0>(), cuda::std::array<intptr_t, 2>{8, 1}, 0, true, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, D, D, D>(0, 0, 0), cuda::std::array<intptr_t, 3>{1, 1, 1}, 0, true, true);

  // All extents zero with non-zero offset
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, D, D>(0, 0), cuda::std::array<intptr_t, 2>{1, 1}, 100, false, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, 0, 0>(), cuda::std::array<intptr_t, 2>{8, 1}, 50, false, true);

  // Zero extent with negative strides
  test_layout_mapping_stride_relaxed(cuda::std::extents<int, D>(0), cuda::std::array<intptr_t, 1>{-1}, 0, true, true);
  test_layout_mapping_stride_relaxed(cuda::std::extents<int, D>(0), cuda::std::array<intptr_t, 1>{-1}, 10, false, true);
  test_layout_mapping_stride_relaxed(cuda::std::extents<int, 0>(), cuda::std::array<intptr_t, 1>{-5}, 0, true, true);

  // Mix of zero and non-zero extents
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, D, D>(0, 5), cuda::std::array<intptr_t, 2>{5, 1}, 0, true, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, D, D>(5, 0), cuda::std::array<intptr_t, 2>{1, 5}, 0, true, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, 0, 5>(), cuda::std::array<intptr_t, 2>{5, 1}, 0, true, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, 5, 0>(), cuda::std::array<intptr_t, 2>{1, 5}, 0, true, true);

  // Mix of zero and non-zero extents with negative strides
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, D, D>(0, 5), cuda::std::array<intptr_t, 2>{-1, 1}, 0, true, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, D, D>(5, 0), cuda::std::array<intptr_t, 2>{-1, 1}, 4, false, true);

  // Zero extent in the middle of non-zero extents
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, D, D, D>(3, 0, 4), cuda::std::array<intptr_t, 3>{12, 4, 1}, 0, true, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, 3, 0, 4>(), cuda::std::array<intptr_t, 3>{12, 4, 1}, 0, true, true);

  // Zero extent with zero stride (broadcasting an empty dimension)
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, D, D>(0, 5), cuda::std::array<intptr_t, 2>{0, 1}, 0, true, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, D, D>(5, 0), cuda::std::array<intptr_t, 2>{1, 0}, 0, true, true);

  // Higher rank with multiple zero extents
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int64_t, D, 0, D, 0>(5, 7), cuda::std::array<intptr_t, 4>{1, 2, 3, 4}, 0, true, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int64_t, 0, D, 0, D>(5, 7), cuda::std::array<intptr_t, 4>{1, 2, 3, 4}, 50, false, true);

  // Zero extent with mixed positive, negative, and zero strides
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, D, D, D>(0, 5, 3), cuda::std::array<intptr_t, 3>{-1, 0, 1}, 0, true, true);
  test_layout_mapping_stride_relaxed(
    cuda::std::extents<int, D, D, D>(5, 0, 3), cuda::std::array<intptr_t, 3>{-1, 0, 1}, 4, false, true);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
