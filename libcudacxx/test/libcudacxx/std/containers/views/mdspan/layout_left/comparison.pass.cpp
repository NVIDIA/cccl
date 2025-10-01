//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// template<class OtherExtents>
//   friend constexpr bool operator==(const mapping& x, const mapping<OtherExtents>& y) noexcept;
//                                      `
// Constraints: extents_type::rank() == OtherExtents::rank() is true.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class To, class From>
__host__ __device__ constexpr void test_comparison(bool equal, To dest_exts, From src_exts)
{
  cuda::std::layout_left::mapping<To> dest(dest_exts);
  cuda::std::layout_left::mapping<From> src(src_exts);
  static_assert(noexcept(dest == src));
  assert((dest == src) == equal);
  assert((dest != src) == !equal);
}

template <class E1, class E2>
_CCCL_CONCEPT can_compare_layouts = _CCCL_REQUIRES_EXPR((E1, E2), E1 e1, E2 e2)(
  static_cast<void>(cuda::std::layout_left::mapping<E1>(e1) == cuda::std::layout_left::mapping<E2>(e2)));

struct X
{
  __host__ __device__ constexpr bool does_not_match()
  {
    return true;
  }
};

template <class E1, class E2, cuda::std::enable_if_t<!can_compare_layouts<E1, E2>, int> = 0>
__host__ __device__ constexpr X compare_layout_mappings(E1, E2)
{
  return {};
}

template <class E1, class E2, cuda::std::enable_if_t<can_compare_layouts<E1, E2>, int> = 0>
__host__ __device__ constexpr auto compare_layout_mappings(E1, E2)
{
  return true;
}

template <class T1, class T2>
__host__ __device__ constexpr void test_comparison_different_rank()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // sanity check same rank
  static_assert(compare_layout_mappings(cuda::std::extents<T1, D>(5), cuda::std::extents<T2, D>(5)), "");
  static_assert(compare_layout_mappings(cuda::std::extents<T1, 5>(), cuda::std::extents<T2, D>(5)), "");
  static_assert(compare_layout_mappings(cuda::std::extents<T1, D>(5), cuda::std::extents<T2, 5>()), "");
  static_assert(compare_layout_mappings(cuda::std::extents<T1, 5>(), cuda::std::extents<T2, 5>()), "");

  // not equality comparable when rank is not the same
  static_assert(compare_layout_mappings(cuda::std::extents<T1>(), cuda::std::extents<T2, D>(1)).does_not_match(), "");
  static_assert(compare_layout_mappings(cuda::std::extents<T1>(), cuda::std::extents<T2, 1>()).does_not_match(), "");

  static_assert(compare_layout_mappings(cuda::std::extents<T1, D>(1), cuda::std::extents<T2>()).does_not_match(), "");
  static_assert(compare_layout_mappings(cuda::std::extents<T1, 1>(), cuda::std::extents<T2>()).does_not_match(), "");

  static_assert(
    compare_layout_mappings(cuda::std::extents<T1, D>(5), cuda::std::extents<T2, D, D>(5, 5)).does_not_match(), "");
  static_assert(compare_layout_mappings(cuda::std::extents<T1, 5>(), cuda::std::extents<T2, 5, D>(5)).does_not_match(),
                "");
  static_assert(compare_layout_mappings(cuda::std::extents<T1, 5>(), cuda::std::extents<T2, 5, 1>()).does_not_match(),
                "");

  static_assert(
    compare_layout_mappings(cuda::std::extents<T1, D, D>(5, 5), cuda::std::extents<T2, D>(5)).does_not_match(), "");
  static_assert(compare_layout_mappings(cuda::std::extents<T1, 5, D>(5), cuda::std::extents<T2, D>(5)).does_not_match(),
                "");
  static_assert(compare_layout_mappings(cuda::std::extents<T1, 5, 5>(), cuda::std::extents<T2, 5>()).does_not_match(),
                "");
}

template <class T1, class T2>
__host__ __device__ constexpr void test_comparison_same_rank()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  test_comparison(true, cuda::std::extents<T1>(), cuda::std::extents<T2>());

  test_comparison(true, cuda::std::extents<T1, D>(5), cuda::std::extents<T2, D>(5));
  test_comparison(true, cuda::std::extents<T1, 5>(), cuda::std::extents<T2, D>(5));
  test_comparison(true, cuda::std::extents<T1, D>(5), cuda::std::extents<T2, 5>());
  test_comparison(true, cuda::std::extents<T1, 5>(), cuda::std::extents<T2, 5>());
  test_comparison(false, cuda::std::extents<T1, D>(5), cuda::std::extents<T2, D>(7));
  test_comparison(false, cuda::std::extents<T1, 5>(), cuda::std::extents<T2, D>(7));
  test_comparison(false, cuda::std::extents<T1, D>(5), cuda::std::extents<T2, 7>());
  test_comparison(false, cuda::std::extents<T1, 5>(), cuda::std::extents<T2, 7>());

  test_comparison(
    true, cuda::std::extents<T1, D, D, D, D, D>(5, 6, 7, 8, 9), cuda::std::extents<T2, D, D, D, D, D>(5, 6, 7, 8, 9));
  test_comparison(true, cuda::std::extents<T1, D, 6, D, 8, D>(5, 7, 9), cuda::std::extents<T2, 5, D, D, 8, 9>(6, 7));
  test_comparison(true, cuda::std::extents<T1, 5, 6, 7, 8, 9>(5, 6, 7, 8, 9), cuda::std::extents<T2, 5, 6, 7, 8, 9>());
  test_comparison(
    false, cuda::std::extents<T1, D, D, D, D, D>(5, 6, 7, 8, 9), cuda::std::extents<T2, D, D, D, D, D>(5, 6, 3, 8, 9));
  test_comparison(false, cuda::std::extents<T1, D, 6, D, 8, D>(5, 7, 9), cuda::std::extents<T2, 5, D, D, 3, 9>(6, 7));
  test_comparison(false, cuda::std::extents<T1, 5, 6, 7, 8, 9>(5, 6, 7, 8, 9), cuda::std::extents<T2, 5, 6, 7, 3, 9>());
}

template <class T1, class T2>
__host__ __device__ constexpr void test_comparison()
{
  test_comparison_same_rank<T1, T2>();
  test_comparison_different_rank<T1, T2>();
}

__host__ __device__ constexpr bool test()
{
  test_comparison<int, int>();
  test_comparison<int, size_t>();
  test_comparison<size_t, int>();
  test_comparison<size_t, long>();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
