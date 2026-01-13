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

// template<class OtherMapping>
//   friend constexpr bool operator==(const mapping& x, const OtherMapping& y) noexcept;
//
// For layout_stride_relaxed to layout_stride_relaxed comparison:
//   Returns: true if x.extents() == y.extents(), x.offset() == y.offset(), and
//            each x.stride(r) == y.stride(r) for r in [0, rank()).
//
// For layout_stride_relaxed to other strided layout comparison:
//   Returns: true if x.extents() == y.extents(), x.offset() == OFFSET(y), and
//            each x.stride(r) == y.stride(r) for r in [0, rank()).

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../CustomTestLayouts.h"
#include "test_macros.h"

template <class E>
using strides = cuda::std::array<cuda::std::intptr_t, E::rank()>;

template <class E1, class E2>
_CCCL_CONCEPT layout_stride_relaxed_mapping_comparable = _CCCL_REQUIRES_EXPR(
  (E1, E2), cuda::layout_stride_relaxed::mapping<E1> e1, cuda::layout_stride_relaxed::mapping<E2> e2)(
  static_cast<void>(e1 == e2));

template <class T1, class T2>
__host__ __device__ constexpr void test_comparison_different_rank()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // sanity check same rank
  static_assert(layout_stride_relaxed_mapping_comparable<cuda::std::extents<T1, D>, cuda::std::extents<T2, D>>);
  static_assert(layout_stride_relaxed_mapping_comparable<cuda::std::extents<T1, 5>, cuda::std::extents<T2, D>>);
  static_assert(layout_stride_relaxed_mapping_comparable<cuda::std::extents<T1, D>, cuda::std::extents<T2, 5>>);
  static_assert(layout_stride_relaxed_mapping_comparable<cuda::std::extents<T1, 5>, cuda::std::extents<T2, 5>>);

  // not equality comparable when rank is not the same
  static_assert(!layout_stride_relaxed_mapping_comparable<cuda::std::extents<T1>, cuda::std::extents<T2, D>>);
  static_assert(!layout_stride_relaxed_mapping_comparable<cuda::std::extents<T1>, cuda::std::extents<T2, 1>>);
  static_assert(!layout_stride_relaxed_mapping_comparable<cuda::std::extents<T1, D>, cuda::std::extents<T2>>);
  static_assert(!layout_stride_relaxed_mapping_comparable<cuda::std::extents<T1, 1>, cuda::std::extents<T2>>);
  static_assert(!layout_stride_relaxed_mapping_comparable<cuda::std::extents<T1, D>, cuda::std::extents<T2, D, D>>);
  static_assert(!layout_stride_relaxed_mapping_comparable<cuda::std::extents<T1, 5>, cuda::std::extents<T2, 5, D>>);
}

template <class To, class From>
__host__ __device__ constexpr void test_comparison(
  bool equal,
  To dest_exts,
  From src_exts,
  cuda::std::array<cuda::std::intptr_t, To::rank()> dest_strides,
  cuda::std::array<cuda::std::intptr_t, From::rank()> src_strides,
  cuda::std::intptr_t dest_offset = 0,
  cuda::std::intptr_t src_offset  = 0)
{
  cuda::layout_stride_relaxed::mapping<To> dest(dest_exts, dest_strides, dest_offset);
  cuda::layout_stride_relaxed::mapping<From> src(src_exts, src_strides, src_offset);
  static_assert(noexcept(dest == src));
  assert((dest == src) == equal);
  assert((dest != src) == !equal);
}

template <class T1, class T2>
__host__ __device__ constexpr void test_comparison_same_rank()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // Rank-0 cases
  test_comparison(
    true,
    cuda::std::extents<T1>(),
    cuda::std::extents<T2>(),
    cuda::std::array<cuda::std::intptr_t, 0>{},
    cuda::std::array<cuda::std::intptr_t, 0>{});
  test_comparison(
    true,
    cuda::std::extents<T1>(),
    cuda::std::extents<T2>(),
    cuda::std::array<cuda::std::intptr_t, 0>{},
    cuda::std::array<cuda::std::intptr_t, 0>{},
    5,
    5);
  test_comparison(
    false,
    cuda::std::extents<T1>(),
    cuda::std::extents<T2>(),
    cuda::std::array<cuda::std::intptr_t, 0>{},
    cuda::std::array<cuda::std::intptr_t, 0>{},
    5,
    10);

  // Rank-1 cases
  test_comparison(
    true,
    cuda::std::extents<T1, D>(5),
    cuda::std::extents<T2, D>(5),
    cuda::std::array<cuda::std::intptr_t, 1>{1},
    cuda::std::array<cuda::std::intptr_t, 1>{1});
  test_comparison(
    false,
    cuda::std::extents<T1, D>(5),
    cuda::std::extents<T2, D>(5),
    cuda::std::array<cuda::std::intptr_t, 1>{2},
    cuda::std::array<cuda::std::intptr_t, 1>{1});
  test_comparison(
    false,
    cuda::std::extents<T1, D>(5),
    cuda::std::extents<T2, D>(7),
    cuda::std::array<cuda::std::intptr_t, 1>{1},
    cuda::std::array<cuda::std::intptr_t, 1>{1});

  // Cases with offset
  test_comparison(
    true,
    cuda::std::extents<T1, D>(5),
    cuda::std::extents<T2, D>(5),
    cuda::std::array<cuda::std::intptr_t, 1>{1},
    cuda::std::array<cuda::std::intptr_t, 1>{1},
    10,
    10);
  test_comparison(
    false,
    cuda::std::extents<T1, D>(5),
    cuda::std::extents<T2, D>(5),
    cuda::std::array<cuda::std::intptr_t, 1>{1},
    cuda::std::array<cuda::std::intptr_t, 1>{1},
    10,
    5);

  // Cases with negative strides
  test_comparison(
    true,
    cuda::std::extents<T1, 5>(),
    cuda::std::extents<T2, 5>(),
    cuda::std::array<cuda::std::intptr_t, 1>{-1},
    cuda::std::array<cuda::std::intptr_t, 1>{-1},
    4,
    4);
  test_comparison(
    false,
    cuda::std::extents<T1, 5>(),
    cuda::std::extents<T2, 5>(),
    cuda::std::array<cuda::std::intptr_t, 1>{-1},
    cuda::std::array<cuda::std::intptr_t, 1>{1},
    4,
    0);

  // Higher rank cases
  test_comparison(
    true,
    cuda::std::extents<T1, D, D, D, D, D>(5, 6, 7, 8, 9),
    cuda::std::extents<T2, D, D, D, D, D>(5, 6, 7, 8, 9),
    cuda::std::array<cuda::std::intptr_t, 5>{2, 20, 200, 2000, 20000},
    cuda::std::array<cuda::std::intptr_t, 5>{2, 20, 200, 2000, 20000});
  test_comparison(
    false,
    cuda::std::extents<T1, D, D, D, D, D>(5, 6, 7, 8, 9),
    cuda::std::extents<T2, D, D, D, D, D>(5, 6, 7, 8, 9),
    cuda::std::array<cuda::std::intptr_t, 5>{2, 20, 200, 20000, 2000},
    cuda::std::array<cuda::std::intptr_t, 5>{2, 20, 200, 2000, 20000});
}

// Test comparison with standard layout mappings
template <class OtherLayout, class E1, class E2, class... OtherArgs>
__host__ __device__ constexpr void test_comparison_with(
  bool expect_equal,
  E1 e1,
  cuda::std::array<cuda::std::intptr_t, E1::rank()> strides,
  cuda::std::intptr_t offset,
  E2 e2,
  OtherArgs... other_args)
{
  typename cuda::layout_stride_relaxed::template mapping<E1> map(e1, strides, offset);
  typename OtherLayout::template mapping<E2> other_map(e2, other_args...);

  assert((map == other_map) == expect_equal);
}

template <class OtherLayout>
__host__ __device__ constexpr void test_comparison_with()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  bool is_left_based                  = cuda::std::is_same_v<OtherLayout, cuda::std::layout_left>;

  // layout_stride_relaxed with zero offset should match standard layouts
  test_comparison_with<OtherLayout>(
    true, cuda::std::extents<int>(), cuda::std::array<cuda::std::intptr_t, 0>{}, 0, cuda::std::extents<unsigned>());
  test_comparison_with<OtherLayout>(
    true, cuda::std::extents<int, 5>(), cuda::std::array<cuda::std::intptr_t, 1>{1}, 0, cuda::std::extents<unsigned, 5>());
  test_comparison_with<OtherLayout>(
    true,
    cuda::std::extents<int, D>(5),
    cuda::std::array<cuda::std::intptr_t, 1>{1},
    0,
    cuda::std::extents<unsigned, 5>());
  test_comparison_with<OtherLayout>(
    false,
    cuda::std::extents<int, D>(5),
    cuda::std::array<cuda::std::intptr_t, 1>{2},
    0,
    cuda::std::extents<unsigned, 5>());

  // layout_stride_relaxed with non-zero offset should not match standard layouts
  test_comparison_with<OtherLayout>(
    false,
    cuda::std::extents<int, 5>(),
    cuda::std::array<cuda::std::intptr_t, 1>{1},
    5,
    cuda::std::extents<unsigned, 5>());

  // 2D cases
  test_comparison_with<OtherLayout>(
    is_left_based,
    cuda::std::extents<int, D, D>(5, 7),
    cuda::std::array<cuda::std::intptr_t, 2>{1, 5},
    0,
    cuda::std::extents<unsigned, D, D>(5, 7));
  test_comparison_with<OtherLayout>(
    !is_left_based,
    cuda::std::extents<int, D, D>(5, 7),
    cuda::std::array<cuda::std::intptr_t, 2>{7, 1},
    0,
    cuda::std::extents<unsigned, D, D>(5, 7));
}

template <class T1, class T2>
__host__ __device__ constexpr void test_comparison_index_type()
{
  test_comparison_same_rank<T1, T2>();
  test_comparison_different_rank<T1, T2>();
  test_comparison_with<cuda::std::layout_right>();
  test_comparison_with<cuda::std::layout_left>();
}

__host__ __device__ constexpr bool test()
{
  test_comparison_index_type<int, int>();
  test_comparison_index_type<int, size_t>();
  test_comparison_index_type<size_t, int>();
  test_comparison_index_type<size_t, long>();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
