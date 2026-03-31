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

using cuda::std::intptr_t;

template <class E>
using strides = cuda::std::array<intptr_t, E::rank()>;

template <class E1, class E2>
_CCCL_CONCEPT layout_stride_relaxed_mapping_comparable = _CCCL_REQUIRES_EXPR(
  (E1, E2), cuda::layout_stride_relaxed::mapping<E1> e1, cuda::layout_stride_relaxed::mapping<E2> e2)(
  static_cast<void>(e1 == e2));

template <class From, class To>
__host__ __device__ constexpr void test_comparison_different_rank()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // sanity check same rank
  static_assert(layout_stride_relaxed_mapping_comparable<cuda::std::extents<From, D>, cuda::std::extents<To, D>>);
  static_assert(layout_stride_relaxed_mapping_comparable<cuda::std::extents<From, 5>, cuda::std::extents<To, D>>);
  static_assert(layout_stride_relaxed_mapping_comparable<cuda::std::extents<From, D>, cuda::std::extents<To, 5>>);
  static_assert(layout_stride_relaxed_mapping_comparable<cuda::std::extents<From, 5>, cuda::std::extents<To, 5>>);

  // not equality comparable when rank is not the same
  static_assert(!layout_stride_relaxed_mapping_comparable<cuda::std::extents<From>, cuda::std::extents<To, D>>);
  static_assert(!layout_stride_relaxed_mapping_comparable<cuda::std::extents<From>, cuda::std::extents<To, 1>>);
  static_assert(!layout_stride_relaxed_mapping_comparable<cuda::std::extents<From, D>, cuda::std::extents<To>>);
  static_assert(!layout_stride_relaxed_mapping_comparable<cuda::std::extents<From, 1>, cuda::std::extents<To>>);
  static_assert(!layout_stride_relaxed_mapping_comparable<cuda::std::extents<From, D>, cuda::std::extents<To, D, D>>);
  static_assert(!layout_stride_relaxed_mapping_comparable<cuda::std::extents<From, 5>, cuda::std::extents<To, 5, D>>);
}

template <class To, class From>
__host__ __device__ constexpr void test_comparison(
  bool equal,
  To dest_exts,
  From src_exts,
  cuda::std::array<intptr_t, To::rank()> dest_strides,
  cuda::std::array<intptr_t, From::rank()> src_strides,
  intptr_t dest_offset = 0,
  intptr_t src_offset  = 0)
{
  using dest_mapping     = cuda::layout_stride_relaxed::mapping<To>;
  using src_mapping      = cuda::layout_stride_relaxed::mapping<From>;
  using strides_type_dst = typename dest_mapping::strides_type;
  using strides_type_src = typename src_mapping::strides_type;
  using offset_type_dest = typename dest_mapping::offset_type;
  using offset_type_src  = typename src_mapping::offset_type;
  auto dest_offseFrom    = static_cast<offset_type_dest>(dest_offset);
  auto src_offseFrom     = static_cast<offset_type_src>(src_offset);
  dest_mapping dest(dest_exts, strides_type_dst(dest_strides), dest_offseFrom);
  src_mapping src(src_exts, strides_type_src(src_strides), src_offseFrom);
  static_assert(noexcept(dest == src));
  assert((dest == src) == equal);
  assert((dest != src) == !equal);
}

template <class From, class To>
__host__ __device__ constexpr void test_comparison_same_rank()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  // Rank-0: same extents, default offsets
  test_comparison(
    true,
    cuda::std::extents<From>(),
    cuda::std::extents<To>(),
    cuda::std::array<intptr_t, 0>{},
    cuda::std::array<intptr_t, 0>{});

  // Rank-0: same extents, same non-zero offsets
  test_comparison(
    true,
    cuda::std::extents<From>(),
    cuda::std::extents<To>(),
    cuda::std::array<intptr_t, 0>{},
    cuda::std::array<intptr_t, 0>{},
    5,
    5);

  // Rank-0: same extents, different offsets
  test_comparison(
    false,
    cuda::std::extents<From>(),
    cuda::std::extents<To>(),
    cuda::std::array<intptr_t, 0>{},
    cuda::std::array<intptr_t, 0>{},
    5,
    10);

  // Rank-1: same extents and strides
  test_comparison(
    true,
    cuda::std::extents<From, D>(5),
    cuda::std::extents<To, D>(5),
    cuda::std::array<intptr_t, 1>{1},
    cuda::std::array<intptr_t, 1>{1});
  // Rank-1: same extents, different strides
  test_comparison(
    false,
    cuda::std::extents<From, D>(5),
    cuda::std::extents<To, D>(5),
    cuda::std::array<intptr_t, 1>{2},
    cuda::std::array<intptr_t, 1>{1});

  // Rank-1: different extents, same strides
  test_comparison(
    false,
    cuda::std::extents<From, D>(5),
    cuda::std::extents<To, D>(7),
    cuda::std::array<intptr_t, 1>{1},
    cuda::std::array<intptr_t, 1>{1});

  // Rank-1: same extents and strides, same non-zero offsets
  test_comparison(
    true,
    cuda::std::extents<From, D>(5),
    cuda::std::extents<To, D>(5),
    cuda::std::array<intptr_t, 1>{1},
    cuda::std::array<intptr_t, 1>{1},
    10,
    10);

  // Rank-1: same extents and strides, different offsets
  test_comparison(
    false,
    cuda::std::extents<From, D>(5),
    cuda::std::extents<To, D>(5),
    cuda::std::array<intptr_t, 1>{1},
    cuda::std::array<intptr_t, 1>{1},
    10,
    5);

  // Rank-1: same negative strides and offsets
  test_comparison(
    true,
    cuda::std::extents<From, 5>(),
    cuda::std::extents<To, 5>(),
    cuda::std::array<intptr_t, 1>{-1},
    cuda::std::array<intptr_t, 1>{-1},
    4,
    4);

  // Rank-1: different stride signs
  test_comparison(
    false,
    cuda::std::extents<From, 5>(),
    cuda::std::extents<To, 5>(),
    cuda::std::array<intptr_t, 1>{-1},
    cuda::std::array<intptr_t, 1>{1},
    4,
    0);

  // Rank-5: same extents and strides
  test_comparison(
    true,
    cuda::std::extents<From, D, D, D, D, D>(5, 6, 7, 8, 9),
    cuda::std::extents<To, D, D, D, D, D>(5, 6, 7, 8, 9),
    cuda::std::array<intptr_t, 5>{2, 20, 200, 2000, 20000},
    cuda::std::array<intptr_t, 5>{2, 20, 200, 2000, 20000});

  // Rank-5: same extents, swapped strides in last two dimensions
  test_comparison(
    false,
    cuda::std::extents<From, D, D, D, D, D>(5, 6, 7, 8, 9),
    cuda::std::extents<To, D, D, D, D, D>(5, 6, 7, 8, 9),
    cuda::std::array<intptr_t, 5>{2, 20, 200, 20000, 2000},
    cuda::std::array<intptr_t, 5>{2, 20, 200, 2000, 20000});
}

// Test comparison with standard layout mappings
template <class OtherLayout, class E1, class E2, class... OtherArgs>
__host__ __device__ constexpr void test_comparison_with(
  bool expect_equal,
  E1 e1,
  cuda::std::array<intptr_t, E1::rank()> strides,
  intptr_t offset,
  E2 e2,
  OtherArgs... other_args)
{
  using layout_type   = cuda::layout_stride_relaxed::mapping<E1>;
  using strides_type  = typename layout_type::strides_type;
  using offset_type   = typename layout_type::offset_type;
  using other_mapping = typename OtherLayout::template mapping<E2>;
  layout_type map(e1, strides_type(strides), static_cast<offset_type>(offset));
  other_mapping other_map(e2, other_args...);

  assert((map == other_map) == expect_equal);
}

template <class OtherLayout>
__host__ __device__ constexpr void test_comparison_with()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  constexpr bool is_left_based        = cuda::std::is_same_v<OtherLayout, cuda::std::layout_left>;

  // Rank-0: zero offset matches standard layout
  test_comparison_with<OtherLayout>(
    true, //
    cuda::std::extents<int>(),
    cuda::std::array<intptr_t, 0>{},
    0,
    cuda::std::extents<unsigned>());

  // Rank-1 static: unit stride, zero offset matches standard layout
  test_comparison_with<OtherLayout>(
    true, //
    cuda::std::extents<int, 5>(),
    cuda::std::array<intptr_t, 1>{1},
    0,
    cuda::std::extents<unsigned, 5>());

  // Rank-1 dynamic: unit stride, zero offset matches standard layout
  test_comparison_with<OtherLayout>(
    true, //
    cuda::std::extents<int, D>(5),
    cuda::std::array<intptr_t, 1>{1},
    0,
    cuda::std::extents<unsigned, 5>());

  // Rank-1 dynamic: non-unit stride does not match standard layout
  test_comparison_with<OtherLayout>(
    false, //
    cuda::std::extents<int, D>(5),
    cuda::std::array<intptr_t, 1>{2},
    0,
    cuda::std::extents<unsigned, 5>());

  // Rank-1 static: non-zero offset does not match standard layout
  test_comparison_with<OtherLayout>(
    false, //
    cuda::std::extents<int, 5>(),
    cuda::std::array<intptr_t, 1>{1},
    5,
    cuda::std::extents<unsigned, 5>());

  // Rank-2: column-major strides match only layout_left
  test_comparison_with<OtherLayout>(
    is_left_based,
    cuda::std::extents<int, D, D>(5, 7),
    cuda::std::array<intptr_t, 2>{1, 5},
    0,
    cuda::std::extents<unsigned, D, D>(5, 7));

  // Rank-2: row-major strides match only layout_right
  test_comparison_with<OtherLayout>(
    !is_left_based,
    cuda::std::extents<int, D, D>(5, 7),
    cuda::std::array<intptr_t, 2>{7, 1},
    0,
    cuda::std::extents<unsigned, D, D>(5, 7));
}

template <class From, class To>
__host__ __device__ constexpr void test_comparison_index_type()
{
  test_comparison_same_rank<From, To>();
  test_comparison_different_rank<From, To>();
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
