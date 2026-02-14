//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// template<class OtherMapping>
//   friend constexpr bool operator==(const mapping& x, const OtherMapping& y) noexcept;
//
// Constraints:
//   - layout-mapping-alike<OtherMapping> is satisfied.
//   - rank_ == OtherMapping::extents_type::rank() is true.
//   - OtherMapping::is_always_strided() is true.
//
// Preconditions: OtherMapping meets the layout mapping requirements ([mdspan.layout.policy.reqmts]).
//
// Returns: true if x.extents() == y.extents() is true, OFFSET(y) == 0 is true, and each of x.stride(r) == y.stride(r)
// is true for r in the range [0, x.extents().rank()). Otherwise, false.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "../CustomTestLayouts.h"
#include "test_macros.h"

template <class E>
using strides = cuda::std::array<typename E::index_type, E::rank()>;

template <class E1, class E2>
_CCCL_CONCEPT layout_mapping_comparable =
  _CCCL_REQUIRES_EXPR((E1, E2), cuda::std::layout_stride::mapping<E1> e1, cuda::std::layout_stride::mapping<E2> e2)(
    static_cast<void>(e1 == e2));

template <class T1, class T2>
__host__ __device__ constexpr void test_comparison_different_rank()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // sanity check same rank
  static_assert(layout_mapping_comparable<cuda::std::extents<T1, D>, cuda::std::extents<T2, D>>);
  static_assert(layout_mapping_comparable<cuda::std::extents<T1, 5>, cuda::std::extents<T2, D>>);
  static_assert(layout_mapping_comparable<cuda::std::extents<T1, D>, cuda::std::extents<T2, 5>>);
  static_assert(layout_mapping_comparable<cuda::std::extents<T1, 5>, cuda::std::extents<T2, 5>>);

  // not equality comparable when rank is not the same
  static_assert(!layout_mapping_comparable<cuda::std::extents<T1>, cuda::std::extents<T2, D>>);
  static_assert(!layout_mapping_comparable<cuda::std::extents<T1>, cuda::std::extents<T2, 1>>);
  static_assert(!layout_mapping_comparable<cuda::std::extents<T1, D>, cuda::std::extents<T2>>);
  static_assert(!layout_mapping_comparable<cuda::std::extents<T1, 1>, cuda::std::extents<T2>>);
  static_assert(!layout_mapping_comparable<cuda::std::extents<T1, D>, cuda::std::extents<T2, D, D>>);
  static_assert(!layout_mapping_comparable<cuda::std::extents<T1, 5>, cuda::std::extents<T2, 5, D>>);
  static_assert(!layout_mapping_comparable<cuda::std::extents<T1, 5>, cuda::std::extents<T2, 5, 1>>);
  static_assert(!layout_mapping_comparable<cuda::std::extents<T1, D, D>, cuda::std::extents<T2, D>>);
  static_assert(!layout_mapping_comparable<cuda::std::extents<T1, 5, D>, cuda::std::extents<T2, 5>>);
  static_assert(!layout_mapping_comparable<cuda::std::extents<T1, 5, 1>, cuda::std::extents<T2, 5>>);
}

template <class To, class From>
__host__ __device__ constexpr void test_comparison(
  bool equal,
  To dest_exts,
  From src_exts,
  cuda::std::array<int, To::rank()> dest_strides,
  cuda::std::array<int, From::rank()> src_strides)
{
  cuda::std::layout_stride::mapping<To> dest(dest_exts, dest_strides);
  cuda::std::layout_stride::mapping<From> src(src_exts, src_strides);
  static_assert(noexcept(dest == src));
  assert((dest == src) == equal);
  assert((dest != src) == !equal);
}

template <class T1, class T2>
__host__ __device__ constexpr void test_comparison_same_rank()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  test_comparison(
    true, cuda::std::extents<T1>(), cuda::std::extents<T2>(), cuda::std::array<int, 0>{}, cuda::std::array<int, 0>{});

  test_comparison(
    true,
    cuda::std::extents<T1, D>(5),
    cuda::std::extents<T2, D>(5),
    cuda::std::array<int, 1>{1},
    cuda::std::array<int, 1>{1});
  test_comparison(
    true,
    cuda::std::extents<T1, D>(0),
    cuda::std::extents<T2, D>(0),
    cuda::std::array<int, 1>{1},
    cuda::std::array<int, 1>{1});
  test_comparison(
    true,
    cuda::std::extents<T1, 5>(),
    cuda::std::extents<T2, D>(5),
    cuda::std::array<int, 1>{3},
    cuda::std::array<int, 1>{3});
  test_comparison(
    true,
    cuda::std::extents<T1, D>(5),
    cuda::std::extents<T2, 5>(),
    cuda::std::array<int, 1>{1},
    cuda::std::array<int, 1>{1});
  test_comparison(
    true,
    cuda::std::extents<T1, 5>(),
    cuda::std::extents<T2, 5>(),
    cuda::std::array<int, 1>{1},
    cuda::std::array<int, 1>{1});
  test_comparison(
    false,
    cuda::std::extents<T1, 5>(),
    cuda::std::extents<T2, D>(5),
    cuda::std::array<int, 1>{2},
    cuda::std::array<int, 1>{1});
  test_comparison(
    false,
    cuda::std::extents<T1, D>(5),
    cuda::std::extents<T2, D>(5),
    cuda::std::array<int, 1>{2},
    cuda::std::array<int, 1>{1});
  test_comparison(
    false,
    cuda::std::extents<T1, D>(5),
    cuda::std::extents<T2, D>(7),
    cuda::std::array<int, 1>{1},
    cuda::std::array<int, 1>{1});
  test_comparison(
    false,
    cuda::std::extents<T1, 5>(),
    cuda::std::extents<T2, D>(7),
    cuda::std::array<int, 1>{1},
    cuda::std::array<int, 1>{1});
  test_comparison(
    false,
    cuda::std::extents<T1, D>(5),
    cuda::std::extents<T2, 7>(),
    cuda::std::array<int, 1>{1},
    cuda::std::array<int, 1>{1});
  test_comparison(
    false,
    cuda::std::extents<T1, 5>(),
    cuda::std::extents<T2, 7>(),
    cuda::std::array<int, 1>{1},
    cuda::std::array<int, 1>{1});

  test_comparison(
    true,
    cuda::std::extents<T1, D, D, D, D, D>(5, 6, 7, 8, 9),
    cuda::std::extents<T2, D, D, D, D, D>(5, 6, 7, 8, 9),
    cuda::std::array<int, 5>{2, 20, 200, 2000, 20000},
    cuda::std::array<int, 5>{2, 20, 200, 2000, 20000});
  test_comparison(
    true,
    cuda::std::extents<T1, D, 6, D, 8, D>(5, 7, 9),
    cuda::std::extents<T2, 5, D, D, 8, 9>(6, 7),
    cuda::std::array<int, 5>{2, 20, 200, 2000, 20000},
    cuda::std::array<int, 5>{2, 20, 200, 2000, 20000});
  test_comparison(
    true,
    cuda::std::extents<T1, 5, 6, 7, 8, 9>(5, 6, 7, 8, 9),
    cuda::std::extents<T2, 5, 6, 7, 8, 9>(),
    cuda::std::array<int, 5>{2, 20, 200, 2000, 20000},
    cuda::std::array<int, 5>{2, 20, 200, 2000, 20000});
  test_comparison(
    false,
    cuda::std::extents<T1, 5, 6, 7, 8, 9>(5, 6, 7, 8, 9),
    cuda::std::extents<T2, 5, 6, 7, 8, 9>(),
    cuda::std::array<int, 5>{2, 20, 200, 20000, 2000},
    cuda::std::array<int, 5>{2, 20, 200, 2000, 20000});
  test_comparison(
    false,
    cuda::std::extents<T1, D, D, D, D, D>(5, 6, 7, 8, 9),
    cuda::std::extents<T2, D, D, D, D, D>(5, 6, 3, 8, 9),
    cuda::std::array<int, 5>{2, 20, 200, 2000, 20000},
    cuda::std::array<int, 5>{2, 20, 200, 2000, 20000});
  test_comparison(
    false,
    cuda::std::extents<T1, D, 6, D, 8, D>(5, 7, 9),
    cuda::std::extents<T2, 5, D, D, 3, 9>(6, 7),
    cuda::std::array<int, 5>{2, 20, 200, 2000, 20000},
    cuda::std::array<int, 5>{2, 20, 200, 2000, 20000});
  test_comparison(
    false,
    cuda::std::extents<T1, 5, 6, 7, 8, 9>(5, 6, 7, 8, 9),
    cuda::std::extents<T2, 5, 6, 7, 3, 9>(),
    cuda::std::array<int, 5>{2, 20, 200, 2000, 20000},
    cuda::std::array<int, 5>{2, 20, 200, 2000, 20000});
}

template <class OtherLayout, class E1, class E2, class... OtherArgs>
__host__ __device__ constexpr void test_comparison_with(
  bool expect_equal,
  E1 e1,
  cuda::std::array<typename E1::index_type, E1::rank()> strides,
  E2 e2,
  OtherArgs... other_args)
{
  typename cuda::std::layout_stride::template mapping<E1> map(e1, strides);
  typename OtherLayout::template mapping<E2> other_map(e2, other_args...);

  assert((map == other_map) == expect_equal);
}

template <class OtherLayout,
          cuda::std::enable_if_t<cuda::std::is_same<OtherLayout, always_convertible_layout>::value, int> = 0>
__host__ __device__ constexpr void test_comparison_with_always_convertible()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  // test layout with strides not equal to product of extents
  test_comparison_with<OtherLayout>(
    true,
    cuda::std::extents<int, D, D>(5, 7),
    cuda::std::array<int, 2>{2, 10},
    cuda::std::extents<unsigned, D, D>(5, 7),
    0,
    2);
  // make sure that offset != 0 results in false
  test_comparison_with<OtherLayout>(
    false,
    cuda::std::extents<int, D, D>(5, 7),
    cuda::std::array<int, 2>{2, 10},
    cuda::std::extents<unsigned, D, D>(5, 7),
    1,
    2);
}

template <class OtherLayout,
          cuda::std::enable_if_t<!cuda::std::is_same<OtherLayout, always_convertible_layout>::value, int> = 0>
__host__ __device__ constexpr void test_comparison_with_always_convertible()
{}

template <class OtherLayout>
__host__ __device__ constexpr void test_comparison_with()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  bool is_left_based                  = cuda::std::is_same_v<OtherLayout, cuda::std::layout_left>
                    || cuda::std::is_same_v<OtherLayout, always_convertible_layout>;
  test_comparison_with<OtherLayout>(
    true, cuda::std::extents<int>(), cuda::std::array<int, 0>{}, cuda::std::extents<unsigned>());
  test_comparison_with<OtherLayout>(
    true, cuda::std::extents<int, 5>(), cuda::std::array<int, 1>{1}, cuda::std::extents<unsigned, 5>());
  test_comparison_with<OtherLayout>(
    true, cuda::std::extents<int, D>(5), cuda::std::array<int, 1>{1}, cuda::std::extents<unsigned, 5>());
  test_comparison_with<OtherLayout>(
    false, cuda::std::extents<int, D>(5), cuda::std::array<int, 1>{2}, cuda::std::extents<unsigned, 5>());
  test_comparison_with<OtherLayout>(
    is_left_based,
    cuda::std::extents<int, D, D>(5, 7),
    cuda::std::array<int, 2>{1, 5},
    cuda::std::extents<unsigned, D, D>(5, 7));
  test_comparison_with<OtherLayout>(
    !is_left_based,
    cuda::std::extents<int, D, D>(5, 7),
    cuda::std::array<int, 2>{7, 1},
    cuda::std::extents<unsigned, D, D>(5, 7));
  test_comparison_with<OtherLayout>(
    false,
    cuda::std::extents<int, D, D>(5, 7),
    cuda::std::array<int, 2>{8, 1},
    cuda::std::extents<unsigned, D, D>(5, 7));

  test_comparison_with_always_convertible<OtherLayout>();
}

template <class T1, class T2>
__host__ __device__ constexpr void test_comparison_index_type()
{
  test_comparison_same_rank<T1, T2>();
  test_comparison_different_rank<T1, T2>();
  test_comparison_with<cuda::std::layout_right>();
  test_comparison_with<cuda::std::layout_left>();
  test_comparison_with<always_convertible_layout>();
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
  static_assert(test(), "");
  return 0;
}
