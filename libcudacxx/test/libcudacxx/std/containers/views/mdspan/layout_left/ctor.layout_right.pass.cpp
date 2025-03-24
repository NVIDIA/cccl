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
//   constexpr explicit(!is_convertible<OtherExtents, extents_type>)
//     mapping(const layout_right::mapping<OtherExtents>&) noexcept;

// Constraints:
//   - extents_type::rank() <= 1 is true, and
//   - is_constructible<extents_type, OtherExtents> is true.
//
// Preconditions: other.required_span_size() is representable as a value of type index_type

#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class To, class From>
__host__ __device__ constexpr void test_implicit_conversion(To dest, From src)
{
  assert(dest.extents() == src.extents());
}

template <bool implicit, class ToExt, class FromExt, cuda::std::enable_if_t<implicit, int> = 0>
__host__ __device__ constexpr void test_conversion(FromExt src_exts)
{
  using To   = cuda::std::layout_left::mapping<ToExt>;
  using From = cuda::std::layout_right::mapping<FromExt>;
  From src(src_exts);

  static_assert(noexcept(To(src)));
  To dest(src);

  assert(dest.extents() == src.extents());
  dest = src;
  assert(dest.extents() == src.extents());
  test_implicit_conversion<To, From>(src, src);
}

template <bool implicit, class ToExt, class FromExt, cuda::std::enable_if_t<!implicit, int> = 0>
__host__ __device__ constexpr void test_conversion(FromExt src_exts)
{
  using To   = cuda::std::layout_left::mapping<ToExt>;
  using From = cuda::std::layout_right::mapping<FromExt>;
  From src(src_exts);

  static_assert(noexcept(To(src)));
  To dest(src);

  assert(dest.extents() == src.extents());
}

template <class T1, class T2>
__host__ __device__ constexpr void test_conversion()
{
  constexpr size_t D             = cuda::std::dynamic_extent;
  constexpr bool idx_convertible = static_cast<size_t>(cuda::std::numeric_limits<T1>::max())
                                >= static_cast<size_t>(cuda::std::numeric_limits<T2>::max());

  // clang-format off
  test_conversion<idx_convertible && true,  cuda::std::extents<T1>>(cuda::std::extents<T2>());
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, D>>(cuda::std::extents<T2, D>(5));
  test_conversion<idx_convertible && false, cuda::std::extents<T1, 5>>(cuda::std::extents<T2, D>(5));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, 5>>(cuda::std::extents<T2, 5>());
  // clang-format on
}

template <class IdxT, size_t... Extents>
using lr_mapping_t = typename cuda::std::layout_right::template mapping<cuda::std::extents<IdxT, Extents...>>;
template <class IdxT, size_t... Extents>
using ll_mapping_t = typename cuda::std::layout_left::template mapping<cuda::std::extents<IdxT, Extents...>>;

__host__ __device__ constexpr void test_no_implicit_conversion()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // Sanity check that one static to dynamic conversion works
  static_assert(cuda::std::is_constructible<ll_mapping_t<int, D>, lr_mapping_t<int, 5>>::value, "");
  static_assert(cuda::std::is_convertible<lr_mapping_t<int, 5>, ll_mapping_t<int, D>>::value, "");

  // Check that dynamic to static conversion only works explicitly
  static_assert(cuda::std::is_constructible<ll_mapping_t<int, 5>, lr_mapping_t<int, D>>::value, "");
  static_assert(!cuda::std::is_convertible<lr_mapping_t<int, D>, ll_mapping_t<int, 5>>::value, "");

  // Sanity check that smaller index_type to larger index_type conversion works
  static_assert(cuda::std::is_constructible<ll_mapping_t<size_t, 5>, lr_mapping_t<int, 5>>::value, "");
  static_assert(cuda::std::is_convertible<lr_mapping_t<int, 5>, ll_mapping_t<size_t, 5>>::value, "");

  // Check that larger index_type to smaller index_type conversion works explicitly only
  static_assert(cuda::std::is_constructible<ll_mapping_t<int, 5>, lr_mapping_t<size_t, 5>>::value, "");
  static_assert(!cuda::std::is_convertible<lr_mapping_t<size_t, 5>, ll_mapping_t<int, 5>>::value, "");
}

__host__ __device__ constexpr void test_rank_mismatch()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  static_assert(!cuda::std::is_constructible<lr_mapping_t<int, D>, lr_mapping_t<int>>::value, "");
  static_assert(!cuda::std::is_constructible<lr_mapping_t<int>, lr_mapping_t<int, D, D>>::value, "");
  static_assert(!cuda::std::is_constructible<lr_mapping_t<int, D>, lr_mapping_t<int, D, D>>::value, "");
  static_assert(!cuda::std::is_constructible<lr_mapping_t<int, D, D, D>, lr_mapping_t<int, D, D>>::value, "");
}

__host__ __device__ constexpr void test_static_extent_mismatch()
{
  static_assert(!cuda::std::is_constructible<ll_mapping_t<int, 5>, lr_mapping_t<int, 4>>::value, "");
}

__host__ __device__ constexpr void test_rank_greater_one()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  static_assert(!cuda::std::is_constructible<ll_mapping_t<int, D, D>, lr_mapping_t<int, D, D>>::value, "");
  static_assert(!cuda::std::is_constructible<ll_mapping_t<int, 1, 1>, lr_mapping_t<int, 1, 1>>::value, "");
  static_assert(!cuda::std::is_constructible<ll_mapping_t<int, D, D, D>, lr_mapping_t<int, D, D, D>>::value, "");
}

__host__ __device__ constexpr bool test()
{
  test_conversion<int, int>();
  test_conversion<int, size_t>();
  test_conversion<size_t, int>();
  test_conversion<size_t, long>();
  test_no_implicit_conversion();
  test_rank_mismatch();
  test_static_extent_mismatch();
  test_rank_greater_one();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
