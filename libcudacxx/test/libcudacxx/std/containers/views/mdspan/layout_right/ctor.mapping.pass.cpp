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
//     mapping(const mapping<OtherExtents>&) noexcept;

// Constraints: is_constructible<extents_type, OtherExtents> is true.
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
  assert(dest == src);
}

template <bool implicit, class ToExt, class FromExt, cuda::std::enable_if_t<implicit, int> = 0>
__host__ __device__ constexpr void test_conversion(FromExt src_exts)
{
  using To   = cuda::std::layout_right::mapping<ToExt>;
  using From = cuda::std::layout_right::mapping<FromExt>;
  From src(src_exts);

  static_assert(noexcept(To(src)));
  To dest(src);

  assert(dest == src);
  dest = src;
  assert(dest == src);
  test_implicit_conversion<To, From>(src, src);
}

template <bool implicit, class ToExt, class FromExt, cuda::std::enable_if_t<!implicit, int> = 0>
__host__ __device__ constexpr void test_conversion(FromExt src_exts)
{
  using To   = cuda::std::layout_right::mapping<ToExt>;
  using From = cuda::std::layout_right::mapping<FromExt>;
  From src(src_exts);

  static_assert(noexcept(To(src)));
  To dest(src);

  assert(dest == src);
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
  test_conversion<idx_convertible && false, cuda::std::extents<T1, 5, D>>(cuda::std::extents<T2, D, D>(5, 5));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, D, D>>(cuda::std::extents<T2, D, D>(5, 5));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, D, D>>(cuda::std::extents<T2, D, 7>(5));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, 5, 7>>(cuda::std::extents<T2, 5, 7>());
  test_conversion<idx_convertible && false, cuda::std::extents<T1, 5, D, 8, D, D>>(cuda::std::extents<T2, D, D, 8, 9, 1>(5, 7));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, D, D, D, D, D>>(
                                            cuda::std::extents<T2, D, D, D, D, D>(5, 7, 8, 9, 1));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, D, D, 8, 9, D>>(cuda::std::extents<T2, D, 7, 8, 9, 1>(5));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, 5, 7, 8, 9, 1>>(cuda::std::extents<T2, 5, 7, 8, 9, 1>());
  // clang-format on
}

template <class IdxT, size_t... Extents>
using mapping_t = typename cuda::std::layout_right::template mapping<cuda::std::extents<IdxT, Extents...>>;

__host__ __device__ constexpr void test_no_implicit_conversion()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // Sanity check that one static to dynamic conversion works
  static_assert(cuda::std::is_constructible<mapping_t<int, D>, mapping_t<int, 5>>::value, "");
  static_assert(cuda::std::is_convertible<mapping_t<int, 5>, mapping_t<int, D>>::value, "");

  // Check that dynamic to static conversion only works explicitly
  static_assert(cuda::std::is_constructible<mapping_t<int, 5>, mapping_t<int, D>>::value, "");
  static_assert(!cuda::std::is_convertible<mapping_t<int, D>, mapping_t<int, 5>>::value, "");

  // Sanity check that one static to dynamic conversion works
  static_assert(cuda::std::is_constructible<mapping_t<int, D, 7>, mapping_t<int, 5, 7>>::value, "");
  static_assert(cuda::std::is_convertible<mapping_t<int, 5, 7>, mapping_t<int, D, 7>>::value, "");

  // Check that dynamic to static conversion only works explicitly
  static_assert(cuda::std::is_constructible<mapping_t<int, 5, 7>, mapping_t<int, D, 7>>::value, "");
  static_assert(!cuda::std::is_convertible<mapping_t<int, D, 7>, mapping_t<int, 5, 7>>::value, "");

  // Sanity check that smaller index_type to larger index_type conversion works
  static_assert(cuda::std::is_constructible<mapping_t<size_t, 5>, mapping_t<int, 5>>::value, "");
  static_assert(cuda::std::is_convertible<mapping_t<int, 5>, mapping_t<size_t, 5>>::value, "");

  // Check that larger index_type to smaller index_type conversion works explicitly only
  static_assert(cuda::std::is_constructible<mapping_t<int, 5>, mapping_t<size_t, 5>>::value, "");
  static_assert(!cuda::std::is_convertible<mapping_t<size_t, 5>, mapping_t<int, 5>>::value, "");
}

__host__ __device__ constexpr void test_rank_mismatch()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  static_assert(!cuda::std::is_constructible<mapping_t<int, D>, mapping_t<int>>::value, "");
  static_assert(!cuda::std::is_constructible<mapping_t<int>, mapping_t<int, D, D>>::value, "");
  static_assert(!cuda::std::is_constructible<mapping_t<int, D>, mapping_t<int, D, D>>::value, "");
  static_assert(!cuda::std::is_constructible<mapping_t<int, D, D, D>, mapping_t<int, D, D>>::value, "");
}

__host__ __device__ constexpr void test_static_extent_mismatch()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  static_assert(!cuda::std::is_constructible<mapping_t<int, D, 5>, mapping_t<int, D, 4>>::value, "");
  static_assert(!cuda::std::is_constructible<mapping_t<int, 5>, mapping_t<int, 4>>::value, "");
  static_assert(!cuda::std::is_constructible<mapping_t<int, 5, D>, mapping_t<int, 4, D>>::value, "");
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
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
