//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// template<class StridedLayoutMapping>
//   constexpr explicit(see below)
//     mapping(const StridedLayoutMapping& other) noexcept;
//
// Constraints:
//   - layout-mapping-alike<StridedLayoutMapping> is satisfied.
//   - is_constructible_v<extents_type, typename StridedLayoutMapping::extents_type> is true.
//   - StridedLayoutMapping::is_always_unique() is true.
//   - StridedLayoutMapping::is_always_strided() is true.
//
// Preconditions:
//   - StridedLayoutMapping meets the layout mapping requirements ([mdspan.layout.policy.reqmts]),
//   - other.stride(r) > 0 is true for every rank index r of extents(),
//   - other.required_span_size() is representable as a value of type index_type ([basic.fundamental]), and
//   - OFFSET(other) == 0 is true.
//
// Effects: Direct-non-list-initializes extents_ with other.extents(), and for all d in the range [0, rank_),
//          direct-non-list-initializes strides_[d] with other.stride(d).
//
// Remarks: The expression inside explicit is equivalent to:
//   - !(is_convertible_v<typename StridedLayoutMapping::extents_type, extents_type> &&
//       (is-mapping-of<layout_left, LayoutStrideMapping> ||
//        is-mapping-of<layout_right, LayoutStrideMapping> ||
//        is-mapping-of<layout_stride, LayoutStrideMapping>))

#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "../CustomTestLayouts.h"
#include "test_macros.h"

template <class FromL,
          class FromExt,
          cuda::std::enable_if_t<cuda::std::is_same<FromL, cuda::std::layout_stride>::value, int> = 0>
__host__ __device__ constexpr auto get_strides(FromExt src_exts)
{
  using From = typename FromL::template mapping<FromExt>;

  // just construct some strides which aren't layout_left/layout_right
  cuda::std::array<size_t, FromExt::rank()> strides{};
  size_t stride = 2;
  for (size_t r = 0; r < FromExt::rank(); r++)
  {
    strides[r] = stride;
    stride *= src_exts.extent(r);
  }
  return From(src_exts, strides);
}
template <class FromL,
          class FromExt,
          cuda::std::enable_if_t<!cuda::std::is_same<FromL, cuda::std::layout_stride>::value, int> = 0>
__host__ __device__ constexpr auto get_strides(FromExt src_exts)
{
  using From = typename FromL::template mapping<FromExt>;
  return From(src_exts);
}

template <bool implicit, class FromL, class ToExt, class FromExt, cuda::std::enable_if_t<implicit, int> = 0>
__host__ __device__ constexpr void test_conversion(FromExt src_exts)
{
  using To   = cuda::std::layout_stride::mapping<ToExt>;
  using From = typename FromL::template mapping<FromExt>;

  From src(get_strides<FromL>(src_exts));
  static_assert(noexcept(To(src)));
  To dest(src);
  assert(dest == src);

  To dest_implicit = src;
  assert(dest_implicit == src);
}

template <bool implicit, class FromL, class ToExt, class FromExt, cuda::std::enable_if_t<!implicit, int> = 0>
__host__ __device__ constexpr void test_conversion(FromExt src_exts)
{
  using To   = cuda::std::layout_stride::mapping<ToExt>;
  using From = typename FromL::template mapping<FromExt>;

  From src(get_strides<FromL>(src_exts));
  static_assert(noexcept(To(src)));
  To dest(src);
  assert(dest == src);
  assert((!cuda::std::is_convertible_v<From, To>) );
}

template <class FromL, class T1, class T2>
__host__ __device__ constexpr void test_conversion()
{
  constexpr size_t D             = cuda::std::dynamic_extent;
  constexpr bool idx_convertible = static_cast<size_t>(cuda::std::numeric_limits<T1>::max())
                                >= static_cast<size_t>(cuda::std::numeric_limits<T2>::max());
  constexpr bool l_convertible =
    cuda::std::is_same_v<FromL, cuda::std::layout_right> || cuda::std::is_same_v<FromL, cuda::std::layout_left>
    || cuda::std::is_same_v<FromL, cuda::std::layout_stride>;
  constexpr bool idx_l_convertible = idx_convertible && l_convertible;

  // clang-format off
  // adding extents convertibility expectation
  test_conversion<idx_l_convertible && true,  FromL, cuda::std::extents<T1>>(cuda::std::extents<T2>());
  test_conversion<idx_l_convertible && true,  FromL, cuda::std::extents<T1, D>>(cuda::std::extents<T2, D>(0));
  test_conversion<idx_l_convertible && true,  FromL, cuda::std::extents<T1, D>>(cuda::std::extents<T2, D>(5));
  test_conversion<idx_l_convertible && false, FromL, cuda::std::extents<T1, 5>>(cuda::std::extents<T2, D>(5));
  test_conversion<idx_l_convertible && true,  FromL, cuda::std::extents<T1, 5>>(cuda::std::extents<T2, 5>());
  test_conversion<idx_l_convertible && false, FromL, cuda::std::extents<T1, 5, D>>(cuda::std::extents<T2, D, D>(5, 5));
  test_conversion<idx_l_convertible && true,  FromL, cuda::std::extents<T1, D, D>>(cuda::std::extents<T2, D, D>(5, 5));
  test_conversion<idx_l_convertible && true,  FromL, cuda::std::extents<T1, D, D>>(cuda::std::extents<T2, D, 7>(5));
  test_conversion<idx_l_convertible && true,  FromL, cuda::std::extents<T1, 5, 7>>(cuda::std::extents<T2, 5, 7>());
  test_conversion<idx_l_convertible && false, FromL, cuda::std::extents<T1, 5, D, 8, D, D>>(cuda::std::extents<T2, D, D, 8, 9, 1>(5, 7));
  test_conversion<idx_l_convertible && true,  FromL, cuda::std::extents<T1, D, D, D, D, D>>(
                                                     cuda::std::extents<T2, D, D, D, D, D>(5, 7, 8, 9, 1));
  test_conversion<idx_l_convertible && true,  FromL, cuda::std::extents<T1, D, D, 8, 9, D>>(cuda::std::extents<T2, D, 7, 8, 9, 1>(5));
  test_conversion<idx_l_convertible && true,  FromL, cuda::std::extents<T1, 5, 7, 8, 9, 1>>(cuda::std::extents<T2, 5, 7, 8, 9, 1>());
  // clang-format on
}

template <class IdxT, size_t... Extents>
using ToM = typename cuda::std::layout_stride::template mapping<cuda::std::extents<IdxT, Extents...>>;

template <class FromL, class IdxT, size_t... Extents>
using FromM = typename FromL::template mapping<cuda::std::extents<IdxT, Extents...>>;

template <class FromL, cuda::std::enable_if_t<!cuda::std::is_same_v<FromL, always_convertible_layout>, int> = 0>
__host__ __device__ constexpr void test_no_implicit_conversion()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // Sanity check that one static to dynamic conversion works
  static_assert(cuda::std::is_constructible<ToM<int, D>, FromM<FromL, int, 5>>::value, "");
  static_assert(cuda::std::is_convertible<FromM<FromL, int, 5>, ToM<int, D>>::value, "");

  // Check that dynamic to static conversion only works explicitly
  static_assert(cuda::std::is_constructible<ToM<int, 5>, FromM<FromL, int, D>>::value, "");
  static_assert(!cuda::std::is_convertible<FromM<FromL, int, D>, ToM<int, 5>>::value, "");

  // Sanity check that one static to dynamic conversion works
  static_assert(cuda::std::is_constructible<ToM<int, D, 7>, FromM<FromL, int, 5, 7>>::value, "");
  static_assert(cuda::std::is_convertible<FromM<FromL, int, 5, 7>, ToM<int, D, 7>>::value, "");

  // Check that dynamic to static conversion only works explicitly
  static_assert(cuda::std::is_constructible<ToM<int, 5, 7>, FromM<FromL, int, D, 7>>::value, "");
  static_assert(!cuda::std::is_convertible<FromM<FromL, int, D, 7>, ToM<int, 5, 7>>::value, "");

  // Sanity check that smaller index_type to larger index_type conversion works
  static_assert(cuda::std::is_constructible<ToM<size_t, 5>, FromM<FromL, int, 5>>::value, "");
  static_assert(cuda::std::is_convertible<FromM<FromL, int, 5>, ToM<size_t, 5>>::value, "");

  // Check that larger index_type to smaller index_type conversion works explicitly only
  static_assert(cuda::std::is_constructible<ToM<int, 5>, FromM<FromL, size_t, 5>>::value, "");
  static_assert(!cuda::std::is_convertible<FromM<FromL, size_t, 5>, ToM<int, 5>>::value, "");
}

// the implicit convertibility test doesn't apply to non cuda::std::layouts
template <class FromL, cuda::std::enable_if_t<cuda::std::is_same_v<FromL, always_convertible_layout>, int> = 0>
__host__ __device__ constexpr void test_no_implicit_conversion()
{}

template <class FromL>
__host__ __device__ constexpr void test_rank_mismatch()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  static_assert(!cuda::std::is_constructible<ToM<int, D>, FromM<FromL, int>>::value, "");
  static_assert(!cuda::std::is_constructible<ToM<int>, FromM<FromL, int, D, D>>::value, "");
  static_assert(!cuda::std::is_constructible<ToM<int, D>, FromM<FromL, int, D, D>>::value, "");
  static_assert(!cuda::std::is_constructible<ToM<int, D, D, D>, FromM<FromL, int, D, D>>::value, "");
}

template <class FromL>
__host__ __device__ constexpr void test_static_extent_mismatch()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  static_assert(!cuda::std::is_constructible<ToM<int, D, 5>, FromM<FromL, int, D, 4>>::value, "");
  static_assert(!cuda::std::is_constructible<ToM<int, 5>, FromM<FromL, int, 4>>::value, "");
  static_assert(!cuda::std::is_constructible<ToM<int, 5, D>, FromM<FromL, int, 4, D>>::value, "");
}

template <class FromL>
__host__ __device__ constexpr void test_layout()
{
  test_conversion<FromL, int, int>();
  test_conversion<FromL, int, size_t>();
  test_conversion<FromL, size_t, int>();
  test_conversion<FromL, size_t, long>();
  test_no_implicit_conversion<FromL>();
  test_rank_mismatch<FromL>();
  test_static_extent_mismatch<FromL>();
}

__host__ __device__ constexpr bool test()
{
  test_layout<cuda::std::layout_right>();
  test_layout<cuda::std::layout_left>();
  test_layout<cuda::std::layout_stride>();
  test_layout<always_convertible_layout>();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
