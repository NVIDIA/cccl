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

// Converting constructor from strided layout mappings:
//
// template<class StridedLayoutMapping>
//   constexpr explicit(see below) mapping(const StridedLayoutMapping& other) noexcept;
//
// Constraints:
//   - layout-mapping-alike<StridedLayoutMapping> is satisfied.
//   - is_constructible_v<extents_type, typename StridedLayoutMapping::extents_type> is true.
//   - StridedLayoutMapping::is_always_unique() is true.
//   - StridedLayoutMapping::is_always_strided() is true.
//
// Remarks: The expression inside explicit is equivalent to:
//   - !(is_convertible_v<typename StridedLayoutMapping::extents_type, extents_type> &&
//       (is-mapping-of<layout_left, LayoutStrideMapping> ||
//        is-mapping-of<layout_right, LayoutStrideMapping> ||
//        is-mapping-of<layout_stride, LayoutStrideMapping>))

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "../CustomTestLayouts.h"
#include "test_macros.h"

using cuda::std::intptr_t;

template <class FromL, class FromExt>
__host__ __device__ constexpr auto get_strides(FromExt src_exts)
{
  using From = typename FromL::template mapping<FromExt>;

  if constexpr (cuda::std::is_same_v<FromL, cuda::std::layout_stride>)
  {
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
  else
  {
    return From(src_exts);
  }
}

template <bool implicit, class FromL, class ToExt, class FromExt>
__host__ __device__ constexpr void test_conversion(FromExt src_exts)
{
  using To   = cuda::layout_stride_relaxed::mapping<ToExt>;
  using From = typename FromL::template mapping<FromExt>;

  From src(get_strides<FromL>(src_exts));
  To dest(src);

  static_assert(noexcept(To(src)));
  assert(dest.extents() == src.extents());
  assert(dest.offset() == 0);

  if constexpr (ToExt::rank() > 0)
  {
    for (typename ToExt::rank_type r = 0; r < ToExt::rank(); r++)
    {
      assert(cuda::std::cmp_equal(dest.stride(r), src.stride(r)));
    }
  }
  if constexpr (implicit)
  {
    To dest_implicit = src;
    assert(dest_implicit.extents() == src.extents());
  }
  else
  {
    assert((!cuda::std::is_convertible_v<From, To>) );
  }
}

template <class FromL, class T1, class T2>
__host__ __device__ constexpr void test_conversion()
{
  using cuda::std::is_same_v;
  constexpr size_t D             = cuda::std::dynamic_extent;
  constexpr bool idx_convertible = static_cast<size_t>(cuda::std::numeric_limits<T1>::max())
                                >= static_cast<size_t>(cuda::std::numeric_limits<T2>::max());
  constexpr bool l_convertible = is_same_v<FromL, cuda::std::layout_right> || is_same_v<FromL, cuda::std::layout_left>
                              || is_same_v<FromL, cuda::std::layout_stride>;
  constexpr bool idx_l_convertible = idx_convertible && l_convertible;

  // clang-format off
  test_conversion<idx_l_convertible, FromL, cuda::std::extents<T1>>(
                                            cuda::std::extents<T2>());

  test_conversion<idx_l_convertible, FromL, cuda::std::extents<T1, D>>(
                                            cuda::std::extents<T2, D>(0));

  test_conversion<idx_l_convertible, FromL, cuda::std::extents<T1, D>>(
                                            cuda::std::extents<T2, D>(5));

  test_conversion<            false, FromL, cuda::std::extents<T1, 5>>(
                                            cuda::std::extents<T2, D>(5));

  test_conversion<idx_l_convertible, FromL, cuda::std::extents<T1, 5>>(
                                            cuda::std::extents<T2, 5>());

  test_conversion<            false, FromL, cuda::std::extents<T1, 5, D>>(
                                            cuda::std::extents<T2, D, D>(5, 5));

  test_conversion<idx_l_convertible, FromL, cuda::std::extents<T1, D, D>>(
                                            cuda::std::extents<T2, D, D>(5, 5));

  test_conversion<idx_l_convertible, FromL, cuda::std::extents<T1, D, D>>(
                                            cuda::std::extents<T2, D, 7>(5));

  test_conversion<idx_l_convertible, FromL, cuda::std::extents<T1, 5, 7>>(
                                            cuda::std::extents<T2, 5, 7>());

  test_conversion<            false, FromL, cuda::std::extents<T1, 5, D, 8, D, D>>(
                                            cuda::std::extents<T2, D, D, 8, 9, 1>(5, 7));

  test_conversion<idx_l_convertible, FromL, cuda::std::extents<T1, D, D, D, D, D>>(
                                            cuda::std::extents<T2, D, D, D, D, D>(5, 7, 8, 9, 1));

  test_conversion<idx_l_convertible, FromL, cuda::std::extents<T1, D, D, 8, 9, D>>(
                                            cuda::std::extents<T2, D, 7, 8, 9, 1>(5));

  test_conversion<            false, FromL, cuda::std::extents<T1, 5, 7, 8, 9, 1>>(
                                            cuda::std::extents<T2, 5, 7, 8, 9, 1>());
  // clang-format on
}

template <class FromL>
__host__ __device__ constexpr void test_layout()
{
  test_conversion<FromL, int, int>();
  test_conversion<FromL, int, size_t>();
  test_conversion<FromL, size_t, int>();
  test_conversion<FromL, size_t, long>();
}

// Test conversion from another layout_stride_relaxed::mapping
__host__ __device__ constexpr void test_self_conversion()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // Test converting ctor from layout_stride_relaxed to layout_stride_relaxed
  {
    using From         = cuda::layout_stride_relaxed::mapping<cuda::std::extents<int, 4, 5>>;
    using To           = cuda::layout_stride_relaxed::mapping<cuda::std::extents<int, D, D>>;
    using strides_type = typename From::strides_type;

    From src(cuda::std::extents<int, 4, 5>(), strides_type(5, 1), 10);
    To dest(src);

    assert(dest.extents().extent(0) == 4);
    assert(dest.extents().extent(1) == 5);
    assert(dest.offset() == 10);
    assert(dest.stride(0) == 5);
    assert(dest.stride(1) == 1);
  }

  // Test with negative strides
  {
    using From         = cuda::layout_stride_relaxed::mapping<cuda::std::extents<int, 4>>;
    using To           = cuda::layout_stride_relaxed::mapping<cuda::std::extents<int, D>>;
    using strides_type = typename From::strides_type;

    From src(cuda::std::extents<int, 4>(), strides_type(-1), 3);
    To dest(src);

    assert(dest.extents().extent(0) == 4);
    assert(dest.offset() == 3);
    assert(dest.stride(0) == -1);
  }
}

__host__ __device__ constexpr bool test()
{
  test_layout<cuda::std::layout_right>();
  test_layout<cuda::std::layout_left>();
  test_layout<cuda::std::layout_stride>();
  test_self_conversion();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
