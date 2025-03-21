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
//   constexpr explicit(extents_type::rank() > 0)
//     mapping(const layout_stride::mapping<OtherExtents>& other);
//
// Constraints: is_constructible_v<extents_type, OtherExtents> is true.
//
// Preconditions:
//   - If extents_type::rank() > 0 is true, then for all r in the range [0, extents_type::rank()),
//     other.stride(r) equals other.extents().fwd-prod-of-extents(r), and
//   - other.required_span_size() is representable as a value of type index_type ([basic.fundamental]).
//
// Effects: Direct-non-list-initializes extents_ with other.extents().

#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <bool implicit, class To, class From, cuda::std::enable_if_t<implicit, int> = 0>
__host__ __device__ constexpr void test_implicit_conversion(From src)
{
  To dest_implicit = src;
  assert(dest_implicit == src);
}

template <bool implicit, class To, class From, cuda::std::enable_if_t<!implicit, int> = 0>
__host__ __device__ constexpr void test_implicit_conversion(From src)
{
  assert((!cuda::std::is_convertible_v<From, To>) );
}

template <class FromExt, cuda::std::enable_if_t<(FromExt::rank() > 0), int> = 0>
__host__ __device__ constexpr auto get_strides(FromExt src_exts)
{
  cuda::std::array<typename FromExt::index_type, FromExt::rank()> strides{};
  strides[0] = 1;
  for (size_t r = 1; r < FromExt::rank(); r++)
  {
    strides[r] = src_exts.extent(r - 1) * strides[r - 1];
  }
  return strides;
}

template <class FromExt, cuda::std::enable_if_t<(FromExt::rank() == 0), int> = 0>
__host__ __device__ constexpr auto get_strides(FromExt)
{
  return cuda::std::array<typename FromExt::index_type, FromExt::rank()>{};
}

template <bool implicit, class ToExt, class FromExt>
__host__ __device__ constexpr void test_conversion(FromExt src_exts)
{
  using To           = cuda::std::layout_left::mapping<ToExt>;
  using From         = cuda::std::layout_stride::mapping<FromExt>;
  const auto strides = get_strides(src_exts);
  From src(src_exts, strides);

  static_assert(noexcept(To(src)));
  To dest(src);
  assert(dest == src);
  test_implicit_conversion<implicit, To, From>(src);
}

template <class T1, class T2>
__host__ __device__ constexpr void test_conversion()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // clang-format off
  test_conversion<true,  cuda::std::extents<T1>>(cuda::std::extents<T2>());
  test_conversion<false, cuda::std::extents<T1, D>>(cuda::std::extents<T2, D>(5));
  test_conversion<false, cuda::std::extents<T1, 5>>(cuda::std::extents<T2, D>(5));
  test_conversion<false, cuda::std::extents<T1, 5>>(cuda::std::extents<T2, 5>());
  test_conversion<false, cuda::std::extents<T1, 5, D>>(cuda::std::extents<T2, D, D>(5, 5));
  test_conversion<false, cuda::std::extents<T1, D, D>>(cuda::std::extents<T2, D, D>(5, 5));
  test_conversion<false, cuda::std::extents<T1, D, D>>(cuda::std::extents<T2, D, 7>(5));
  test_conversion<false, cuda::std::extents<T1, 5, 7>>(cuda::std::extents<T2, 5, 7>());
  test_conversion<false, cuda::std::extents<T1, 5, D, 8, D, D>>(cuda::std::extents<T2, D, D, 8, 9, 1>(5, 7));
  test_conversion<false, cuda::std::extents<T1, D, D, D, D, D>>(
                         cuda::std::extents<T2, D, D, D, D, D>(5, 7, 8, 9, 1));
  test_conversion<false, cuda::std::extents<T1, D, D, 8, 9, D>>(cuda::std::extents<T2, D, 7, 8, 9, 1>(5));
  test_conversion<false, cuda::std::extents<T1, 5, 7, 8, 9, 1>>(cuda::std::extents<T2, 5, 7, 8, 9, 1>());
  // clang-format on
}

template <class IdxT, size_t... Extents>
using lr_mapping_t = typename cuda::std::layout_right::template mapping<cuda::std::extents<IdxT, Extents...>>;
template <class IdxT, size_t... Extents>
using ls_mapping_t = typename cuda::std::layout_stride::template mapping<cuda::std::extents<IdxT, Extents...>>;

__host__ __device__ constexpr void test_rank_mismatch()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  static_assert(!cuda::std::is_constructible<lr_mapping_t<int, D>, ls_mapping_t<int>>::value, "");
  static_assert(!cuda::std::is_constructible<lr_mapping_t<int>, ls_mapping_t<int, D, D>>::value, "");
  static_assert(!cuda::std::is_constructible<lr_mapping_t<int, D>, ls_mapping_t<int, D, D>>::value, "");
  static_assert(!cuda::std::is_constructible<lr_mapping_t<int, D, D, D>, ls_mapping_t<int, D, D>>::value, "");
}

__host__ __device__ constexpr void test_static_extent_mismatch()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  static_assert(!cuda::std::is_constructible<lr_mapping_t<int, D, 5>, ls_mapping_t<int, D, 4>>::value, "");
  static_assert(!cuda::std::is_constructible<lr_mapping_t<int, 5>, ls_mapping_t<int, 4>>::value, "");
  static_assert(!cuda::std::is_constructible<lr_mapping_t<int, 5, D>, ls_mapping_t<int, 4, D>>::value, "");
}

__host__ __device__ constexpr bool test()
{
  test_conversion<int, int>();
  test_conversion<int, size_t>();
  test_conversion<size_t, int>();
  test_conversion<size_t, long>();
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
