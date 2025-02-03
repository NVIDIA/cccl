//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// template<class OtherIndexType, size_t... OtherExtents>
//     constexpr explicit(see below) extents(const extents<OtherIndexType, OtherExtents...>&) noexcept;
//
// Constraints:
//   * sizeof...(OtherExtents) == rank() is true.
//   * ((OtherExtents == dynamic_extent || Extents == dynamic_extent ||
//       OtherExtents == Extents) && ...) is true.
//
// Preconditions:
//   * other.extent(r) equals Er for each r for which Er is a static extent, and
//   * either
//      - sizeof...(OtherExtents) is zero, or
//      - other.extent(r) is representable as a value of type index_type for
//        every rank index r of other.
//
// Remarks: The expression inside explicit is equivalent to:
//          (((Extents != dynamic_extent) && (OtherExtents == dynamic_extent)) || ... ) ||
//          (numeric_limits<index_type>::max() < numeric_limits<OtherIndexType>::max())

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class To, class From>
__host__ __device__ constexpr void test_implicit_conversion(To dest, From src)
{
  assert(dest == src);
}

template <bool implicit, class To, class From, cuda::std::enable_if_t<implicit, int> = 0>
__host__ __device__ constexpr void test_conversion(From src)
{
  To dest(src);
  assert(dest == src);
  dest = src;
  assert(dest == src);
  test_implicit_conversion<To, From>(src, src);
}

template <bool implicit, class To, class From, cuda::std::enable_if_t<!implicit, int> = 0>
__host__ __device__ constexpr void test_conversion(From src)
{
  To dest(src);
  assert(dest == src);
}

template <class T1, class T2>
__host__ __device__ constexpr void test_conversion()
{
  constexpr bool idx_convertible = static_cast<size_t>(cuda::std::numeric_limits<T1>::max())
                                >= static_cast<size_t>(cuda::std::numeric_limits<T2>::max());

  // clang-format off
  test_conversion<idx_convertible && true,  cuda::std::extents<T1>>(cuda::std::extents<T2>());
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, cuda::std::dynamic_extent>>(cuda::std::extents<T2, cuda::std::dynamic_extent>(5));
  test_conversion<idx_convertible && false, cuda::std::extents<T1, 5>>(cuda::std::extents<T2, cuda::std::dynamic_extent>(5));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, 5>>(cuda::std::extents<T2, 5>());
  test_conversion<idx_convertible && false, cuda::std::extents<T1, 5, cuda::std::dynamic_extent>>(cuda::std::extents<T2, cuda::std::dynamic_extent, cuda::std::dynamic_extent>(5, 5));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, cuda::std::dynamic_extent, cuda::std::dynamic_extent>>(cuda::std::extents<T2, cuda::std::dynamic_extent, cuda::std::dynamic_extent>(5, 5));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, cuda::std::dynamic_extent, cuda::std::dynamic_extent>>(cuda::std::extents<T2, cuda::std::dynamic_extent, 7>(5));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, 5, 7>>(cuda::std::extents<T2, 5, 7>());
  test_conversion<idx_convertible && false, cuda::std::extents<T1, 5, cuda::std::dynamic_extent, 8, cuda::std::dynamic_extent, cuda::std::dynamic_extent>>(cuda::std::extents<T2, cuda::std::dynamic_extent, cuda::std::dynamic_extent, 8, 9, 1>(5, 7));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>>(
                                            cuda::std::extents<T2, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>(5, 7, 8, 9, 1));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, cuda::std::dynamic_extent, cuda::std::dynamic_extent, 8, 9, cuda::std::dynamic_extent>>(cuda::std::extents<T2, cuda::std::dynamic_extent, 7, 8, 9, 1>(5));
  test_conversion<idx_convertible && true,  cuda::std::extents<T1, 5, 7, 8, 9, 1>>(cuda::std::extents<T2, 5, 7, 8, 9, 1>());
  // clang-format on
}

__host__ __device__ constexpr void test_no_implicit_conversion()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  // Sanity check that one static to dynamic conversion works
  static_assert(
    cuda::std::is_constructible<cuda::std::extents<int, cuda::std::dynamic_extent>, cuda::std::extents<int, 5>>::value,
    "");
  static_assert(
    cuda::std::is_convertible<cuda::std::extents<int, 5>, cuda::std::extents<int, cuda::std::dynamic_extent>>::value,
    "");

  // Check that dynamic to static conversion only works explicitly only
  static_assert(
    cuda::std::is_constructible<cuda::std::extents<int, 5>, cuda::std::extents<int, cuda::std::dynamic_extent>>::value,
    "");
  static_assert(
    !cuda::std::is_convertible<cuda::std::extents<int, cuda::std::dynamic_extent>, cuda::std::extents<int, 5>>::value,
    "");

  // Sanity check that one static to dynamic conversion works
  static_assert(cuda::std::is_constructible<cuda::std::extents<int, cuda::std::dynamic_extent, 7>,
                                            cuda::std::extents<int, 5, 7>>::value,
                "");
  static_assert(cuda::std::is_convertible<cuda::std::extents<int, 5, 7>,
                                          cuda::std::extents<int, cuda::std::dynamic_extent, 7>>::value,
                "");

  // Check that dynamic to static conversion only works explicitly only
  static_assert(cuda::std::is_constructible<cuda::std::extents<int, 5, 7>,
                                            cuda::std::extents<int, cuda::std::dynamic_extent, 7>>::value,
                "");
  static_assert(!cuda::std::is_convertible<cuda::std::extents<int, cuda::std::dynamic_extent, 7>,
                                           cuda::std::extents<int, 5, 7>>::value,
                "");

  // Sanity check that smaller index_type to larger index_type conversion works
  static_assert(cuda::std::is_constructible<cuda::std::extents<size_t, 5>, cuda::std::extents<int, 5>>::value, "");
  static_assert(cuda::std::is_convertible<cuda::std::extents<int, 5>, cuda::std::extents<size_t, 5>>::value, "");

  // Check that larger index_type to smaller index_type conversion works explicitly only
  static_assert(cuda::std::is_constructible<cuda::std::extents<int, 5>, cuda::std::extents<size_t, 5>>::value, "");
  static_assert(!cuda::std::is_convertible<cuda::std::extents<size_t, 5>, cuda::std::extents<int, 5>>::value, "");
}

__host__ __device__ constexpr void test_rank_mismatch()
{
  static_assert(
    !cuda::std::is_constructible<cuda::std::extents<int, cuda::std::dynamic_extent>, cuda::std::extents<int>>::value,
    "");
  static_assert(
    !cuda::std::is_constructible<cuda::std::extents<int>,
                                 cuda::std::extents<int, cuda::std::dynamic_extent, cuda::std::dynamic_extent>>::value,
    "");
  static_assert(
    !cuda::std::is_constructible<cuda::std::extents<int, cuda::std::dynamic_extent>,
                                 cuda::std::extents<int, cuda::std::dynamic_extent, cuda::std::dynamic_extent>>::value,
    "");
  static_assert(
    !cuda::std::is_constructible<
      cuda::std::extents<int, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>,
      cuda::std::extents<int, cuda::std::dynamic_extent, cuda::std::dynamic_extent>>::value,
    "");
}

__host__ __device__ constexpr void test_static_extent_mismatch()
{
  static_assert(!cuda::std::is_constructible<cuda::std::extents<int, cuda::std::dynamic_extent, 5>,
                                             cuda::std::extents<int, cuda::std::dynamic_extent, 4>>::value,
                "");
  static_assert(!cuda::std::is_constructible<cuda::std::extents<int, 5>, cuda::std::extents<int, 4>>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::extents<int, 5, cuda::std::dynamic_extent>,
                                             cuda::std::extents<int, 4, cuda::std::dynamic_extent>>::value,
                "");
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
