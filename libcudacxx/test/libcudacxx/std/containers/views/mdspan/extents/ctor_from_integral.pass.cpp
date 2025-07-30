//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// Test construction from integral:
//
// template<class ... OtherIndexTypes>
//     constexpr explicit extents(OtherIndexTypes ... exts) noexcept;
//
// Let N be sizeof...(OtherIndexTypes), and let
// exts_arr be array<index_type, N>{static_cast<index_type>(cuda::std::move(exts))...}.
//
// Constraints:
//   * (is_convertible_v<OtherIndexTypes, index_type> && ...) is true,
//   * (is_nothrow_constructible_v<index_type, OtherIndexType> && ...) is true, and
//   * N == rank_dynamic() || N == rank() is true.
//
// Preconditions:
//   * If N != rank_dynamic() is true, exts_arr[r] equals Er for each r for which
//     Er is a static extent, and
//   * either
//     - sizeof...(exts) == 0 is true, or
//     - each element of exts is nonnegative and is representable as a value of type index_type.
//

#include <cuda/std/cassert>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "../ConvertibleToIntegral.h"
#include "CtorTestCombinations.h"
#include "test_macros.h"

struct IntegralCtorTest
{
  template <class E, class AllExtents, class Extents, size_t... Indices>
  __host__ __device__ static constexpr void
  test_construction(AllExtents all_ext, Extents ext, cuda::std::index_sequence<Indices...>)
  {
    // construction from indices
    static_assert(noexcept(E(ext[Indices]...)));
    test_runtime_observers(E(ext[Indices]...), all_ext);
  }
};

int main(int, char**)
{
  test_index_type_combo<IntegralCtorTest>();
#if TEST_STD_VER >= 2020
  static_assert(test_index_type_combo<IntegralCtorTest>(), "");
#endif // TEST_STD_VER >= 2020

  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  using E                             = cuda::std::extents<int, 1, D, 3, D>;

  // check can't construct from too few arguments
  static_assert(!cuda::std::is_constructible<E, int>::value, "extents constructible from illegal arguments");
  // check can't construct from rank_dynamic < #args < rank
  static_assert(!cuda::std::is_constructible<E, int, int, int>::value, "extents constructible from illegal arguments");
  // check can't construct from too many arguments
  static_assert(!cuda::std::is_constructible<E, int, int, int, int, int>::value,
                "extents constructible from illegal arguments");

  // test construction fails from types not convertible to index_type but convertible to other integer types
  static_assert(cuda::std::is_convertible<IntType, int>::value,
                "Test helper IntType unexpectedly not convertible to int");
  static_assert(!cuda::std::is_constructible<cuda::std::extents<unsigned long, D>, IntType>::value,
                "extents constructible from illegal arguments");
  return 0;
}
