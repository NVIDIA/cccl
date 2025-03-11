//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// Test construction from span:
//
// template<class OtherIndexType, size_t N>
//     constexpr explicit(N != rank_dynamic()) extents(span<OtherIndexType, N> exts) noexcept;
//
// Constraints:
//   * is_convertible_v<const OtherIndexType&, index_type> is true,
//   * is_nothrow_constructible_v<index_type, const OtherIndexType&> is true, and
//   * N == rank_dynamic() || N == rank() is true.
//
// Preconditions:
//   * If N != rank_dynamic() is true, exts[r] equals Er for each r for which
//     Er is a static extent, and
//   * either
//     - N is zero, or
//     - exts[r] is nonnegative and is representable as a value of type index_type
//       for every rank index r.
//

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/mdspan>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "../ConvertibleToIntegral.h"
#include "CtorTestCombinations.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(2912) // if-constexpr is a C++17 feature

struct SpanCtorTest
{
  template <class E,
            class T,
            size_t N,
            class Extents,
            size_t... Indices,
            cuda::std::enable_if_t<(N == E::rank_dynamic()), int> = 0>
  __host__ __device__ static constexpr void
  test_construction(cuda::std::array<T, N> all_ext, Extents ext, cuda::std::index_sequence<Indices...>)
  {
    static_assert(noexcept(E(ext)));
    test_implicit_construction_call<E>(cuda::std::span<typename Extents::value_type, sizeof...(Indices)>(ext), all_ext);
    test_runtime_observers(E(cuda::std::span<typename Extents::value_type, sizeof...(Indices)>(ext)), all_ext);
  }

  template <class E,
            class T,
            size_t N,
            class Extents,
            size_t... Indices,
            cuda::std::enable_if_t<(N != E::rank_dynamic()), int> = 0>
  __host__ __device__ static constexpr void
  test_construction(cuda::std::array<T, N> all_ext, Extents ext, cuda::std::index_sequence<Indices...>)
  {
    static_assert(noexcept(E(ext)));
    test_runtime_observers(E(cuda::std::span<typename Extents::value_type, sizeof...(Indices)>(ext)), all_ext);
  }
};

template <class E>
struct implicit_construction
{
  bool value;
  __host__ __device__ implicit_construction(E)
      : value(true)
  {}
  template <class T>
  __host__ __device__ implicit_construction(T)
      : value(false)
  {}
};

int main(int, char**)
{
  test_index_type_combo<SpanCtorTest>();
#if TEST_STD_VER >= 2020
  static_assert(test_index_type_combo<SpanCtorTest>(), "");
#endif // TEST_STD_VER >= 2020

  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  using E                             = cuda::std::extents<int, 1, D, 3, D>;

  // check can't construct from too few arguments
  static_assert(!cuda::std::is_constructible<E, cuda::std::span<int, 1>>::value,
                "extents constructible from illegal arguments");
  // check can't construct from rank_dynamic < #args < rank
  static_assert(!cuda::std::is_constructible<E, cuda::std::span<int, 3>>::value,
                "extents constructible from illegal arguments");
  // check can't construct from too many arguments
  static_assert(!cuda::std::is_constructible<E, cuda::std::span<int, 5>>::value,
                "extents constructible from illegal arguments");

  // test implicit construction fails from span and array if all extents are given
  cuda::std::array<int, 5> a5{3, 4, 5, 6, 7};
  cuda::std::span<int, 5> s5(a5.data(), 5);
  // check that explicit construction works, i.e. no error
  static_assert(cuda::std::is_constructible<cuda::std::extents<int, D, D, 5, D, D>, decltype(s5)>::value,
                "extents unexpectectly not constructible");
  // check that implicit construction doesn't work
  assert((implicit_construction<cuda::std::extents<int, D, D, 5, D, D>>(s5).value == false));

  // test construction fails from types not convertible to index_type but convertible to other integer types
  static_assert(cuda::std::is_convertible<IntType, int>::value,
                "Test helper IntType unexpectedly not convertible to int");
  static_assert(!cuda::std::is_constructible<cuda::std::extents<unsigned long, D>, cuda::std::span<IntType, 1>>::value,
                "extents constructible from illegal arguments");

  // index_type is not nothrow constructible
  static_assert(cuda::std::is_convertible<IntType, unsigned char>::value, "");
  static_assert(cuda::std::is_convertible<const IntType&, unsigned char>::value, "");
  static_assert(!cuda::std::is_nothrow_constructible<unsigned char, const IntType&>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::dextents<unsigned char, 2>, cuda::std::span<IntType, 2>>::value,
                "");

  // convertible from non-const to index_type but not  from const
  static_assert(cuda::std::is_convertible<IntTypeNC, int>::value, "");
  static_assert(!cuda::std::is_convertible<const IntTypeNC&, int>::value, "");
  static_assert(cuda::std::is_nothrow_constructible<int, IntTypeNC>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::dextents<int, 2>, cuda::std::span<IntTypeNC, 2>>::value, "");
  return 0;
}
