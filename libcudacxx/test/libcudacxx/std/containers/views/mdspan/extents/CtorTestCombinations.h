//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/mdspan>
#include <cuda/std/span>

#include "../ConvertibleToIntegral.h"
#include "test_macros.h"

// Helper file to implement combinatorial testing of extents constructor
//
// cuda::std::extents can be constructed from just indices, a cuda::std::array, or a cuda::std::span
// In each of those cases one can either provide all extents, or just the dynamic ones
// If constructed from cuda::std::span, the span needs to have a static extent
// Furthermore, the indices/array/span can have integer types other than index_type

template <class E, class AllExtents, cuda::std::enable_if_t<E::rank() != 0, int> = 0>
__host__ __device__ constexpr void test_runtime_observers(E ext, AllExtents expected)
{
  for (typename E::rank_type r = 0; r < ext.rank(); r++)
  {
    static_assert(cuda::std::is_same_v<decltype(ext.extent(0)), typename E::index_type>);
    static_assert(noexcept(ext.extent(0)));
    assert(ext.extent(r) == static_cast<typename E::index_type>(expected[r]));
  }
}

template <class E, class AllExtents, cuda::std::enable_if_t<E::rank() == 0, int> = 0>
__host__ __device__ constexpr void test_runtime_observers(E ext, AllExtents expected)
{
  // Nothing to do here
}

template <class E, class AllExtents>
__host__ __device__ constexpr void test_implicit_construction_call(E e, AllExtents all_ext)
{
  test_runtime_observers(e, all_ext);
}

template <class E, class Test, class AllExtents, cuda::std::enable_if_t<E::rank() == 0, int> = 0>
__host__ __device__ constexpr void test_construction(AllExtents all_ext)
{
  // test construction from all extents
  Test::template test_construction<E>(all_ext, all_ext, cuda::std::make_index_sequence<E::rank()>());

  // test construction from just dynamic extents
  // create an array of just the extents corresponding to dynamic values
  cuda::std::array<typename AllExtents::value_type, E::rank_dynamic()> dyn_ext{0};
  Test::template test_construction<E>(all_ext, dyn_ext, cuda::std::make_index_sequence<E::rank_dynamic()>());
}

template <class E, class Test, class AllExtents, cuda::std::enable_if_t<E::rank() != 0, int> = 0>
__host__ __device__ constexpr void test_construction(AllExtents all_ext)
{
  // test construction from all extents
  Test::template test_construction<E>(all_ext, all_ext, cuda::std::make_index_sequence<E::rank()>());

  // test construction from just dynamic extents
  // create an array of just the extents corresponding to dynamic values
  cuda::std::array<typename AllExtents::value_type, E::rank_dynamic()> dyn_ext{0};
  size_t dynamic_idx = 0;
  for (size_t r = 0; r < E::rank(); r++)
  {
    if (E::static_extent(r) == cuda::std::dynamic_extent)
    {
      dyn_ext[dynamic_idx] = all_ext[r];
      dynamic_idx++;
    }
  }
  Test::template test_construction<E>(all_ext, dyn_ext, cuda::std::make_index_sequence<E::rank_dynamic()>());
}

template <class T, class TArg, class Test>
__host__ __device__ constexpr void test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  test_construction<cuda::std::extents<T>, Test>(cuda::std::array<TArg, 0>{});

  test_construction<cuda::std::extents<T, 3>, Test>(cuda::std::array<TArg, 1>{3});
  test_construction<cuda::std::extents<T, D>, Test>(cuda::std::array<TArg, 1>{3});

  test_construction<cuda::std::extents<T, 3, 7>, Test>(cuda::std::array<TArg, 2>{3, 7});
  test_construction<cuda::std::extents<T, 3, D>, Test>(cuda::std::array<TArg, 2>{3, 7});
  test_construction<cuda::std::extents<T, D, 7>, Test>(cuda::std::array<TArg, 2>{3, 7});
  test_construction<cuda::std::extents<T, D, D>, Test>(cuda::std::array<TArg, 2>{3, 7});

  test_construction<cuda::std::extents<T, 3, 7, 9>, Test>(cuda::std::array<TArg, 3>{3, 7, 9});
  test_construction<cuda::std::extents<T, 3, 7, D>, Test>(cuda::std::array<TArg, 3>{3, 7, 9});
  test_construction<cuda::std::extents<T, 3, D, D>, Test>(cuda::std::array<TArg, 3>{3, 7, 9});
  test_construction<cuda::std::extents<T, D, 7, D>, Test>(cuda::std::array<TArg, 3>{3, 7, 9});
  test_construction<cuda::std::extents<T, D, D, D>, Test>(cuda::std::array<TArg, 3>{3, 7, 9});
  test_construction<cuda::std::extents<T, 3, D, 9>, Test>(cuda::std::array<TArg, 3>{3, 7, 9});
  test_construction<cuda::std::extents<T, D, D, 9>, Test>(cuda::std::array<TArg, 3>{3, 7, 9});
  test_construction<cuda::std::extents<T, D, 7, 9>, Test>(cuda::std::array<TArg, 3>{3, 7, 9});

  test_construction<cuda::std::extents<T, 1, 2, 3, 4, 5, 6, 7, 8, 9>, Test>(
    cuda::std::array<TArg, 9>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  test_construction<cuda::std::extents<T, D, 2, 3, D, 5, D, 7, D, 9>, Test>(
    cuda::std::array<TArg, 9>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  test_construction<cuda::std::extents<T, D, D, D, D, D, D, D, D, D>, Test>(
    cuda::std::array<TArg, 9>{1, 2, 3, 4, 5, 6, 7, 8, 9});
}

template <class Test>
__host__ __device__ constexpr bool test_index_type_combo()
{
  test<int, int, Test>();
  test<int, size_t, Test>();
  test<unsigned, int, Test>();
  test<char, size_t, Test>();
  test<long long, unsigned, Test>();
  test<size_t, int, Test>();
  test<size_t, size_t, Test>();
  test<int, IntType, Test>();
  test<char, IntType, Test>();
  return true;
}
