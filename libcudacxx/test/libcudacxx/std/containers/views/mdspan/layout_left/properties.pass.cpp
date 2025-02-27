//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// namespace std {
//   template<class Extents>
//   class layout_left::mapping {
//
//     ...
//     static constexpr bool is_always_unique() noexcept { return true; }
//     static constexpr bool is_always_exhaustive() noexcept { return true; }
//     static constexpr bool is_always_strided() noexcept { return true; }
//
//     static constexpr bool is_unique() noexcept { return true; }
//     static constexpr bool is_exhaustive() noexcept { return true; }
//     static constexpr bool is_strided() noexcept { return true; }
//     ...
//   };
// }

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class E>
__host__ __device__ constexpr void test_layout_mapping_left()
{
  using M = cuda::std::layout_left::template mapping<E>;
  assert(M::is_unique() == true);
  assert(M::is_exhaustive() == true);
  assert(M::is_strided() == true);
  assert(M::is_always_unique() == true);
  assert(M::is_always_exhaustive() == true);
  assert(M::is_always_strided() == true);
  ASSERT_NOEXCEPT(cuda::std::declval<M>().is_unique());
  ASSERT_NOEXCEPT(cuda::std::declval<M>().is_exhaustive());
  ASSERT_NOEXCEPT(cuda::std::declval<M>().is_strided());
  ASSERT_NOEXCEPT(M::is_always_unique());
  ASSERT_NOEXCEPT(M::is_always_exhaustive());
  ASSERT_NOEXCEPT(M::is_always_strided());
}

__host__ __device__ constexpr bool test()
{
  constexpr size_t D = cuda::std::dynamic_extent;
  test_layout_mapping_left<cuda::std::extents<int>>();
  test_layout_mapping_left<cuda::std::extents<char, 4, 5>>();
  test_layout_mapping_left<cuda::std::extents<unsigned, D, 4>>();
  test_layout_mapping_left<cuda::std::extents<size_t, D, D, D, D>>();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
