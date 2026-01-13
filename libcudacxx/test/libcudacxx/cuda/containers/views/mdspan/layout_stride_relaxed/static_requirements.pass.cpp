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

// Test static requirements for cuda::layout_stride_relaxed::mapping
//
// A type M meets the layout mapping requirements if:
//    - M models copyable,
//    - is_nothrow_move_constructible_v<M> is true,
//    - is_nothrow_move_assignable_v<M> is true,
//    - is_nothrow_swappable_v<M> is true

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

// Common requirements of all layout mappings
template <class M, size_t... Idxs>
__host__ __device__ void test_mapping_requirements(cuda::std::index_sequence<Idxs...>)
{
  using E = typename M::extents_type;
  static_assert(cuda::std::__is_cuda_std_extents_v<E>);
  static_assert(cuda::std::is_copy_constructible_v<M>);
  static_assert(cuda::std::is_nothrow_move_constructible_v<M>);
  static_assert(cuda::std::is_nothrow_move_assignable_v<M>);
  static_assert(cuda::std::is_nothrow_swappable_v<M>);
  static_assert(cuda::std::is_same_v<typename M::index_type, typename E::index_type>);
  static_assert(cuda::std::is_same_v<typename M::size_type, typename E::size_type>);
  static_assert(cuda::std::is_same_v<typename M::rank_type, typename E::rank_type>);
  static_assert(cuda::std::is_same_v<typename M::layout_type, cuda::layout_stride_relaxed>);
  static_assert(cuda::std::is_same_v<typename M::layout_type::template mapping<E>, M>);
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<M>().extents()), const E&>::value, "");
  // Note: layout_stride_relaxed uses intptr_t for strides, not index_type
  static_assert(
    cuda::std::is_same<decltype(cuda::std::declval<M>().strides()),
                       cuda::std::array<cuda::std::intptr_t, E::rank()>>::value,
    "");
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<M>()(Idxs...)), typename M::index_type>::value, "");
  static_assert(
    cuda::std::is_same<decltype(cuda::std::declval<M>().required_span_size()), typename M::index_type>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<M>().is_unique()), bool>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<M>().is_exhaustive()), bool>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<M>().is_strided()), bool>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<M>().stride(0)), typename M::index_type>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<M>().offset()), cuda::std::intptr_t>::value, "");
  static_assert(cuda::std::is_same<decltype(M::is_always_unique()), bool>::value, "");
  static_assert(cuda::std::is_same<decltype(M::is_always_exhaustive()), bool>::value, "");
  static_assert(cuda::std::is_same<decltype(M::is_always_strided()), bool>::value, "");

  // layout_stride_relaxed specific: is_always_* are all false
  static_assert(M::is_always_unique() == false);
  static_assert(M::is_always_exhaustive() == false);
  static_assert(M::is_always_strided() == false);
}

template <class L, class E>
__host__ __device__ void test_layout_mapping_requirements()
{
  using M = typename L::template mapping<E>;
  test_mapping_requirements<M>(cuda::std::make_index_sequence<E::rank()>());
}

template <class E>
__host__ __device__ void test_layout_mapping_stride_relaxed()
{
  test_layout_mapping_requirements<cuda::layout_stride_relaxed, E>();
}

int main(int, char**)
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_layout_mapping_stride_relaxed<cuda::std::extents<int>>();
  test_layout_mapping_stride_relaxed<cuda::std::extents<char, 4, 5>>();
  test_layout_mapping_stride_relaxed<cuda::std::extents<unsigned, D, 4>>();
  test_layout_mapping_stride_relaxed<cuda::std::extents<size_t, D, D, D, D>>();
  return 0;
}
