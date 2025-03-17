//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// A type M meets the layout mapping requirements if
//    - M models copyable and equality_comparable,
//    - is_nothrow_move_constructible<M> is true,
//    - is_nothrow_move_assignable<M> is true,
//    - is_nothrow_swappable<M> is true, and
//
// the following types and expressions are well-formed and have the specified semantics.
//
//  typename M::extents_type
//    Result: A type that is a specialization of extents.
//
//  typename M::index_type
//    Result: typename M::extents_type::index_type.
//
//  typename M::rank_type
//    Result: typename M::extents_type::rank_type.
//
//  typename M::layout_type
//    Result: A type MP that meets the layout mapping policy requirements ([mdspan.layout.policy.reqmts]) and for which
//    is-mapping-of<MP, M> is true.
//
//  m.extents()
//    Result: const typename M::extents_type&
//
//  m(i...)
//    Result: typename M::index_type
//    Returns: A nonnegative integer less than numeric_limits<typename M::index_type>::max() and less than or equal to
//    numeric_limits<size_t>::max().
//
//  m(i...) == m(static_cast<typename M::index_type>(i)...)
//    Result: bool
//    Returns: true
//
//  m.required_span_size()
//    Result: typename M::index_type
//    Returns: If the size of the multidimensional index space m.extents() is 0, then 0, else 1 plus the maximum value
//    of m(i...) for all i.
//
//  m.is_unique()
//    Result: bool
//    Returns: true only if for every i and j where (i != j || ...) is true, m(i...) != m(j...) is true.
//
//  m.is_exhaustive()
//    Result: bool
//    Returns: true only if for all k in the range [0, m.required_span_size()) there exists an i such that m(i...)
//    equals k.
//
//  m.is_strided()
//    Result: bool
//    Returns: true only if for every rank index r of m.extents() there exists an integer
//             sr such that, for all i where (i+dr) is a multidimensional index in m.extents() ([mdspan.overview]),
//             m((i + dr)...) - m(i...) equals sr
//
//  m.stride(r)
//    Preconditions: m.is_strided() is true.
//    Result: typename M::index_type
//    Returns: sr as defined in m.is_strided() above.
//
//  M::is_always_unique()
//    Result: A constant expression ([expr.const]) of type bool.
//    Returns: true only if m.is_unique() is true for all possible objects m of type M.
//
//  M::is_always_exhaustive()
//    Result: A constant expression ([expr.const]) of type bool.
//    Returns: true only if m.is_exhaustive() is true for all possible objects m of type M.
//
//  M::is_always_strided()
//    Result: A constant expression ([expr.const]) of type bool.
//    Returns: true only if m.is_strided() is true for all possible objects m of type M.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class M, class E = typename M::extents_type, cuda::std::enable_if_t<(E::rank() > 0), int> = 0>
__host__ __device__ constexpr void test_mapping_stride_requirements()
{
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<M>().stride(0)), typename M::index_type>::value, "");
}

template <class M, class E = typename M::extents_type, cuda::std::enable_if_t<(E::rank() == 0), int> = 0>
__host__ __device__ constexpr void test_mapping_stride_requirements()
{}

// Common requirements of all layout mappings
template <class M, size_t... Idxs>
__host__ __device__ void test_mapping_requirements(cuda::std::index_sequence<Idxs...>)
{
  using E = typename M::extents_type;
  static_assert(cuda::std::__mdspan_detail::__is_extents<E>::value, "");
  static_assert(cuda::std::is_copy_constructible<M>::value, "");
  static_assert(cuda::std::is_nothrow_move_constructible<M>::value, "");
  static_assert(cuda::std::is_nothrow_move_assignable<M>::value, "");
  static_assert(cuda::std::is_nothrow_swappable<M>::value, "");
  static_assert(cuda::std::is_same_v<typename M::index_type, typename E::index_type>);
  static_assert(cuda::std::is_same_v<typename M::size_type, typename E::size_type>);
  static_assert(cuda::std::is_same_v<typename M::rank_type, typename E::rank_type>);
  static_assert(cuda::std::is_same_v<typename M::layout_type, cuda::std::layout_left>);
  static_assert(cuda::std::is_same_v<typename M::layout_type::template mapping<E>, M>);
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<M>().extents()), const E&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<M>()(Idxs...)), typename M::index_type>::value, "");
  static_assert(
    cuda::std::is_same<decltype(cuda::std::declval<M>().required_span_size()), typename M::index_type>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<M>().is_unique()), bool>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<M>().is_exhaustive()), bool>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<M>().is_strided()), bool>::value, "");
  test_mapping_stride_requirements<M>();
  static_assert(cuda::std::is_same<decltype(M::is_always_unique()), bool>::value, "");
  static_assert(cuda::std::is_same<decltype(M::is_always_exhaustive()), bool>::value, "");
  static_assert(cuda::std::is_same<decltype(M::is_always_strided()), bool>::value, "");
}

template <class L, class E>
__host__ __device__ void test_layout_mapping_requirements()
{
  using M = typename L::template mapping<E>;
  test_mapping_requirements<M>(cuda::std::make_index_sequence<E::rank()>());
}

template <class E>
__host__ __device__ void test_layout_mapping_left()
{
  test_layout_mapping_requirements<cuda::std::layout_left, E>();
}

int main(int, char**)
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_layout_mapping_left<cuda::std::extents<int>>();
  test_layout_mapping_left<cuda::std::extents<char, 4, 5>>();
  test_layout_mapping_left<cuda::std::extents<unsigned, D, 4>>();
  test_layout_mapping_left<cuda::std::extents<size_t, D, D, D, D>>();
  return 0;
}
