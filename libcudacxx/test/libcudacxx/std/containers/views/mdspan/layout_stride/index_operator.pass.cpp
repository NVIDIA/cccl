//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// Test default iteration:
//
// template<class... Indices>
//   constexpr index_type operator()(Indices...) const noexcept;
//
// Constraints:
//   * sizeof...(Indices) == extents_type::rank() is true,
//   * (is_convertible_v<Indices, index_type> && ...) is true, and
//   * (is_nothrow_constructible_v<index_type, Indices> && ...) is true.
//
// Preconditions:
//   * extents_type::index-cast(i) is a multidimensional index in extents_.

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/mdspan>

#include "../ConvertibleToIntegral.h"
#include "test_macros.h"

template <class Mapping, class... Indices>
_CCCL_CONCEPT operator_constraints = _CCCL_REQUIRES_EXPR((Mapping, variadic Indices), Mapping m, Indices... idxs)(
  _Same_as(typename Mapping::index_type) m(idxs...));

_CCCL_TEMPLATE(class Mapping, class... Indices)
_CCCL_REQUIRES(operator_constraints<Mapping, Indices...>)
__host__ __device__ constexpr bool check_operator_constraints(Mapping m, Indices... idxs)
{
  (void) m(idxs...);
  return true;
}

_CCCL_TEMPLATE(class Mapping, class... Indices)
_CCCL_REQUIRES((!operator_constraints<Mapping, Indices...>) )
__host__ __device__ constexpr bool check_operator_constraints(Mapping, Indices...)
{
  return false;
}

template <class M, class... Args, size_t... Pos>
__host__ __device__ constexpr size_t get_strides(
  const cuda::std::array<int, M::extents_type::rank()>& strides, cuda::std::index_sequence<Pos...>, Args... args)
{
  return (size_t{0} + ... + (args * strides[Pos]));
}

template <class M, class... Args, cuda::std::enable_if_t<(M::extents_type::rank() == sizeof...(Args)), int> = 0>
__host__ __device__ constexpr void
iterate_stride(M m, const cuda::std::array<int, M::extents_type::rank()>& strides, Args... args)
{
  static_assert(noexcept(m(args...)));
  const size_t expected_val =
    get_strides<M>(strides, cuda::std::make_index_sequence<M::extents_type::rank()>(), args...);
  assert(expected_val == static_cast<size_t>(m(args...)));
}

template <class M, class... Args, cuda::std::enable_if_t<(M::extents_type::rank() != sizeof...(Args)), int> = 0>
__host__ __device__ constexpr void
iterate_stride(M m, const cuda::std::array<int, M::extents_type::rank()>& strides, Args... args)
{
  constexpr size_t r = sizeof...(Args);
  for (typename M::index_type i = 0; i < m.extents().extent(r); i++)
  {
    iterate_stride(m, strides, i, args...);
  }
}

template <class E, class... Args>
__host__ __device__ constexpr void test_iteration(cuda::std::array<int, E::rank()> strides, Args... args)
{
  using M = cuda::std::layout_stride::mapping<E>;
  M m(E(args...), strides);

  iterate_stride(m, strides);
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_iteration<cuda::std::extents<int>>(cuda::std::array<int, 0>{});
  test_iteration<cuda::std::extents<unsigned, D>>(cuda::std::array<int, 1>{2}, 1);
  test_iteration<cuda::std::extents<unsigned, D>>(cuda::std::array<int, 1>{3}, 7);
  test_iteration<cuda::std::extents<unsigned, 7>>(cuda::std::array<int, 1>{4});
  test_iteration<cuda::std::extents<unsigned, 7, 8>>(cuda::std::array<int, 2>{25, 3});
  test_iteration<cuda::std::extents<char, D, D, D, D>>(cuda::std::array<int, 4>{1, 1, 1, 1}, 1, 1, 1, 1);

  // Check operator constraint for number of arguments
  static_assert(check_operator_constraints(
    cuda::std::layout_stride::mapping<cuda::std::extents<int, D>>(
      cuda::std::extents<int, D>(1), cuda::std::array<int, 1>{1}),
    0));
  static_assert(!check_operator_constraints(
    cuda::std::layout_stride::mapping<cuda::std::extents<int, D>>(
      cuda::std::extents<int, D>(1), cuda::std::array<int, 1>{1}),
    0,
    0));

  // Check operator constraint for convertibility of arguments to index_type
  static_assert(check_operator_constraints(
    cuda::std::layout_stride::mapping<cuda::std::extents<int, D>>(
      cuda::std::extents<int, D>(1), cuda::std::array<int, 1>{1}),
    IntType(0)));
  static_assert(!check_operator_constraints(
    cuda::std::layout_stride::mapping<cuda::std::extents<unsigned, D>>(
      cuda::std::extents<unsigned, D>(1), cuda::std::array<int, 1>{1}),
    IntType(0)));

  // Check operator constraint for no-throw-constructibility of index_type from arguments
  static_assert(!check_operator_constraints(
    cuda::std::layout_stride::mapping<cuda::std::extents<unsigned char, D>>(
      cuda::std::extents<unsigned char, D>(1), cuda::std::array<int, 1>{1}),
    IntType(0)));

  return true;
}

__host__ __device__ constexpr bool test_large()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_iteration<cuda::std::extents<int64_t, D, 8, D, D>>(cuda::std::array<int, 4>{2000, 2, 20, 200}, 7, 9, 10);
  test_iteration<cuda::std::extents<int64_t, D, 8, 1, D>>(cuda::std::array<int, 4>{2000, 20, 20, 200}, 7, 10);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  // The large test iterates over ~10k loop indices.
  // With assertions enabled this triggered the maximum default limit
  // for steps in consteval expressions. Assertions roughly double the
  // total number of instructions, so this was already close to the maximum.
  test_large();
  return 0;
}
