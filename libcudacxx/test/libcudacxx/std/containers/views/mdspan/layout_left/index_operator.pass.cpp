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
__host__ __device__ constexpr bool check_operator_constraints(Mapping, Indices...)
{
  return true;
}

_CCCL_TEMPLATE(class Mapping, class... Indices)
_CCCL_REQUIRES((!operator_constraints<Mapping, Indices...>) )
__host__ __device__ constexpr bool check_operator_constraints(Mapping, Indices...)
{
  return false;
}

template <class M, class T, class... Args>
__host__ __device__ constexpr void iterate_left(M m, T& count, Args... args)
{
  using extents = typename M::extents_type;
  if constexpr (extents::rank() == sizeof...(Args))
  {
    static_assert(noexcept(m(args...)));
    assert(count == m(args...));
    count++;
  }
  else
  {
    constexpr int r = static_cast<int>(extents::rank()) - 1 - static_cast<int>(sizeof...(Args));
    for (typename M::index_type i = 0; i < m.extents().extent(r); i++)
    {
      iterate_left(m, count, i, args...);
    }
  }
}

template <class E, class... Args>
__host__ __device__ constexpr void test_iteration(Args... args)
{
  using M = cuda::std::layout_left::mapping<E>;
  M m{E{args...}};

  typename E::index_type count = 0;
  iterate_left(m, count);
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_iteration<cuda::std::extents<int>>();
  test_iteration<cuda::std::extents<unsigned, D>>(1);
  test_iteration<cuda::std::extents<unsigned, D>>(7);
  test_iteration<cuda::std::extents<unsigned, 7>>();
  test_iteration<cuda::std::extents<unsigned, 7, 8>>();
  test_iteration<cuda::std::extents<char, D, D, D, D>>(1, 1, 1, 1);

  // Check operator constraint for number of arguments
  static_assert(check_operator_constraints(
                  cuda::std::layout_left::mapping<cuda::std::extents<int, D>>(cuda::std::extents<int, D>(1)), 0),
                "");
  static_assert(!check_operator_constraints(
                  cuda::std::layout_left::mapping<cuda::std::extents<int, D>>(cuda::std::extents<int, D>(1)), 0, 0),
                "");

  // Check operator constraint for convertibility of arguments to index_type
  static_assert(
    check_operator_constraints(
      cuda::std::layout_left::mapping<cuda::std::extents<int, D>>(cuda::std::extents<int, D>(1)), IntType(0)),
    "");
  static_assert(
    !check_operator_constraints(
      cuda::std::layout_left::mapping<cuda::std::extents<unsigned, D>>(cuda::std::extents<unsigned, D>(1)), IntType(0)),
    "");

  // Check operator constraint for no-throw-constructibility of index_type from arguments
  static_assert(
    !check_operator_constraints(
      cuda::std::layout_left::mapping<cuda::std::extents<unsigned char, D>>(cuda::std::extents<unsigned char, D>(1)),
      IntType(0)),
    "");

  return true;
}

__host__ __device__ constexpr bool test_large()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_iteration<cuda::std::extents<int64_t, D, 8, D, D>>(7, 9, 10);
  test_iteration<cuda::std::extents<int64_t, D, 8, 1, D>>(7, 10);
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
