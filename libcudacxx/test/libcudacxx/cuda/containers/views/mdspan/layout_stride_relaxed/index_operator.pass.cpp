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

// Test index operator:
//
// template<class... Indices>
//   constexpr index_type operator()(Indices...) const noexcept;
//
// Constraints:
//   * sizeof...(Indices) == extents_type::rank() is true,
//   * (is_convertible_v<Indices, index_type> && ...) is true, and
//   * (is_nothrow_constructible_v<index_type, Indices> && ...) is true.
//
// Returns: offset_ + (indices * strides_[r] + ...)

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "../ConvertibleToIntegral.h"
#include "test_macros.h"

using cuda::std::intptr_t;

template <class Mapping, class... Indices>
_CCCL_CONCEPT operator_constraints = _CCCL_REQUIRES_EXPR((Mapping, variadic Indices), Mapping m, Indices... idxs)(
  _Same_as(typename Mapping::index_type) m(idxs...));

template <class Mapping, class... Indices>
__host__ __device__ constexpr bool check_operator_constraints(Mapping m, Indices... idxs)
{
  if constexpr (operator_constraints<Mapping, Indices...>)
  {
    (void) m(idxs...);
    return true;
  }
  else
  {
    return false;
  }
}

template <class M, class... Args, size_t... Pos>
__host__ __device__ constexpr intptr_t get_expected_index(
  const cuda::std::array<intptr_t, M::extents_type::rank()>& strides,
  intptr_t offset,
  cuda::std::index_sequence<Pos...>,
  Args... args)
{
  return offset + (static_cast<intptr_t>(0) + ... + (static_cast<intptr_t>(args) * strides[Pos]));
}

template <class M, class... Args, cuda::std::enable_if_t<(M::extents_type::rank() == sizeof...(Args)), int> = 0>
__host__ __device__ constexpr void
iterate_stride(M m, const cuda::std::array<intptr_t, M::extents_type::rank()>& strides, intptr_t offset, Args... args)
{
  static_assert(noexcept(m(args...)));
  const intptr_t expected_val =
    get_expected_index<M>(strides, offset, cuda::std::make_index_sequence<M::extents_type::rank()>(), args...);
  assert(expected_val == static_cast<intptr_t>(m(args...)));
}

template <class M, class... Args, cuda::std::enable_if_t<(M::extents_type::rank() != sizeof...(Args)), int> = 0>
__host__ __device__ constexpr void
iterate_stride(M m, const cuda::std::array<intptr_t, M::extents_type::rank()>& strides, intptr_t offset, Args... args)
{
  constexpr size_t r = sizeof...(Args);
  for (typename M::index_type i = 0; i < m.extents().extent(r); i++)
  {
    iterate_stride(m, strides, offset, args..., i); // append i after args to maintain dimension order
  }
}

template <class E, class... Extents>
__host__ __device__ constexpr void
test_iteration(cuda::std::array<intptr_t, E::rank()> strides, intptr_t offset, Extents... extents)
{
  using M            = cuda::layout_stride_relaxed::mapping<E>;
  using strides_type = typename M::strides_type;
  using index_type   = typename M::index_type;
  using offset_type  = typename M::offset_type;
  M m(E(static_cast<index_type>(extents)...), strides_type(strides), static_cast<offset_type>(offset));

  iterate_stride(m, strides, offset);
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // Basic cases with zero offset
  test_iteration<cuda::std::extents<int>>(cuda::std::array<intptr_t, 0>{}, 0);
  test_iteration<cuda::std::extents<unsigned, D>>(cuda::std::array<intptr_t, 1>{2}, 0, 1);
  test_iteration<cuda::std::extents<unsigned, D>>(cuda::std::array<intptr_t, 1>{3}, 0, 7);
  test_iteration<cuda::std::extents<unsigned, 7>>(cuda::std::array<intptr_t, 1>{4}, 0);
  test_iteration<cuda::std::extents<unsigned, 7, 8>>(cuda::std::array<intptr_t, 2>{25, 3}, 0);
  test_iteration<cuda::std::extents<signed char, D, D, D, D>>(cuda::std::array<intptr_t, 4>{1, 1, 1, 1}, 0, 1, 1, 1, 1);

  // Cases with non-zero offset
  test_iteration<cuda::std::extents<int>>(cuda::std::array<intptr_t, 0>{}, 5);
  test_iteration<cuda::std::extents<unsigned, D>>(cuda::std::array<intptr_t, 1>{2}, 10, 1);
  test_iteration<cuda::std::extents<unsigned, 7>>(cuda::std::array<intptr_t, 1>{4}, 20);
  test_iteration<cuda::std::extents<unsigned, 7, 8>>(cuda::std::array<intptr_t, 2>{25, 3}, 100);

  // Cases with negative strides (reverse iteration)
  // For a 1D array with negative stride, we need offset to point to the last element
  test_iteration<cuda::std::extents<int, 4>>(cuda::std::array<intptr_t, 1>{-1}, 3);

  // Check operator constraint for number of arguments
  {
    using M = cuda::layout_stride_relaxed::mapping<cuda::std::extents<int, D>>;
    static_assert(check_operator_constraints(M(cuda::std::extents<int, D>(1), typename M::strides_type(1)), 0));
    static_assert(!check_operator_constraints(M(cuda::std::extents<int, D>(1), typename M::strides_type(1)), 0, 0));
  }

  // Check operator constraint for convertibility of arguments to index_type
  {
    using M1 = cuda::layout_stride_relaxed::mapping<cuda::std::extents<int, D>>;
    using M2 = cuda::layout_stride_relaxed::mapping<cuda::std::extents<unsigned, D>>;
    static_assert(
      check_operator_constraints(M1(cuda::std::extents<int, D>(1), typename M1::strides_type(1)), IntType(0)));
    static_assert(
      !check_operator_constraints(M2(cuda::std::extents<unsigned, D>(1), typename M2::strides_type(1)), IntType(0)));
  }

  // Check operator constraint for no-throw-constructibility of index_type from arguments
  {
    using M = cuda::layout_stride_relaxed::mapping<cuda::std::extents<unsigned char, D>>;
    static_assert(
      !check_operator_constraints(M(cuda::std::extents<unsigned char, D>(1), typename M::strides_type(1)), IntType(0)));
  }

  return true;
}

__host__ __device__ constexpr bool test_large()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_iteration<cuda::std::extents<int64_t, D, 8, D, D>>(cuda::std::array<intptr_t, 4>{2000, 2, 20, 200}, 0, 7, 9, 10);
  test_iteration<cuda::std::extents<int64_t, D, 8, 1, D>>(cuda::std::array<intptr_t, 4>{2000, 20, 20, 200}, 0, 7, 10);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  // The large test iterates over ~10k loop indices.
  test_large();
  return 0;
}
