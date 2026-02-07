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

// Test layout_stride_relaxed::mapping with integral_constant as _OffsetType:

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::ptrdiff_t;

template <ptrdiff_t Offset>
using contant_offset_t = cuda::std::integral_constant<ptrdiff_t, Offset>;

// Helper alias for mapping with constant offset
template <class E, ptrdiff_t Offset>
using mapping_with_constant_offset =
  cuda::layout_stride_relaxed::mapping<E, cuda::steps<E::rank()>, contant_offset_t<Offset>>;

__host__ __device__ constexpr void test_basic_properties()
{
  using Extents         = cuda::std::extents<int, 4>;
  constexpr auto Offset = 7;
  using M               = mapping_with_constant_offset<Extents, Offset>;
  static_assert(cuda::std::is_same_v<typename M::offset_type, contant_offset_t<Offset>>);
  static_assert(M::offset_type::value == Offset);
  static_assert(cuda::std::is_copy_constructible_v<M>);
  static_assert(cuda::std::is_nothrow_move_constructible_v<M>);
  static_assert(cuda::std::is_nothrow_move_assignable_v<M>);
}

__host__ __device__ constexpr void test_explicit_construction_1d()
{
  using Extents         = cuda::std::extents<int, 4>;
  constexpr auto Offset = 7;
  using M               = mapping_with_constant_offset<Extents, Offset>;
  Extents ext{};
  typename M::strides_type strides(2);
  typename M::offset_type offset{};
  M m(ext, strides, offset);
  assert(m.offset() == Offset);
  assert(m.strides().stride(0) == 2);
}

__host__ __device__ constexpr void test_access_1d()
{
  // backward access
  using E = cuda::std::extents<int, 5>;
  using M = mapping_with_constant_offset<E, 4>;
  M m(E{}, typename M::strides_type(-1), typename M::offset_type{});
  assert(m.offset() == 4);
  assert(m(0) == 4);
  assert(m(1) == 3);
  assert(m(2) == 2);
  assert(m(3) == 1);
  assert(m(4) == 0);
}

__host__ __device__ constexpr void test_required_span_size()
{
  using E = cuda::std::extents<int, 4>;
  using M = mapping_with_constant_offset<E, 10>;
  M m(E{}, typename M::strides_type(1), typename M::offset_type{});
  assert(m.required_span_size() == 14); // 10 + 4 = 14
}

__host__ __device__ constexpr void test_is_strided_zero_offset()
{
  using E = cuda::std::extents<int, 4>;
  using M = mapping_with_constant_offset<E, 0>;
  M m(E{}, typename M::strides_type(1), typename M::offset_type{});
  assert(m.offset() == 0);
  assert(m.is_strided() == true);
  assert(m.is_always_strided() == true);
}

__host__ __device__ constexpr void test_is_strided_nonzero_offset()
{
  using E = cuda::std::extents<int, 4>;
  using M = mapping_with_constant_offset<E, 10>;
  M m(E{}, typename M::strides_type(1), typename M::offset_type{});
  assert(m.offset() == 10);
  assert(m.is_strided() == false);
  assert(m.is_always_strided() == false);
}

__host__ __device__ constexpr bool test()
{
  test_basic_properties();
  test_explicit_construction_1d();
  test_access_1d();
  test_required_span_size();
  test_is_strided_zero_offset();
  test_is_strided_nonzero_offset();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
