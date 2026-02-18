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

// Test default construction:
//
// constexpr mapping() noexcept;
//
// Effects: Direct-non-list-initializes extents_ with extents_type(), and for all d in the range [0, rank_),
//          direct-non-list-initializes strides_[d] with layout_right::mapping<extents_type>().stride(d).
//          offset_ is initialized to 0.

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"

template <class E, cuda::std::enable_if_t<E::rank() != 0, int> = 0>
__host__ __device__ constexpr void test_construction()
{
  using M = cuda::layout_stride_relaxed::mapping<E>;
  static_assert(noexcept(M{}));
  M m{};
  E e{};

  // check correct extents are returned
  static_assert(noexcept(m.extents()));
  assert(m.extents() == e);

  // check offset is zero
  assert(m.offset() == 0);

  // check required_span_size()
  typename E::index_type expected_size = 1;
  for (typename E::rank_type r = 0; r < E::rank(); r++)
  {
    expected_size *= e.extent(r);
  }
  assert(m.required_span_size() == expected_size);

  // check strides: uses layout_right strides by default
  static_assert(noexcept(m.strides()));

  if constexpr (E::rank() != 0)
  {
    auto strides_obj = m.strides();
    cuda::std::layout_right::mapping<E> m_right{};
    for (typename E::rank_type r = 0; r < E::rank(); r++)
    {
      assert(cuda::std::cmp_equal(m.stride(r), m_right.stride(r)));
      assert(cuda::std::cmp_equal(strides_obj.stride(r), m_right.stride(r)));
    }
  }
}

template <class E, cuda::std::enable_if_t<E::rank() == 0, int> = 0>
__host__ __device__ constexpr void test_construction()
{
  using M = cuda::layout_stride_relaxed::mapping<E>;
  static_assert(noexcept(M{}));
  M m{};
  E e{};

  // check correct extents are returned
  static_assert(noexcept(m.extents()));
  assert(m.extents() == e);

  // check offset is zero
  assert(m.offset() == 0);

  // check required_span_size()
  typename E::index_type expected_size = 1;
  assert(m.required_span_size() == expected_size);

  // check strides: node stride function is constrained on rank>0
  static_assert(noexcept(m.strides()));
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_construction<cuda::std::extents<int>>();
  test_construction<cuda::std::extents<unsigned, D>>();
  test_construction<cuda::std::extents<unsigned, 7>>();
  test_construction<cuda::std::extents<unsigned, 0>>();
  test_construction<cuda::std::extents<unsigned, 7, 8>>();
  test_construction<cuda::std::extents<int64_t, D, 8, D, D>>();

  // Additional edge cases for zero extents
  test_construction<cuda::std::extents<int, 0>>();
  test_construction<cuda::std::extents<int, 0, 0>>();
  test_construction<cuda::std::extents<int, 0, 5>>();
  test_construction<cuda::std::extents<int, 5, 0>>();
  test_construction<cuda::std::extents<int, 3, 0, 4>>();
  test_construction<cuda::std::extents<int64_t, 0, D, 0, D>>();
  test_construction<cuda::std::extents<int64_t, D, 0, D, 0>>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
