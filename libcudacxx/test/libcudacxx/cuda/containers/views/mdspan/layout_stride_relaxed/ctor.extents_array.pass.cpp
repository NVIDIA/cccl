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

// template<class OtherIndexType>
//  constexpr mapping(const extents_type& e, array<OtherIndexType, rank_> s, offset_type offset = 0) noexcept;
//
// Constraints:
//    - is_convertible_v<const OtherIndexType&, offset_type> is true, and
//    - is_nothrow_constructible_v<offset_type, const OtherIndexType&> is true.
//
// Effects: Direct-non-list-initializes extents_ with e, strides_ from s, and offset_ with offset.

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "../ConvertibleToIntegral.h"
#include "test_macros.h"

template <class E, class S, cuda::std::enable_if_t<E::rank() != 0, int> = 0>
__host__ __device__ constexpr void test_construction(E e, S s, cuda::std::intptr_t input_offset = 0)
{
  using M            = cuda::layout_stride_relaxed::mapping<E>;
  using offset_type  = typename M::offset_type;
  offset_type offset = static_cast<offset_type>(input_offset);
  static_assert(noexcept(M{e, s}));
  static_assert(noexcept(M{e, s, offset}));
  M m(e, s, offset);

  // check correct extents are returned
  static_assert(noexcept(m.extents()));
  assert(m.extents() == e);

  // check offset
  assert(m.offset() == offset);

  // check strides
  auto strides_obj = m.strides();
  static_assert(noexcept(m.strides()));
  for (typename E::rank_type r = 0; r < E::rank(); r++)
  {
    assert(cuda::std::cmp_equal(m.stride(r), s[r]));
    assert(cuda::std::cmp_equal(strides_obj.stride(r), s[r]));
  }
}

template <class E, class S, cuda::std::enable_if_t<E::rank() == 0, int> = 0>
__host__ __device__ constexpr void test_construction(E e, S s, cuda::std::intptr_t input_offset = 0)
{
  using M            = cuda::layout_stride_relaxed::mapping<E>;
  using offset_type  = typename M::offset_type;
  offset_type offset = static_cast<offset_type>(input_offset);
  static_assert(noexcept(M{e, s}));
  static_assert(noexcept(M{e, s, offset}));
  M m(e, s, offset);

  // check correct extents are returned
  static_assert(noexcept(m.extents()));
  assert(m.extents() == e);

  // check offset
  assert(m.offset() == offset);

  // check strides
  static_assert(noexcept(m.strides()));
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  // Basic cases with zero offset
  {
    cuda::std::array<int, 0> s{};
    test_construction(cuda::std::extents<int>(), s);
  }
  {
    cuda::std::array<int, 1> s{1};
    test_construction(cuda::std::extents<unsigned, D>(7), s);
  }
  {
    cuda::std::array<int, 1> s{1};
    test_construction(cuda::std::extents<unsigned, D>(0), s);
  }
  {
    cuda::std::array<int, 1> s{2};
    test_construction(cuda::std::extents<unsigned, 7>(), s);
  }
  {
    cuda::std::array<int, 2> s{3, 30};
    test_construction(cuda::std::extents<unsigned, 7, 8>(), s);
  }
  {
    cuda::std::array<int, 4> s{20, 2, 200, 2000};
    test_construction(cuda::std::extents<int64_t, D, 8, D, D>(7, 9, 10), s);
  }

  // Cases with non-zero offset
  {
    cuda::std::array<int, 1> s{1};
    test_construction(cuda::std::extents<unsigned, D>(7), s, 5);
  }
  {
    cuda::std::array<int, 2> s{3, 30};
    test_construction(cuda::std::extents<unsigned, 7, 8>(), s, 10);
  }

  // Cases with negative strides
  {
    cuda::std::array<cuda::std::intptr_t, 1> s{-1};
    test_construction(cuda::std::extents<int, D>(7), s, 6);
  }
  {
    cuda::std::array<cuda::std::intptr_t, 2> s{-1, 7};
    test_construction(cuda::std::extents<int, 7, 8>(), s, 6);
  }
  {
    cuda::std::array<cuda::std::intptr_t, 2> s{8, -1};
    test_construction(cuda::std::extents<int, 7, 8>(), s, 7);
  }

  // Cases with zero strides (broadcasting)
  {
    cuda::std::array<cuda::std::intptr_t, 2> s{0, 1};
    test_construction(cuda::std::extents<int, 7, 8>(), s);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
