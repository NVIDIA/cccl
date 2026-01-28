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

// constexpr mapping(const extents_type& e, const strides_type& s, offset_type offset = 0) noexcept;
//
// Effects: Direct-non-list-initializes extents_ with e, strides_ with s, and offset_ with offset.

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "../ConvertibleToIntegral.h"
#include "test_macros.h"

using cuda::std::intptr_t;

template <class E, cuda::std::enable_if_t<E::rank() != 0, int> = 0>
__host__ __device__ constexpr void
test_construction(E e, cuda::std::array<intptr_t, E::rank()> s, intptr_t input_offset = 0)
{
  using M            = cuda::layout_stride_relaxed::mapping<E>;
  using strides_type = typename M::strides_type;
  using offset_type  = typename M::offset_type;
  strides_type strides(s);
  offset_type offset = static_cast<offset_type>(input_offset);
  static_assert(noexcept(M{e, strides}));
  static_assert(noexcept(M{e, strides, offset}));
  M m(e, strides, offset);

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

template <class E, cuda::std::enable_if_t<E::rank() == 0, int> = 0>
__host__ __device__ constexpr void
test_construction(E e, cuda::std::array<intptr_t, E::rank()> s, intptr_t input_offset = 0)
{
  using M            = cuda::layout_stride_relaxed::mapping<E>;
  using strides_type = typename M::strides_type;
  using offset_type  = typename M::offset_type;
  strides_type strides(s);
  offset_type offset = static_cast<offset_type>(input_offset);
  static_assert(noexcept(M{e, strides}));
  static_assert(noexcept(M{e, strides, offset}));
  M m(e, strides, offset);

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
  test_construction(cuda::std::extents<int>(), cuda::std::array<intptr_t, 0>{});
  test_construction(cuda::std::extents<unsigned, D>(7), cuda::std::array<intptr_t, 1>{1});
  test_construction(cuda::std::extents<unsigned, D>(0), cuda::std::array<intptr_t, 1>{1});
  test_construction(cuda::std::extents<unsigned, 7>(), cuda::std::array<intptr_t, 1>{2});
  test_construction(cuda::std::extents<unsigned, 7, 8>(), cuda::std::array<intptr_t, 2>{3, 30});
  test_construction(cuda::std::extents<int64_t, D, 8, D, D>(7, 9, 10), cuda::std::array<intptr_t, 4>{20, 2, 200, 2000});

  // Cases with non-zero offset
  test_construction(cuda::std::extents<unsigned, D>(7), cuda::std::array<intptr_t, 1>{1}, 5);
  test_construction(cuda::std::extents<unsigned, 7, 8>(), cuda::std::array<intptr_t, 2>{3, 30}, 10);

  // Cases with negative strides
  test_construction(cuda::std::extents<int, D>(7), cuda::std::array<intptr_t, 1>{-1}, 6);
  test_construction(cuda::std::extents<int, 7, 8>(), cuda::std::array<intptr_t, 2>{-1, 7}, 6);
  test_construction(cuda::std::extents<int, 7, 8>(), cuda::std::array<intptr_t, 2>{8, -1}, 7);

  // Cases with zero strides (broadcasting)
  test_construction(cuda::std::extents<int, 7, 8>(), cuda::std::array<intptr_t, 2>{0, 1});

  // ============================================================================
  // Edge cases with zero extents
  // ============================================================================

  // Single zero extent (static and dynamic)
  test_construction(cuda::std::extents<int, 0>(), cuda::std::array<intptr_t, 1>{1});
  test_construction(cuda::std::extents<int, D>(0), cuda::std::array<intptr_t, 1>{1});
  test_construction(cuda::std::extents<int, 0>(), cuda::std::array<intptr_t, 1>{1}, 10);

  // All extents zero (multiple dimensions)
  test_construction(cuda::std::extents<int, 0, 0>(), cuda::std::array<intptr_t, 2>{1, 1});
  test_construction(cuda::std::extents<int, D, D>(0, 0), cuda::std::array<intptr_t, 2>{8, 1});
  test_construction(cuda::std::extents<int, 0, 0, 0>(), cuda::std::array<intptr_t, 3>{1, 1, 1});

  // All extents zero with non-zero offset
  test_construction(cuda::std::extents<int, 0, 0>(), cuda::std::array<intptr_t, 2>{1, 1}, 100);
  test_construction(cuda::std::extents<int, D, D>(0, 0), cuda::std::array<intptr_t, 2>{8, 1}, 50);

  // Zero extent with negative strides
  test_construction(cuda::std::extents<int, 0>(), cuda::std::array<intptr_t, 1>{-1});
  test_construction(cuda::std::extents<int, D>(0), cuda::std::array<intptr_t, 1>{-1}, 10);
  test_construction(cuda::std::extents<int, 0>(), cuda::std::array<intptr_t, 1>{-5});

  // Mix of zero and non-zero extents
  test_construction(cuda::std::extents<int, 0, 5>(), cuda::std::array<intptr_t, 2>{5, 1});
  test_construction(cuda::std::extents<int, 5, 0>(), cuda::std::array<intptr_t, 2>{1, 5});
  test_construction(cuda::std::extents<int, D, D>(0, 5), cuda::std::array<intptr_t, 2>{5, 1});
  test_construction(cuda::std::extents<int, D, D>(5, 0), cuda::std::array<intptr_t, 2>{1, 5});

  // Mix of zero and non-zero extents with negative strides
  test_construction(cuda::std::extents<int, 0, 5>(), cuda::std::array<intptr_t, 2>{-1, 1});
  test_construction(cuda::std::extents<int, D, D>(5, 0), cuda::std::array<intptr_t, 2>{-1, 1}, 4);

  // Zero extent in the middle of non-zero extents
  test_construction(cuda::std::extents<int, 3, 0, 4>(), cuda::std::array<intptr_t, 3>{12, 4, 1});
  test_construction(cuda::std::extents<int, D, D, D>(3, 0, 4), cuda::std::array<intptr_t, 3>{12, 4, 1});

  // Zero extent with zero stride (broadcasting an empty dimension)
  test_construction(cuda::std::extents<int, 0, 5>(), cuda::std::array<intptr_t, 2>{0, 1});
  test_construction(cuda::std::extents<int, 5, 0>(), cuda::std::array<intptr_t, 2>{1, 0});

  // Higher rank with multiple zero extents
  test_construction(cuda::std::extents<int64_t, D, 0, D, 0>(5, 7), cuda::std::array<intptr_t, 4>{1, 2, 3, 4});
  test_construction(cuda::std::extents<int64_t, 0, D, 0, D>(5, 7), cuda::std::array<intptr_t, 4>{1, 2, 3, 4}, 50);

  // Zero extent with mixed positive, negative, and zero strides
  test_construction(cuda::std::extents<int, 0, 5, 3>(), cuda::std::array<intptr_t, 3>{-1, 0, 1});
  test_construction(cuda::std::extents<int, D, D, D>(5, 0, 3), cuda::std::array<intptr_t, 3>{-1, 0, 1}, 4);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
