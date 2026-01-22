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

// Test conversion operator from layout_stride_relaxed::mapping to layout_stride::mapping:

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::intptr_t;

template <class E>
__host__ __device__ constexpr void test_conversion(E ext, cuda::std::array<intptr_t, E::rank()> input_strides)
{
  using RelaxedMapping = cuda::layout_stride_relaxed::template mapping<E>;
  using StrideMapping  = cuda::std::layout_stride::template mapping<E>;
  using strides_type   = typename RelaxedMapping::strides_type;
  RelaxedMapping src(ext, strides_type(input_strides), 0); // zero offset
  static_assert(!cuda::std::is_convertible_v<RelaxedMapping, StrideMapping>,
                "conversion should be explicit, not implicit");
  static_assert(cuda::std::is_constructible_v<StrideMapping, RelaxedMapping>,
                "conversion should be possible via explicit cast");
  static_assert(noexcept(static_cast<StrideMapping>(src)), "conversion should be noexcept");

  StrideMapping dest = static_cast<StrideMapping>(src);
  assert(dest.extents() == src.extents());
  if constexpr (E::rank() > 0)
  {
    for (typename E::rank_type r = 0; r < E::rank(); r++)
    {
      assert(cuda::std::cmp_equal(dest.stride(r), src.stride(r)));
    }
  }
  assert(dest.required_span_size() == src.required_span_size());
}

__host__ __device__ constexpr bool test()
{
  constexpr size_t D = cuda::std::dynamic_extent;
  // Rank-0 case
  test_conversion(cuda::std::extents<int>(), cuda::std::array<intptr_t, 0>{});
  // Rank-1 cases
  test_conversion(cuda::std::extents<int, 5>(), cuda::std::array<intptr_t, 1>{1});
  test_conversion(cuda::std::extents<int, D>(5), cuda::std::array<intptr_t, 1>{1});
  test_conversion(cuda::std::extents<int, D>(5), cuda::std::array<intptr_t, 1>{2}); // non-unit stride
  test_conversion(cuda::std::extents<int, 4, 5>(), cuda::std::array<intptr_t, 2>{10, 2});
  test_conversion(cuda::std::extents<int, D, D>(4, 5), cuda::std::array<intptr_t, 2>{10, 2});
  test_conversion(cuda::std::extents<int, 2, 3, 4>(), cuda::std::array<intptr_t, 3>{12, 4, 1});
  test_conversion(cuda::std::extents<int, D, D, D>(2, 3, 4), cuda::std::array<intptr_t, 3>{12, 4, 1});
  // Cases with zero extents (valid conversions)
  test_conversion(cuda::std::extents<int, 0>(), cuda::std::array<intptr_t, 1>{1});
  test_conversion(cuda::std::extents<int, D>(0), cuda::std::array<intptr_t, 1>{1});
  test_conversion(cuda::std::extents<int, 0, 5>(), cuda::std::array<intptr_t, 2>{5, 1});
  test_conversion(cuda::std::extents<int, 5, 0>(), cuda::std::array<intptr_t, 2>{1, 5});
  // Different index types
  test_conversion(cuda::std::extents<size_t, 4, 5>(), cuda::std::array<intptr_t, 2>{5, 1});
  test_conversion(cuda::std::extents<int64_t, D, D>(4, 5), cuda::std::array<intptr_t, 2>{5, 1});
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
