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

// constexpr index_type required_span_size() const noexcept;
//
// For layout_stride_relaxed:
//   - For rank-0: offset_ + 1
//   - Otherwise: offset_ + 1 + sum of (max_index[r] * stride[r]) for all r,
//     where max_index[r] = (extent[r] - 1) if stride[r] >= 0, else 0
//
// This differs from layout_stride because:
//   - We account for the offset
//   - We handle negative strides by using 0 for the corresponding dimension's contribution
//     when the stride is negative (those dimensions iterate backwards from the offset)

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"

using cuda::std::intptr_t;

template <class E>
__host__ __device__ constexpr void
test_required_span_size(E e, cuda::std::array<intptr_t, E::rank()> input_strides, intptr_t offset, typename E::index_type expected_size)
{
  using M            = cuda::layout_stride_relaxed::mapping<E>;
  using offset_type  = typename M::offset_type;
  using index_type   = typename M::index_type;
  using stride_array = cuda::std::array<offset_type, E::rank()>;
  stride_array strides{};
  if constexpr (E::rank() > 0)
  {
    for (typename E::rank_type r = 0; r < E::rank(); r++)
    {
      strides[r] = static_cast<offset_type>(input_strides[r]);
    }
  }
  const M m(e, strides, static_cast<offset_type>(offset));

  static_assert(noexcept(m.required_span_size()));
  assert(cuda::std::cmp_equal(m.required_span_size(), expected_size));
}

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  // Rank-0 cases: required_span_size = offset + 1
  test_required_span_size(cuda::std::extents<int>(), cuda::std::array<intptr_t, 0>{}, 0, 1);
  test_required_span_size(cuda::std::extents<int>(), cuda::std::array<intptr_t, 0>{}, 5, 6);
  test_required_span_size(cuda::std::extents<int>(), cuda::std::array<intptr_t, 0>{}, 10, 11);

  // Rank-1 cases with zero extent
  // When extent is 0, there are no valid indices, so required_span_size = 0
  test_required_span_size(cuda::std::extents<unsigned, D>(0), cuda::std::array<intptr_t, 1>{5}, 0, 0);
  test_required_span_size(cuda::std::extents<unsigned, D>(0), cuda::std::array<intptr_t, 1>{5}, 10, 0);

  // Rank-1 cases with positive strides
  // extent=1: required_span_size = offset + 1 + (1-1)*stride = offset + 1
  test_required_span_size(cuda::std::extents<unsigned, D>(1), cuda::std::array<intptr_t, 1>{5}, 0, 1);
  // extent=7, stride=5: required_span_size = offset + 1 + (7-1)*5 = 0 + 1 + 30 = 31
  test_required_span_size(cuda::std::extents<unsigned, D>(7), cuda::std::array<intptr_t, 1>{5}, 0, 31);
  // extent=7, stride=5, offset=10: required_span_size = 10 + 1 + 30 = 41
  test_required_span_size(cuda::std::extents<unsigned, D>(7), cuda::std::array<intptr_t, 1>{5}, 10, 41);
  test_required_span_size(cuda::std::extents<unsigned, 7>(), cuda::std::array<intptr_t, 1>{5}, 0, 31);

  // Rank-1 cases with negative strides
  // For negative stride, max contribution is at index 0, so we add 0
  // extent=7, stride=-1, offset=6: max at index 0 gives offset, so required_span_size = offset + 1 = 7
  test_required_span_size(cuda::std::extents<int, D>(7), cuda::std::array<intptr_t, 1>{-1}, 6, 7);
  // extent=4, stride=-1, offset=3: required_span_size = 3 + 1 = 4
  test_required_span_size(cuda::std::extents<int, 4>(), cuda::std::array<intptr_t, 1>{-1}, 3, 4);

  // Rank-2 cases with positive strides
  // extents(7,8), strides(20,2), offset=0: required = 0 + 1 + 6*20 + 7*2 = 1 + 120 + 14 = 135
  test_required_span_size(cuda::std::extents<unsigned, 7, 8>(), cuda::std::array<intptr_t, 2>{20, 2}, 0, 135);
  // Same with offset=10: required = 10 + 1 + 120 + 14 = 145
  test_required_span_size(cuda::std::extents<unsigned, 7, 8>(), cuda::std::array<intptr_t, 2>{20, 2}, 10, 145);

  // Rank-2 cases with mixed strides
  // extents(7,8), strides(-1, 7), offset=6:
  // dim0 has negative stride, contributes 0; dim1 has positive stride, contributes 7*7=49
  // required = 6 + 1 + 0 + 49 = 56
  test_required_span_size(cuda::std::extents<int, 7, 8>(), cuda::std::array<intptr_t, 2>{-1, 7}, 6, 56);

  // Rank-2 with zero stride (broadcasting)
  // extents(7,8), strides(0, 1), offset=0:
  // dim0 with stride 0 contributes 0; dim1 contributes 7*1=7
  // required = 0 + 1 + 0 + 7 = 8
  test_required_span_size(cuda::std::extents<int, 7, 8>(), cuda::std::array<intptr_t, 2>{0, 1}, 0, 8);

  // Higher rank cases
  // extents(7,8,9,10), strides(1,7,56,504), offset=0
  // required = 0 + 1 + 6*1 + 7*7 + 8*56 + 9*504 = 1 + 6 + 49 + 448 + 4536 = 5040
  test_required_span_size(
    cuda::std::extents<int64_t, D, 8, D, D>(7, 9, 10), cuda::std::array<intptr_t, 4>{1, 7, 7 * 8, 7 * 8 * 9}, 0, 5040);

  // With one extent = 1
  // extents(1,8,9,10), the dimension with extent 1 contributes 0
  test_required_span_size(
    cuda::std::extents<int64_t, 1, 8, D, D>(9, 10), cuda::std::array<intptr_t, 4>{1, 7, 7 * 8, 7 * 8 * 9}, 0, 5034);

  // With one extent = 0 (empty mdspan)
  // When any extent is 0, there are no valid indices, so required_span_size = 0
  test_required_span_size(
    cuda::std::extents<int64_t, 1, 0, D, D>(9, 10), cuda::std::array<intptr_t, 4>{1, 7, 7 * 8, 7 * 8 * 9}, 0, 0);

  // ============================================================================
  // Additional edge cases for zero extents
  // ============================================================================

  // All extents zero (multiple dimensions)
  // When all extents are 0, there are no valid indices, so required_span_size = 0
  test_required_span_size(cuda::std::extents<int, D, D>(0, 0), cuda::std::array<intptr_t, 2>{1, 1}, 0, 0);
  test_required_span_size(cuda::std::extents<int, 0, 0>(), cuda::std::array<intptr_t, 2>{8, 1}, 0, 0);
  test_required_span_size(cuda::std::extents<int, D, D, D>(0, 0, 0), cuda::std::array<intptr_t, 3>{1, 1, 1}, 0, 0);
  test_required_span_size(cuda::std::extents<int64_t, 0, 0, 0, 0>(), cuda::std::array<intptr_t, 4>{1, 2, 3, 4}, 0, 0);

  // All extents zero with non-zero offset
  // Even with offset, zero extents mean no valid indices, so required_span_size = 0
  test_required_span_size(cuda::std::extents<int, D, D>(0, 0), cuda::std::array<intptr_t, 2>{1, 1}, 100, 0);
  test_required_span_size(cuda::std::extents<int, 0, 0>(), cuda::std::array<intptr_t, 2>{8, 1}, 50, 0);

  // Zero extent with negative strides
  // Zero extent still results in required_span_size = 0 regardless of stride sign
  test_required_span_size(cuda::std::extents<int, D>(0), cuda::std::array<intptr_t, 1>{-1}, 0, 0);
  test_required_span_size(cuda::std::extents<int, D>(0), cuda::std::array<intptr_t, 1>{-1}, 10, 0);
  test_required_span_size(cuda::std::extents<int, 0>(), cuda::std::array<intptr_t, 1>{-5}, 0, 0);

  // Mix of zero and non-zero extents with negative strides
  // Only the zero extent matters - it makes the whole thing empty
  test_required_span_size(cuda::std::extents<int, D, D>(0, 5), cuda::std::array<intptr_t, 2>{-1, 1}, 0, 0);
  test_required_span_size(cuda::std::extents<int, D, D>(5, 0), cuda::std::array<intptr_t, 2>{-1, 1}, 4, 0);
  test_required_span_size(cuda::std::extents<int, 0, 5>(), cuda::std::array<intptr_t, 2>{-5, 1}, 0, 0);
  test_required_span_size(cuda::std::extents<int, 5, 0>(), cuda::std::array<intptr_t, 2>{1, -1}, 0, 0);

  // Mix of zero and non-zero extents with all negative strides
  test_required_span_size(cuda::std::extents<int, D, D>(0, 5), cuda::std::array<intptr_t, 2>{-1, -1}, 4, 0);
  test_required_span_size(cuda::std::extents<int, D, D>(5, 0), cuda::std::array<intptr_t, 2>{-1, -1}, 4, 0);

  // Zero extent in the middle of non-zero extents
  test_required_span_size(cuda::std::extents<int, D, D, D>(3, 0, 4), cuda::std::array<intptr_t, 3>{12, 4, 1}, 0, 0);
  test_required_span_size(cuda::std::extents<int, 3, 0, 4>(), cuda::std::array<intptr_t, 3>{12, 4, 1}, 0, 0);

  // Zero extent with zero stride (broadcasting an empty dimension)
  test_required_span_size(cuda::std::extents<int, D, D>(0, 5), cuda::std::array<intptr_t, 2>{0, 1}, 0, 0);
  test_required_span_size(cuda::std::extents<int, D, D>(5, 0), cuda::std::array<intptr_t, 2>{1, 0}, 0, 0);

  // All dimensions with zero strides (full broadcasting) - non-zero extents
  // With zero strides, all indices map to the same location
  // extents(3,4), strides(0,0), offset=5: all indices -> 5, required = 5 + 1 = 6
  test_required_span_size(cuda::std::extents<int, 3, 4>(), cuda::std::array<intptr_t, 2>{0, 0}, 0, 1);
  test_required_span_size(cuda::std::extents<int, 3, 4>(), cuda::std::array<intptr_t, 2>{0, 0}, 5, 6);
  test_required_span_size(cuda::std::extents<int, D, D>(3, 4), cuda::std::array<intptr_t, 2>{0, 0}, 10, 11);

  // Single zero extent (static)
  test_required_span_size(cuda::std::extents<int, 0>(), cuda::std::array<intptr_t, 1>{1}, 0, 0);
  test_required_span_size(cuda::std::extents<int, 0>(), cuda::std::array<intptr_t, 1>{1}, 100, 0);

  // Higher rank with multiple zero extents
  test_required_span_size(
    cuda::std::extents<int64_t, D, 0, D, 0>(5, 7), cuda::std::array<intptr_t, 4>{1, 2, 3, 4}, 0, 0);
  test_required_span_size(
    cuda::std::extents<int64_t, 0, D, 0, D>(5, 7), cuda::std::array<intptr_t, 4>{1, 2, 3, 4}, 50, 0);

  // Zero extent with mixed positive, negative, and zero strides
  test_required_span_size(cuda::std::extents<int, D, D, D>(0, 5, 3), cuda::std::array<intptr_t, 3>{-1, 0, 1}, 0, 0);
  test_required_span_size(cuda::std::extents<int, D, D, D>(5, 0, 3), cuda::std::array<intptr_t, 3>{-1, 0, 1}, 4, 0);
  test_required_span_size(cuda::std::extents<int, D, D, D>(5, 3, 0), cuda::std::array<intptr_t, 3>{-1, 0, 1}, 4, 0);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
