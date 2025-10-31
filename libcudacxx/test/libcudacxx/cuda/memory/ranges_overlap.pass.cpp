//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/iterator>
#include <cuda/memory>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include "test_iterators.h"

template <typename Pointer>
__host__ __device__ void test_overlaps(
  [[maybe_unused]] Pointer first_begin,
  [[maybe_unused]] Pointer first_end,
  [[maybe_unused]] Pointer second_begin,
  [[maybe_unused]] Pointer second_end)
{
  assert(cuda::ranges_overlap(first_begin, first_end, second_begin, second_end));
  assert(cuda::ranges_overlap(second_begin, second_end, first_begin, first_end));
}

template <typename Pointer>
__host__ __device__ void test_non_overlaps(
  [[maybe_unused]] Pointer first_begin,
  [[maybe_unused]] Pointer first_end,
  [[maybe_unused]] Pointer second_begin,
  [[maybe_unused]] Pointer second_end)
{
  assert(!cuda::ranges_overlap(first_begin, first_end, second_begin, second_end));
  assert(!cuda::ranges_overlap(second_begin, second_end, first_begin, first_end));
}

template <typename T>
__host__ __device__ void test_variants(T a_begin, T a_end, T b_begin, T b_end)
{
  static_assert(noexcept(cuda::ranges_overlap(
    cuda::std::declval<T>(), cuda::std::declval<T>(), cuda::std::declval<T>(), cuda::std::declval<T>())));
  using ret_type = decltype(cuda::ranges_overlap(
    cuda::std::declval<T>(), cuda::std::declval<T>(), cuda::std::declval<T>(), cuda::std::declval<T>()));
  static_assert(::cuda::std::is_same_v<bool, ret_type>);

  test_overlaps(a_begin, cuda::std::next(a_begin, 4), a_begin, cuda::std::next(a_begin, 4)); // same range
  test_overlaps(
    a_begin, cuda::std::next(a_begin, 4), cuda::std::next(a_begin, 1), cuda::std::next(a_begin, 3)); // within-range
  test_overlaps(cuda::std::next(a_begin, 3),
                cuda::std::next(a_begin, 5),
                cuda::std::next(a_begin, 1),
                cuda::std::next(a_begin, 5)); // left-side overlap
  test_overlaps(cuda::std::next(a_begin, 2),
                cuda::std::next(a_begin, 4),
                cuda::std::next(a_begin, 2),
                cuda::std::next(a_begin, 3)); // right-side overlap

  test_non_overlaps(a_begin, cuda::std::next(a_begin, 2), b_begin, cuda::std::next(b_begin, 2));
  test_non_overlaps(a_begin, cuda::std::next(a_begin, 2), cuda::std::next(a_begin, 2), cuda::std::next(a_begin, 4));
}

__host__ __device__ bool test()
{
  int buffer_a[8] = {};
  int buffer_b[8] = {};
  test_variants<int*>(buffer_a, buffer_a + 8, buffer_b, buffer_b + 8);
  test_variants<const int*>(buffer_a, buffer_a + 8, buffer_b, buffer_b + 8);

  using fwd_t = forward_iterator<int*>;
  test_variants<fwd_t>(fwd_t{buffer_a}, fwd_t{buffer_a + 8}, fwd_t{buffer_b}, fwd_t{buffer_b + 8});

  using bid_t = bidirectional_iterator<int*>;
  test_variants<bid_t>(bid_t{buffer_a}, bid_t{buffer_a + 8}, bid_t{buffer_b}, bid_t{buffer_b + 8});

  using rnd_t = random_access_iterator<int*>;
  test_variants<rnd_t>(rnd_t{buffer_a}, rnd_t{buffer_a + 8}, rnd_t{buffer_b}, rnd_t{buffer_b + 8});

  // Test empty ranges
  assert(!cuda::ranges_overlap(buffer_a, buffer_a, buffer_a, buffer_a)); // same empty range
  assert(!cuda::ranges_overlap(buffer_a, buffer_a, buffer_b, buffer_b)); // different empty ranges
  assert(!cuda::ranges_overlap(buffer_a, buffer_a, buffer_a, buffer_a + 4)); // empty vs non-empty
  assert(!cuda::ranges_overlap(buffer_a, buffer_a + 4, buffer_b, buffer_b)); // non-empty vs empty

  return true;
}

__host__ __device__ constexpr bool constexpr_test()
{
  constexpr int array[8] = {0, 1, 2, 3, 4};
  assert(cuda::ranges_overlap(array + 1, array + 3, array, array + 4));
  assert(!cuda::ranges_overlap(array + 5, array + 7, array, array + 5));
  return true;
}

int main(int, char**)
{
  assert(test());
  // clang < 20 has a defect with constexpr evaluation of pointer ranges
  // https://releases.llvm.org/20.1.0/tools/clang/docs/ReleaseNotes.html#resolutions-to-c-defect-reports
#if !_CCCL_COMPILER(CLANG, <, 20)
  static_assert(constexpr_test());
#endif // !_CCCL_COMPILER(CLANG, <, 20)
  return 0;
}
