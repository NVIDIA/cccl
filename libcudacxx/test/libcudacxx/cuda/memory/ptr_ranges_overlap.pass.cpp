//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

template <typename Pointer>
__host__ __device__ void test_overlaps(
  [[maybe_unused]] Pointer first_begin,
  [[maybe_unused]] Pointer first_end,
  [[maybe_unused]] Pointer second_begin,
  [[maybe_unused]] Pointer second_end)
{
  assert(cuda::ptr_ranges_overlap(first_begin, first_end, second_begin, second_end));
  assert(cuda::ptr_ranges_overlap(second_begin, second_end, first_begin, first_end));
}

template <typename Pointer>
__host__ __device__ void test_non_overlaps(
  [[maybe_unused]] Pointer first_begin,
  [[maybe_unused]] Pointer first_end,
  [[maybe_unused]] Pointer second_begin,
  [[maybe_unused]] Pointer second_end)
{
  assert(!cuda::ptr_ranges_overlap(first_begin, first_end, second_begin, second_end));
  assert(!cuda::ptr_ranges_overlap(second_begin, second_end, first_begin, first_end));
}

template <typename T>
__host__ __device__ void test_variants()
{
  T buffer_a[8]    = {};
  T buffer_b[8]    = {};
  const T* a_begin = buffer_a;
  const T* b_begin = buffer_b;

  test_overlaps(a_begin, a_begin + 4, a_begin, a_begin + 4); // same range
  test_overlaps(a_begin, a_begin + 4, a_begin + 1, a_begin + 3); // within-range
  test_overlaps(a_begin + 3, a_begin + 5, a_begin + 1, a_begin + 5); // left-side overlap
  test_overlaps(a_begin + 2, a_begin + 4, a_begin + 2, a_begin + 3); // right-side overlap

  test_non_overlaps(a_begin, a_begin + 2, b_begin, b_begin + 2);
  test_non_overlaps(a_begin, a_begin + 2, a_begin + 2, a_begin + 4);
}

__host__ __device__ bool test()
{
  static_assert(noexcept(cuda::ptr_ranges_overlap(nullptr, nullptr, nullptr, nullptr)));
  using ret_type = decltype(cuda::ptr_ranges_overlap(nullptr, nullptr, nullptr, nullptr));
  static_assert(::cuda::std::is_same_v<bool, ret_type>);

  test_variants<int>();
  test_variants<const int>();
  return true;
}

__host__ __device__ constexpr bool constexpr_test()
{
  constexpr int array[5] = {0, 1, 2, 3, 4};
  assert(cuda::ptr_ranges_overlap(array + 1, array + 5, array, array + 5));
  assert(!cuda::ptr_ranges_overlap(array + 5, array + 5, array, array + 5));
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(constexpr_test());
  return 0;
}
