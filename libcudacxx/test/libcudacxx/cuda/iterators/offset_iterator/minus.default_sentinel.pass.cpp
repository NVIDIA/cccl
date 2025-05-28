//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr iter_difference_t<I> operator-(const offset_iterator& x, default_sentinel_t);
// friend constexpr iter_difference_t<I> operator-(default_sentinel_t, const offset_iterator& y);

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    const int offset = 2;
    cuda::offset_iterator iter(buffer, offset);
    assert(iter - cuda::std::default_sentinel == -offset);
    assert(cuda::std::default_sentinel - iter == offset);
    assert(iter.offset() == offset);

    static_assert(
      cuda::std::is_same_v<decltype(iter - cuda::std::default_sentinel), cuda::std::iter_difference_t<int*>>);
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::default_sentinel - iter), cuda::std::iter_difference_t<int*>>);
  }

  {
    const int offset = 2;
    const cuda::offset_iterator iter(buffer, offset);
    assert(iter - cuda::std::default_sentinel == -offset);
    assert(cuda::std::default_sentinel - iter == offset);
    assert(iter.offset() == offset);

    static_assert(
      cuda::std::is_same_v<decltype(iter - cuda::std::default_sentinel), cuda::std::iter_difference_t<int*>>);
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::default_sentinel - iter), cuda::std::iter_difference_t<int*>>);
  }

  {
    const int offset[] = {2};
    cuda::offset_iterator iter(buffer, random_access_iterator<const int*>(offset));
    assert(iter - cuda::std::default_sentinel == -*offset);
    assert(cuda::std::default_sentinel - iter == *offset);
    assert(iter.offset() == *offset);

    static_assert(
      cuda::std::is_same_v<decltype(iter - cuda::std::default_sentinel), cuda::std::iter_difference_t<int*>>);
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::default_sentinel - iter), cuda::std::iter_difference_t<int*>>);
  }

  {
    const int offset[] = {2};
    const cuda::offset_iterator iter(buffer, random_access_iterator<const int*>(offset));
    assert(iter - cuda::std::default_sentinel == -*offset);
    assert(cuda::std::default_sentinel - iter == *offset);
    assert(iter.offset() == *offset);

    static_assert(
      cuda::std::is_same_v<decltype(iter - cuda::std::default_sentinel), cuda::std::iter_difference_t<int*>>);
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::default_sentinel - iter), cuda::std::iter_difference_t<int*>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
