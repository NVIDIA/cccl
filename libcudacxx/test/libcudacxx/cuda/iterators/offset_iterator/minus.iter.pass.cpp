//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr iter_difference_t operator-(const offset_iterator& x, const offset_iterator& y);

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    using offset_iterator = cuda::offset_iterator<int*>;
    const int offset1     = 4;
    const int offset2     = 2;
    offset_iterator iter1(buffer, offset1);
    offset_iterator iter2(buffer, offset2);
    assert(iter1 - iter2 == offset2 - offset1);
    assert(iter2 - iter1 == offset1 - offset2);
    assert(iter1.offset() == offset1);
    assert(iter2.offset() == offset2);

    static_assert(cuda::std::is_same_v<decltype(iter1 - iter2), cuda::std::iter_difference_t<int*>>);
  }

  {
    using offset_iterator = cuda::offset_iterator<int*>;
    const int offset1     = 4;
    const int offset2     = 2;
    const offset_iterator iter1(buffer, offset1);
    const offset_iterator iter2(buffer, offset2);
    assert(iter1 - iter2 == offset2 - offset1);
    assert(iter2 - iter1 == offset1 - offset2);
    assert(iter1.offset() == offset1);
    assert(iter2.offset() == offset2);

    static_assert(cuda::std::is_same_v<decltype(iter1 - iter2), cuda::std::iter_difference_t<int*>>);
  }

  {
    using offset_iterator = cuda::offset_iterator<int*, random_access_iterator<const int*>>;
    const int offset1[]   = {4};
    const int offset2[]   = {2};
    offset_iterator iter1(buffer, random_access_iterator<const int*>{offset1});
    offset_iterator iter2(buffer, random_access_iterator<const int*>{offset2});
    assert(iter1 - iter2 == -2);
    assert(iter2 - iter1 == 2);
    assert(iter1.offset() == 4);
    assert(iter2.offset() == 2);

    static_assert(cuda::std::is_same_v<decltype(iter1 - iter2), cuda::std::iter_difference_t<int*>>);
  }

  {
    using offset_iterator = cuda::offset_iterator<int*, random_access_iterator<const int*>>;
    const int offset1[]   = {4};
    const int offset2[]   = {2};
    const offset_iterator iter1(buffer, random_access_iterator<const int*>{offset1});
    const offset_iterator iter2(buffer, random_access_iterator<const int*>{offset2});
    assert(iter1 - iter2 == -2);
    assert(iter2 - iter1 == 2);
    assert(iter1.offset() == 4);
    assert(iter2.offset() == 2);

    static_assert(cuda::std::is_same_v<decltype(iter1 - iter2), cuda::std::iter_difference_t<int*>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
