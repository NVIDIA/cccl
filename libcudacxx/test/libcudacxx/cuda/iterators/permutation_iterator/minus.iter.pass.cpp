//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr iter_difference_t operator-(const permutation_iterator& x, const permutation_iterator& y);

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using indexIter = random_access_iterator<const int*>;
  int buffer[]    = {1, 2, 3, 4, 5, 6, 7, 8};

  { // Iterators with same base
    const int offset[] = {2, 4};
    cuda::permutation_iterator iter1(buffer, indexIter{offset});
    cuda::permutation_iterator iter2(buffer, indexIter{offset + 1});
    assert(iter1 - iter2 == -1);
    assert(iter2 - iter1 == 1);
    assert(iter1.index() == offset[0]);
    assert(iter2.index() == offset[1]);

    static_assert(cuda::std::is_same_v<decltype(iter1 - iter2), cuda::std::iter_difference_t<int*>>);
  }

  { // const iterators with same base
    const int offset[] = {2, 4};
    const cuda::permutation_iterator iter1(buffer, indexIter{offset});
    const cuda::permutation_iterator iter2(buffer, indexIter{offset + 1});
    assert(iter1 - iter2 == -1);
    assert(iter2 - iter1 == 1);
    assert(iter1.index() == offset[0]);
    assert(iter2.index() == offset[1]);

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
