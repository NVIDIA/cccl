//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr permutation_iterator operator-(iter_difference_t<I> n) const;

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};

  { // operator-(iterator, iter_difference)
    using indexIter            = const int*;
    using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
    const int offset[]         = {4, 3, 2, 5};
    const int diff             = 2;
    permutation_iterator iter(buffer, offset + 3);
    assert(iter - diff == permutation_iterator(buffer, offset + 1));
    assert(iter - 0 == permutation_iterator(buffer, offset + 3));
    assert(iter.index() == offset[3]);

    static_assert(cuda::std::is_same_v<decltype(iter - diff), permutation_iterator>);
    static_assert(noexcept(iter - diff));
  }

  { // operator-(const iterator, iter_difference)
    using indexIter            = const int*;
    using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
    const int offset[]         = {4, 3, 2, 5};
    const int diff             = 2;
    const permutation_iterator iter(buffer, offset + 3);
    assert(iter - diff == permutation_iterator(buffer, offset + 1));
    assert(iter - 0 == permutation_iterator(buffer, offset + 3));
    assert(iter.index() == offset[3]);

    static_assert(cuda::std::is_same_v<decltype(iter - diff), permutation_iterator>);
    static_assert(noexcept(iter - diff));
  }

  { // operator-(iterator, iter_difference), with custom index iterator
    using indexIter            = random_access_iterator<const int*>;
    using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
    const int offset[]         = {4, 3, 2, 5};
    const int diff             = 2;
    permutation_iterator iter(buffer, indexIter{offset + 3});
    assert(iter - diff == permutation_iterator(buffer, indexIter{offset + 1}));
    assert(iter - 0 == permutation_iterator(buffer, indexIter{offset + 3}));
    assert(iter.index() == offset[3]);

    static_assert(cuda::std::is_same_v<decltype(iter - diff), permutation_iterator>);

    // The test iterators are not noexcept
    static_assert(!noexcept(iter - diff));
  }

  { // operator-(const iterator, iter_difference), with custom index iterator
    using indexIter            = random_access_iterator<const int*>;
    using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
    const int offset[]         = {4, 3, 2, 5};
    const int diff             = 2;
    const permutation_iterator iter(buffer, indexIter{offset + 3});
    assert(iter - diff == permutation_iterator(buffer, indexIter{offset + 1}));
    assert(iter - 0 == permutation_iterator(buffer, indexIter{offset + 3}));
    assert(iter.index() == offset[3]);

    static_assert(cuda::std::is_same_v<decltype(iter - diff), permutation_iterator>);

    // The test iterators are not noexcept
    static_assert(!noexcept(iter - diff));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
