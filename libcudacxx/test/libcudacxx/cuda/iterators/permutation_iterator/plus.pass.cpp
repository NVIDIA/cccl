//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr permutation_iterator operator+(iter_difference_t<I> n) const;
// friend constexpr permutation_iterator operator+(iter_difference_t<I> n, const permutation_iterator& x);
// constexpr permutation_iterator& operator+=(iter_difference_t<I> n);

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};

  { // operator+(iter_difference_t<I> n)
    {
      using indexIter            = const int*;
      using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
      const int offset[]         = {2, 3, 4, 5};
      const int diff             = 3;
      permutation_iterator iter(buffer, offset);
      assert(iter + diff == permutation_iterator(buffer, offset + diff));
      assert(iter + 0 == permutation_iterator(buffer, offset));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), permutation_iterator>);
      static_assert(noexcept(iter + diff));
    }

    { // const iter
      using indexIter            = const int*;
      using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
      const int offset[]         = {2, 3, 4, 5};
      const int diff             = 3;
      const permutation_iterator iter(buffer, offset);
      assert(iter + diff == permutation_iterator(buffer, offset + diff));
      assert(iter + 0 == permutation_iterator(buffer, offset));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), permutation_iterator>);
      static_assert(noexcept(iter + diff));
    }

    { // custom index iterator
      using indexIter            = random_access_iterator<const int*>;
      using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
      const int offset[]         = {2, 3, 4, 5};
      const int diff             = 3;
      permutation_iterator iter(buffer, indexIter{offset});
      assert(iter + diff == permutation_iterator(buffer, indexIter{offset + diff}));
      assert(iter + 0 == permutation_iterator(buffer, indexIter{offset}));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), permutation_iterator>);
      static_assert(!noexcept(iter + diff));
    }

    { // const iter and custom index iterator
      using indexIter            = random_access_iterator<const int*>;
      using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
      const int offset[]         = {2, 3, 4, 5};
      const int diff             = 3;
      const permutation_iterator iter(buffer, indexIter{offset});
      assert(iter + diff == permutation_iterator(buffer, indexIter{offset + diff}));
      assert(iter + 0 == permutation_iterator(buffer, indexIter{offset}));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), permutation_iterator>);
      static_assert(!noexcept(iter + diff));
    }
  }

  { // operator+(iter_difference_t<I> n, const permutation_iterator& x)
    {
      using indexIter            = const int*;
      using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
      const int offset[]         = {2, 3, 4, 5};
      const int diff             = 3;
      permutation_iterator iter(buffer, offset);
      assert(diff + iter == permutation_iterator(buffer, offset + diff));
      assert(0 + iter == permutation_iterator(buffer, offset));

      static_assert(cuda::std::is_same_v<decltype(2 + iter), permutation_iterator>);
      static_assert(noexcept(diff + iter));
    }

    { // const iter
      using indexIter            = const int*;
      using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
      const int offset[]         = {2, 3, 4, 5};
      const int diff             = 3;
      const permutation_iterator iter(buffer, offset);
      assert(diff + iter == permutation_iterator(buffer, offset + diff));
      assert(0 + iter == permutation_iterator(buffer, offset));

      static_assert(cuda::std::is_same_v<decltype(2 + iter), permutation_iterator>);
      static_assert(noexcept(diff + iter));
    }

    { // custom index iterator
      using indexIter            = random_access_iterator<const int*>;
      using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
      const int offset[]         = {2, 3, 4, 5};
      const int diff             = 3;
      permutation_iterator iter(buffer, indexIter{offset});
      assert(diff + iter == permutation_iterator(buffer, indexIter{offset + diff}));
      assert(0 + iter == permutation_iterator(buffer, indexIter{offset}));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), permutation_iterator>);
      static_assert(!noexcept(diff + iter));
    }

    { // const iter and custom index iterator
      using indexIter            = random_access_iterator<const int*>;
      using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
      const int offset[]         = {2, 3, 4, 5};
      const int diff             = 3;
      const permutation_iterator iter(buffer, indexIter{offset});
      assert(diff + iter == permutation_iterator(buffer, indexIter{offset + diff}));
      assert(0 + iter == permutation_iterator(buffer, indexIter{offset}));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), permutation_iterator>);
      static_assert(!noexcept(diff + iter));
    }
  }

  { // operator+=(iter_difference_t<I> n)
    {
      using indexIter            = const int*;
      using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
      const int offset[]         = {2, 3, 4, 5};
      const int diff             = 3;
      permutation_iterator iter(buffer, offset);
      assert((iter += 0) == permutation_iterator(buffer, offset));
      assert((iter += diff) == permutation_iterator(buffer, offset + diff));

      static_assert(cuda::std::is_same_v<decltype(iter += 2), permutation_iterator&>);
      static_assert(noexcept(iter += diff));
    }

    { // custom index iterator
      using indexIter            = random_access_iterator<const int*>;
      using permutation_iterator = cuda::permutation_iterator<int*, indexIter>;
      const int offset[]         = {2, 3, 4, 5};
      const int diff             = 3;
      permutation_iterator iter(buffer, indexIter{offset});
      assert((iter += 0) == permutation_iterator(buffer, indexIter{offset}));
      assert((iter += diff) == permutation_iterator(buffer, indexIter{offset + diff}));

      static_assert(cuda::std::is_same_v<decltype(iter += 2), permutation_iterator&>);
      static_assert(!noexcept(iter += diff));
    }
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
