//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr discard_iterator operator-(iter_difference_t<I> n) const;
// friend constexpr iter_difference_t operator-(const discard_iterator& x, const discard_iterator& y)
// constexpr discard_iterator& operator-=(iter_difference_t<I> n);
// friend constexpr iter_difference_t<I> operator-(const discard_iterator& x, default_sentinel_t);
// friend constexpr iter_difference_t<I> operator-(default_sentinel_t, const discard_iterator& y);

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
_CCCL_CONCEPT MinusEnabled = _CCCL_REQUIRES_EXPR((Iter), Iter& iter)((iter - 1));

__host__ __device__ constexpr bool test()
{
  { // operator-(iter_difference_t<I> n)
    {
      const int counter = 3;
      const int diff    = 2;
      cuda::discard_iterator iter(counter);
      assert(iter - diff == cuda::discard_iterator(counter - diff));
      assert(iter - 0 == cuda::discard_iterator(counter));

      static_assert(cuda::std::is_same_v<decltype(iter - 2), cuda::discard_iterator>);
    }

    {
      const int counter = 3;
      const int diff    = 2;
      const cuda::discard_iterator iter(counter);
      assert(iter - diff == cuda::discard_iterator(counter - diff));
      assert(iter - 0 == cuda::discard_iterator(counter));

      static_assert(cuda::std::is_same_v<decltype(iter - 2), cuda::discard_iterator>);
    }
  }

  { // operator-(const discard_iterator& x, const discard_iterator& y)
    {
      const int counter1 = 4;
      const int counter2 = 2;
      cuda::discard_iterator iter1(counter1);
      cuda::discard_iterator iter2(counter2);
      assert(iter1 - iter2 == counter2 - counter1);
      assert(iter2 - iter1 == counter1 - counter2);

      static_assert(cuda::std::is_same_v<decltype(iter1 - iter2), cuda::std::ptrdiff_t>);
    }

    {
      const int counter1 = 4;
      const int counter2 = 2;
      const cuda::discard_iterator iter1(counter1);
      const cuda::discard_iterator iter2(counter2);
      assert(iter1 - iter2 == counter2 - counter1);
      assert(iter2 - iter1 == counter1 - counter2);

      static_assert(cuda::std::is_same_v<decltype(iter1 - iter2), cuda::std::ptrdiff_t>);
    }
  }

  { // operator-=(iter_difference_t<I> n)
    const int counter = 3;
    const int diff    = 2;
    cuda::discard_iterator iter(counter);
    assert((iter -= diff) == cuda::discard_iterator(1));
    assert((iter -= 0) == cuda::discard_iterator(1));

    static_assert(cuda::std::is_same_v<decltype(iter -= 2), cuda::discard_iterator&>);
  }

  { // operator-(const discard_iterator& x, default_sentinel_t)
    {
      const int counter = 3;
      cuda::discard_iterator iter(counter);
      assert((iter - cuda::std::default_sentinel) == -counter);

      static_assert(cuda::std::is_same_v<decltype(iter - cuda::std::default_sentinel), cuda::std::ptrdiff_t>);
    }

    {
      const int counter = 3;
      const cuda::discard_iterator iter(counter);
      assert((iter - cuda::std::default_sentinel) == -counter);

      static_assert(cuda::std::is_same_v<decltype(iter - cuda::std::default_sentinel), cuda::std::ptrdiff_t>);
    }
  }

  { // operator-(default_sentinel_t, const discard_iterator& y)
    {
      const int counter = 3;
      cuda::discard_iterator iter(counter);
      assert((cuda::std::default_sentinel - iter) == counter);

      static_assert(cuda::std::is_same_v<decltype(cuda::std::default_sentinel - iter), cuda::std::ptrdiff_t>);
    }

    {
      const int counter = 3;
      const cuda::discard_iterator iter(counter);
      assert((cuda::std::default_sentinel - iter) == counter);

      static_assert(cuda::std::is_same_v<decltype(cuda::std::default_sentinel - iter), cuda::std::ptrdiff_t>);
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
