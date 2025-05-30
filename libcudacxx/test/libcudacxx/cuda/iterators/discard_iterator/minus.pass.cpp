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
      const int index = 3;
      const int diff  = 2;
      cuda::discard_iterator iter(index);
      assert(iter - diff == cuda::discard_iterator(index - diff));
      assert(iter - 0 == cuda::discard_iterator(index));

      static_assert(cuda::std::is_same_v<decltype(iter - 2), cuda::discard_iterator>);
    }

    {
      const int index = 3;
      const int diff  = 2;
      const cuda::discard_iterator iter(index);
      assert(iter - diff == cuda::discard_iterator(index - diff));
      assert(iter - 0 == cuda::discard_iterator(index));

      static_assert(cuda::std::is_same_v<decltype(iter - 2), cuda::discard_iterator>);
    }
  }

  { // operator-(const discard_iterator& x, const discard_iterator& y)
    {
      const int index1 = 4;
      const int index2 = 2;
      cuda::discard_iterator iter1(index1);
      cuda::discard_iterator iter2(index2);
      assert(iter1 - iter2 == index2 - index1);
      assert(iter2 - iter1 == index1 - index2);

      static_assert(cuda::std::is_same_v<decltype(iter1 - iter2), cuda::std::ptrdiff_t>);
    }

    {
      const int index1 = 4;
      const int index2 = 2;
      const cuda::discard_iterator iter1(index1);
      const cuda::discard_iterator iter2(index2);
      assert(iter1 - iter2 == index2 - index1);
      assert(iter2 - iter1 == index1 - index2);

      static_assert(cuda::std::is_same_v<decltype(iter1 - iter2), cuda::std::ptrdiff_t>);
    }
  }

  { // operator-=(iter_difference_t<I> n)
    const int index = 3;
    const int diff  = 2;
    cuda::discard_iterator iter(index);
    assert((iter -= diff) == cuda::discard_iterator(1));
    assert((iter -= 0) == cuda::discard_iterator(1));

    static_assert(cuda::std::is_same_v<decltype(iter -= 2), cuda::discard_iterator&>);
  }

  { // operator-(const discard_iterator& x, default_sentinel_t)
    {
      const int index = 3;
      cuda::discard_iterator iter(index);
      assert((iter - cuda::std::default_sentinel) == -index);

      static_assert(cuda::std::is_same_v<decltype(iter - cuda::std::default_sentinel), cuda::std::ptrdiff_t>);
    }

    {
      const int index = 3;
      const cuda::discard_iterator iter(index);
      assert((iter - cuda::std::default_sentinel) == -index);

      static_assert(cuda::std::is_same_v<decltype(iter - cuda::std::default_sentinel), cuda::std::ptrdiff_t>);
    }
  }

  { // operator-(default_sentinel_t, const discard_iterator& y)
    {
      const int index = 3;
      cuda::discard_iterator iter(index);
      assert((cuda::std::default_sentinel - iter) == index);

      static_assert(cuda::std::is_same_v<decltype(cuda::std::default_sentinel - iter), cuda::std::ptrdiff_t>);
    }

    {
      const int index = 3;
      const cuda::discard_iterator iter(index);
      assert((cuda::std::default_sentinel - iter) == index);

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
