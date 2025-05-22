//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr offset_iterator operator+(iter_difference_t<I> n) const;
// friend constexpr offset_iterator operator+(iter_difference_t<I> n, const offset_iterator& x);
// constexpr offset_iterator& operator+=(iter_difference_t<I> n);

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
_CCCL_CONCEPT PlusEnabled = _CCCL_REQUIRES_EXPR((Iter), Iter& iter)((iter + 1));

template <class Iter>
_CCCL_CONCEPT PlusEqEnabled = _CCCL_REQUIRES_EXPR((Iter), Iter& iter)((iter += 1));

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  { // operator+(iter_difference_t<I> n)
    {
      using offset_iterator = cuda::offset_iterator<int*>;
      const int offset      = 2;
      const int diff        = 3;
      offset_iterator iter(buffer, offset);
      assert(iter + diff == offset_iterator(buffer + offset, diff));
      assert(iter + 0 == offset_iterator(buffer, offset));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), offset_iterator>);
    }

    {
      using offset_iterator = cuda::offset_iterator<int*>;
      const int offset      = 2;
      const int diff        = 3;
      const offset_iterator iter(buffer, offset);
      assert(iter + diff == offset_iterator(buffer + offset, diff));
      assert(iter + 0 == offset_iterator(buffer, offset));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), offset_iterator>);
    }

    {
      using offset_iterator = cuda::offset_iterator<int*, const int*>;
      const int offset[]    = {2, 3, 4, 5};
      const int diff        = 3;
      offset_iterator iter(buffer, offset);
      assert(iter + diff == offset_iterator(buffer, offset + diff));
      assert(iter + 0 == offset_iterator(buffer, offset));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), offset_iterator>);
    }

    {
      using offset_iterator = cuda::offset_iterator<int*, const int*>;
      const int offset[]    = {2, 3, 4, 5};
      const int diff        = 3;
      const offset_iterator iter(buffer, offset);
      assert(iter + diff == offset_iterator(buffer, offset + diff));
      assert(iter + 0 == offset_iterator(buffer, offset));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), offset_iterator>);
    }

    {
      using offset_iterator = cuda::offset_iterator<int*, random_access_iterator<const int*>>;
      const int offset[]    = {2, 3, 4, 5};
      const int diff        = 3;
      offset_iterator iter(buffer, random_access_iterator<const int*>{offset});
      assert(iter + diff == offset_iterator(buffer, random_access_iterator<const int*>{offset + diff}));
      assert(iter + 0 == offset_iterator(buffer, random_access_iterator<const int*>{offset}));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), offset_iterator>);
    }

    {
      using offset_iterator = cuda::offset_iterator<int*, random_access_iterator<const int*>>;
      const int offset[]    = {2, 3, 4, 5};
      const int diff        = 3;
      const offset_iterator iter(buffer, random_access_iterator<const int*>{offset});
      assert(iter + diff == offset_iterator(buffer, random_access_iterator<const int*>{offset + diff}));
      assert(iter + 0 == offset_iterator(buffer, random_access_iterator<const int*>{offset}));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), offset_iterator>);
    }
  }

  { // operator+(iter_difference_t<I> n, const offset_iterator& x)
    {
      using offset_iterator = cuda::offset_iterator<int*>;
      const int offset      = 2;
      const int diff        = 3;
      offset_iterator iter(buffer, offset);
      assert(diff + iter == offset_iterator(buffer, offset + diff));
      assert(0 + iter == offset_iterator(buffer, offset));

      static_assert(cuda::std::is_same_v<decltype(2 + iter), offset_iterator>);
    }

    {
      using offset_iterator = cuda::offset_iterator<int*>;
      const int offset      = 2;
      const int diff        = 3;
      const offset_iterator iter(buffer, offset);
      assert(diff + iter == offset_iterator(buffer, offset + diff));
      assert(0 + iter == offset_iterator(buffer, offset));

      static_assert(cuda::std::is_same_v<decltype(2 + iter), offset_iterator>);
    }

    {
      using offset_iterator = cuda::offset_iterator<int*, const int*>;
      const int offset[]    = {2, 3, 4, 5};
      const int diff        = 3;
      offset_iterator iter(buffer, offset);
      assert(diff + iter == offset_iterator(buffer, offset + diff));
      assert(0 + iter == offset_iterator(buffer, offset));

      static_assert(cuda::std::is_same_v<decltype(2 + iter), offset_iterator>);
    }

    {
      using offset_iterator = cuda::offset_iterator<int*, const int*>;
      const int offset[]    = {2, 3, 4, 5};
      const int diff        = 3;
      const offset_iterator iter(buffer, offset);
      assert(diff + iter == offset_iterator(buffer, offset + diff));
      assert(0 + iter == offset_iterator(buffer, offset));

      static_assert(cuda::std::is_same_v<decltype(2 + iter), offset_iterator>);
    }

    {
      using offset_iterator = cuda::offset_iterator<int*, random_access_iterator<const int*>>;
      const int offset[]    = {2, 3, 4, 5};
      const int diff        = 3;
      offset_iterator iter(buffer, random_access_iterator<const int*>{offset});
      assert(diff + iter == offset_iterator(buffer, random_access_iterator<const int*>{offset + diff}));
      assert(0 + iter == offset_iterator(buffer, random_access_iterator<const int*>{offset}));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), offset_iterator>);
    }

    {
      using offset_iterator = cuda::offset_iterator<int*, random_access_iterator<const int*>>;
      const int offset[]    = {2, 3, 4, 5};
      const int diff        = 3;
      const offset_iterator iter(buffer, random_access_iterator<const int*>{offset});
      assert(diff + iter == offset_iterator(buffer, random_access_iterator<const int*>{offset + diff}));
      assert(0 + iter == offset_iterator(buffer, random_access_iterator<const int*>{offset}));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), offset_iterator>);
    }
  }

  { // operator+=(iter_difference_t<I> n)
    {
      using offset_iterator = cuda::offset_iterator<int*>;
      const int offset      = 2;
      const int diff        = 3;
      offset_iterator iter(buffer, offset);
      assert((iter += 0) == offset_iterator(buffer, offset));
      assert((iter += diff) == offset_iterator(buffer, offset + diff));

      static_assert(cuda::std::is_same_v<decltype(iter += 2), offset_iterator&>);
    }

    {
      using offset_iterator = cuda::offset_iterator<int*, const int*>;
      const int offset[]    = {2, 3, 4, 5};
      const int diff        = 3;
      offset_iterator iter(buffer, offset);
      assert((iter += 0) == offset_iterator(buffer, offset));
      assert((iter += diff) == offset_iterator(buffer, offset + diff));

      static_assert(cuda::std::is_same_v<decltype(iter += 2), offset_iterator&>);
    }

    {
      using offset_iterator = cuda::offset_iterator<int*, random_access_iterator<const int*>>;
      const int offset[]    = {2, 3, 4, 5};
      const int diff        = 3;
      offset_iterator iter(buffer, random_access_iterator<const int*>{offset});
      assert((iter += 0) == offset_iterator(buffer, random_access_iterator<const int*>{offset}));
      assert((iter += diff) == offset_iterator(buffer, random_access_iterator<const int*>{offset + diff}));

      static_assert(cuda::std::is_same_v<decltype(iter += 2), offset_iterator&>);
    }

    {
      static_assert(PlusEnabled<cuda::offset_iterator<int*>>);
      static_assert(PlusEqEnabled<cuda::offset_iterator<int*>>);
      static_assert(!PlusEqEnabled<const cuda::offset_iterator<int*>>);
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
