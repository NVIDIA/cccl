//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr offset_iterator operator-(iter_difference_t<I> n) const;

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
_CCCL_CONCEPT MinusEnabled = _CCCL_REQUIRES_EXPR((Iter), Iter& iter)((iter - 1));

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    using offset_iterator = cuda::offset_iterator<int*>;
    const int offset      = 3;
    const int diff        = 2;
    offset_iterator iter(buffer, offset);
    assert(iter - diff == offset_iterator(buffer, 1));
    assert(iter - 0 == offset_iterator(buffer, offset));
    assert(iter.offset() == offset);

    static_assert(cuda::std::is_same_v<decltype(iter - 2), offset_iterator>);
  }

  {
    using offset_iterator = cuda::offset_iterator<int*>;
    const int offset      = 3;
    const int diff        = 2;
    const offset_iterator iter(buffer, offset);
    assert(iter - diff == offset_iterator(buffer, 1));
    assert(iter - 0 == offset_iterator(buffer, offset));
    assert(iter.offset() == offset);

    static_assert(cuda::std::is_same_v<decltype(iter - 2), offset_iterator>);
  }

  {
    using offset_iterator = cuda::offset_iterator<int*, const int*>;
    const int offset[]    = {4, 3, 2, 5};
    const int diff        = 2;
    offset_iterator iter(buffer, offset + 3);
    assert(iter - diff == offset_iterator(buffer, offset + 1));
    assert(iter - 0 == offset_iterator(buffer, offset + 3));
    assert(iter.offset() == 5);

    static_assert(cuda::std::is_same_v<decltype(iter - 2), offset_iterator>);
  }

  {
    using offset_iterator = cuda::offset_iterator<int*, const int*>;
    const int offset[]    = {4, 3, 2, 5};
    const int diff        = 2;
    const offset_iterator iter(buffer, offset + 3);
    assert(iter - diff == offset_iterator(buffer, offset + 1));
    assert(iter - 0 == offset_iterator(buffer, offset + 3));
    assert(iter.offset() == 5);

    static_assert(cuda::std::is_same_v<decltype(iter - 2), offset_iterator>);
  }

  {
    using offset_iterator = cuda::offset_iterator<int*, random_access_iterator<const int*>>;
    const int offset[]    = {4, 3, 2, 5};
    const int diff        = 2;
    offset_iterator iter(buffer, random_access_iterator<const int*>{offset + 3});
    assert(iter - diff == offset_iterator(buffer, random_access_iterator<const int*>{offset + 1}));
    assert(iter - 0 == offset_iterator(buffer, random_access_iterator<const int*>{offset + 3}));
    assert(iter.offset() == 5);

    static_assert(cuda::std::is_same_v<decltype(iter - 2), offset_iterator>);
  }

  {
    using offset_iterator = cuda::offset_iterator<int*, random_access_iterator<const int*>>;
    const int offset[]    = {4, 3, 2, 5};
    const int diff        = 2;
    const offset_iterator iter(buffer, random_access_iterator<const int*>{offset + 3});
    assert(iter - diff == offset_iterator(buffer, random_access_iterator<const int*>{offset + 1}));
    assert(iter - 0 == offset_iterator(buffer, random_access_iterator<const int*>{offset + 3}));
    assert(iter.offset() == 5);

    static_assert(cuda::std::is_same_v<decltype(iter - 2), offset_iterator>);
  }

  {
    static_assert(MinusEnabled<cuda::offset_iterator<int*, random_access_iterator<int*>>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
