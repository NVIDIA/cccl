//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr offset_iterator& operator--()
// constexpr offset_iterator operator--(int)

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
_CCCL_CONCEPT MinusEnabled = _CCCL_REQUIRES_EXPR((Iter), Iter& iter)((iter--), (--iter));

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    const int offset      = 3;
    using offset_iterator = cuda::offset_iterator<random_access_iterator<int*>>;
    offset_iterator iter(random_access_iterator<int*>{buffer}, offset);
    assert(*iter == buffer[offset]);
    assert(iter-- == offset_iterator(random_access_iterator<int*>{buffer}, offset + 0));
    assert(*iter == buffer[offset - 1]);
    assert(--iter == offset_iterator(random_access_iterator<int*>{buffer}, offset - 2));
    assert(*iter == buffer[offset - 2]);
    assert(iter.offset() == 1);

    static_assert(cuda::std::is_same_v<decltype(iter--), offset_iterator>);
    static_assert(cuda::std::is_same_v<decltype(--iter), offset_iterator&>);
  }

  {
    const int offset[]    = {4, 2, 6};
    using offset_iterator = cuda::offset_iterator<random_access_iterator<int*>, random_access_iterator<const int*>>;
    offset_iterator iter(random_access_iterator<int*>{buffer}, random_access_iterator<const int*>{offset + 2});
    assert(*iter == buffer[offset[2]]);
    assert(iter--
           == offset_iterator(random_access_iterator<int*>{buffer}, random_access_iterator<const int*>{offset + 2}));
    assert(*iter == buffer[offset[1]]);
    assert(--iter
           == offset_iterator(random_access_iterator<int*>{buffer}, random_access_iterator<const int*>{offset + 0}));
    assert(*iter == buffer[offset[0]]);
    assert(iter.offset() == 4);

    static_assert(cuda::std::is_same_v<decltype(iter--), offset_iterator>);
    static_assert(cuda::std::is_same_v<decltype(--iter), offset_iterator&>);
  }

  {
    static_assert(MinusEnabled<cuda::offset_iterator<contiguous_iterator<int*>>>);
    static_assert(!MinusEnabled<const cuda::offset_iterator<contiguous_iterator<int*>>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
