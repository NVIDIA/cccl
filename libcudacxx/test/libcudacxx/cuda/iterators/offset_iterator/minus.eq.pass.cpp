//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr offset_iterator& operator-=(iter_difference_t<I> n);

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
_CCCL_CONCEPT MinusEqEnabled = _CCCL_REQUIRES_EXPR((Iter), Iter& iter)((iter -= 1));

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    using offset_iter = cuda::offset_iterator<int*>;
    const int offset  = 3;
    const int diff    = 2;
    offset_iter iter(buffer, offset);
    assert((iter -= diff) == offset_iter(buffer, 1));
    assert((iter -= 0) == offset_iter(buffer, 1));
    assert(iter.offset() == 1);

    static_assert(cuda::std::is_same_v<decltype(iter -= 2), offset_iter&>);
  }

  {
    using offset_iter  = cuda::offset_iterator<int*, random_access_iterator<const int*>>;
    const int offset[] = {4, 3, 2, 5};
    const int diff     = 2;
    offset_iter iter(buffer, random_access_iterator<const int*>{offset + 3});
    assert((iter -= diff) == offset_iter(buffer, random_access_iterator<const int*>{offset + 1}));
    assert((iter -= 0) == offset_iter(buffer, random_access_iterator<const int*>{offset + 1}));
    assert(iter.offset() == 3);

    static_assert(cuda::std::is_same_v<decltype(iter -= 2), offset_iter&>);
  }

  {
    static_assert(MinusEqEnabled<cuda::offset_iterator<int*>>);
    static_assert(!MinusEqEnabled<const cuda::offset_iterator<int*>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
