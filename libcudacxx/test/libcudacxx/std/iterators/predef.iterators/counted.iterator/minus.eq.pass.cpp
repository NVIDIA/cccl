//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr counted_iterator& operator-=(iter_difference_t<I> n)
//   requires random_access_iterator<I>;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
_CCCL_CONCEPT MinusEqEnabled = _CCCL_REQUIRES_EXPR((Iter), Iter& iter)((iter -= 1));

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    using Counted = cuda::std::counted_iterator<random_access_iterator<int*>>;
    Counted iter(random_access_iterator<int*>{buffer + 2}, 6);
    assert((iter -= 2) == Counted(random_access_iterator<int*>{buffer}, 8));
    assert((iter -= 0) == Counted(random_access_iterator<int*>{buffer}, 8));
    assert(iter.count() == 8);

    static_assert(cuda::std::is_same_v<decltype(iter -= 2), Counted&>);
  }
  {
    using Counted = cuda::std::counted_iterator<contiguous_iterator<int*>>;
    Counted iter(contiguous_iterator<int*>{buffer + 2}, 6);
    assert((iter -= 2) == Counted(contiguous_iterator<int*>{buffer}, 8));
    assert((iter -= 0) == Counted(contiguous_iterator<int*>{buffer}, 8));
    assert(iter.count() == 8);

    static_assert(cuda::std::is_same_v<decltype(iter -= 2), Counted&>);
  }
  {
    static_assert(MinusEqEnabled<cuda::std::counted_iterator<random_access_iterator<int*>>>);
    static_assert(!MinusEqEnabled<const cuda::std::counted_iterator<random_access_iterator<int*>>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
