//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr decltype(auto) operator[](iter_difference_t<I> n) const
//   requires random_access_iterator<I>;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
_CCCL_CONCEPT SubscriptEnabled = _CCCL_REQUIRES_EXPR((Iter), Iter& iter)(unused(iter[1]));

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    cuda::std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
    for (int i = 1; i < 9; ++i)
    {
      assert(iter[i - 1] == i);
    }
  }
  {
    cuda::std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    for (int i = 1; i < 9; ++i)
    {
      assert(iter[i - 1] == i);
    }
  }

  {
    const cuda::std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
    assert(iter[0] == 1);
  }
  {
    const cuda::std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    assert(iter[7] == 8);
  }

  {
    static_assert(SubscriptEnabled<cuda::std::counted_iterator<random_access_iterator<int*>>>);
    static_assert(!SubscriptEnabled<cuda::std::counted_iterator<bidirectional_iterator<int*>>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
