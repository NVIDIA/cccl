//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile

// UNSUPPORTED: nvrtc

#include <cuda/iterator>
#include <cuda/std/cassert>

#include <vector>

#include "test_macros.h"

void test()
{
  std::vector<int> vec{1, 2, 3, 0};

  { // constructors
    using Iter                 = typename std::vector<int>::iterator;
    using permutation_iterator = cuda::permutation_iterator<Iter, Iter>;

    const permutation_iterator default_constructed{};
    permutation_iterator value_constructed{vec.begin(), vec.begin()};

    permutation_iterator copy_constructed{default_constructed};
    permutation_iterator move_constructed{::cuda::std::move(value_constructed)};

    [[maybe_unused]] permutation_iterator copy_assigned{};
    copy_assigned = copy_constructed;

    [[maybe_unused]] permutation_iterator move_assigned{};
    move_assigned = ::cuda::std::move(move_constructed);
  }

  cuda::permutation_iterator iter1{vec.begin(), vec.begin()};
  const cuda::permutation_iterator iter2{vec.begin(), vec.begin() + 1};
  assert(iter1 != iter2);

  {
    assert(++iter1 == iter2);
    assert(--iter1 != iter2);
  }

  {
    assert(iter1++ != iter2);
    assert(iter1-- == iter2);
  }

  {
    assert(iter1 + 1 == iter2);
    assert(iter1 - 1 != iter2);
    assert(iter2 - iter1 == 1);
  }

  {
    iter1 += 1;
    assert(iter1 == iter2);
    iter1 -= 1;
    assert(iter1 != iter2);
  }

  {
    assert(iter1[1] == vec[vec[1]]);
    assert(*iter1 == vec[vec[0]]);

    assert(iter2[1] == vec[vec[2]]);
    assert(*iter2 == vec[vec[1]]);
  }

  {
    assert(iter1.base() == vec.begin());
    assert(iter2.base() == vec.begin());
  }

  {
    cuda::std::ranges::iter_swap(iter1, iter2);
    assert(cuda::std::ranges::iter_move(iter1) == vec[vec[0]]);
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
