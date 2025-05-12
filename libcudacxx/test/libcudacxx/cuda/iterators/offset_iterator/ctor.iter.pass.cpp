//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr offset_iterator(I x, iter_difference_t<I> n);

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    const int offset = 2;
    cuda::offset_iterator iter(random_access_iterator<int*>{buffer}, offset);
    assert(iter.base() == random_access_iterator<int*>{buffer});
    assert(iter.offset() == 2);
  }

  {
    const int offset[] = {2};
    cuda::offset_iterator iter(random_access_iterator<int*>{buffer}, random_access_iterator<const int*>{offset});
    assert(iter.base() == random_access_iterator<int*>{buffer});
    assert(iter.offset() == 2);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
