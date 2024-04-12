//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// ranges::prev(it)

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"

template <class It>
__host__ __device__ constexpr void check(int* first, int* expected)
{
  It it(first);
  decltype(auto) result = cuda::std::ranges::prev(cuda::std::move(it));
  static_assert(cuda::std::same_as<decltype(result), It>);
  assert(base(result) == expected);
}

__host__ __device__ constexpr bool test()
{
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  for (int n = 1; n != 10; ++n)
  {
    check<bidirectional_iterator<int*>>(range + n, range + n - 1);
    check<random_access_iterator<int*>>(range + n, range + n - 1);
    check<contiguous_iterator<int*>>(range + n, range + n - 1);
    check<int*>(range + n, range + n - 1);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
