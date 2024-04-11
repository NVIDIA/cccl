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

// ranges::prev(it, n, bound)

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/iterator>
#include <cuda/std/utility>

#include "test_iterators.h"

template <typename It>
__host__ __device__ constexpr void check(int* first, int* last, cuda::std::iter_difference_t<It> n, int* expected)
{
  It it(last);
  It sent(first); // for cuda::std::ranges::prev, the sentinel *must* have the same type as the iterator

  decltype(auto) result = cuda::std::ranges::prev(cuda::std::move(it), n, cuda::std::move(sent));
  static_assert(cuda::std::same_as<decltype(result), It>);
  assert(base(result) == expected);
}

__host__ __device__ constexpr bool test()
{
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  for (int size = 0; size != 10; ++size)
  {
    for (int n = 0; n != 20; ++n)
    {
      int* expected = n > size ? range : range + size - n;
      check<bidirectional_iterator<int*>>(range, range + size, n, expected);
      check<random_access_iterator<int*>>(range, range + size, n, expected);
      check<contiguous_iterator<int*>>(range, range + size, n, expected);
      check<int*>(range, range + size, n, expected);
    }
  }

  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
