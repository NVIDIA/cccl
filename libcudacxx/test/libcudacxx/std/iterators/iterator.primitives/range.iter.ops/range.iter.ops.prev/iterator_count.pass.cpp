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

// ranges::prev(it, n)

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/iterator>
#include <cuda/std/utility>

#include "test_iterators.h"

template <typename It>
__host__ __device__ constexpr void check(int* first, cuda::std::iter_difference_t<It> n, int* expected)
{
  It it(first);
  decltype(auto) result = cuda::std::ranges::prev(cuda::std::move(it), n);
  static_assert(cuda::std::same_as<decltype(result), It>);
  assert(base(result) == expected);
}

__host__ __device__ constexpr bool test()
{
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  // Check prev() forward
  for (int n = 0; n != 10; ++n)
  {
    check<bidirectional_iterator<int*>>(range + n, n, range);
    check<random_access_iterator<int*>>(range + n, n, range);
    check<contiguous_iterator<int*>>(range + n, n, range);
    check<int*>(range + n, n, range);
  }

  // Check prev() backward
  for (int n = 0; n != 10; ++n)
  {
    check<bidirectional_iterator<int*>>(range, -n, range + n);
    check<random_access_iterator<int*>>(range, -n, range + n);
    check<contiguous_iterator<int*>>(range, -n, range + n);
    check<int*>(range, -n, range + n);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
