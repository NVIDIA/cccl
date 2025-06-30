//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<indirectly_swappable<I> I2>
//   friend constexpr void
//     iter_swap(const permutation_iterator& x, const permutation_iterator<I2>& y)
//       noexcept(noexcept(ranges::iter_swap(x.current, y.current)));

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using baseIter      = random_access_iterator<int*>;
  using indexIter     = random_access_iterator<const int*>;
  int buffer[]        = {1, 2, 3, 4, 5, 6, 7, 8};
  const int offset1[] = {2};
  const int offset2[] = {4};
  auto iter1          = cuda::permutation_iterator(baseIter{buffer}, indexIter{offset1});
  auto iter2          = cuda::permutation_iterator(baseIter{buffer}, indexIter{offset2});

  assert(*iter1 == 3);
  assert(*iter2 == 5);
  cuda::std::ranges::iter_swap(iter1, iter2);
  assert(*iter1 == 5);
  assert(*iter2 == 3);
  cuda::std::ranges::iter_swap(iter1, iter2);
  assert(*iter1 == 3);
  assert(*iter2 == 5);

  // The test iterators are not noexcept
  static_assert(!noexcept(cuda::std::ranges::iter_swap(iter1, iter2)));
  static_assert(noexcept(
    cuda::std::ranges::iter_swap(cuda::permutation_iterator<int*, int*>(), cuda::permutation_iterator<int*, int*>())));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
