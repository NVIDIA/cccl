//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr iter_rvalue_reference_t<I>
//   iter_move(const permutation_iterator& i)
//     noexcept(noexcept(ranges::iter_move(i.current)));

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using baseIter     = random_access_iterator<int*>;
  using indexIter    = random_access_iterator<const int*>;
  int buffer[]       = {1, 2, 3, 4, 5, 6, 7, 8};
  const int offset[] = {2};
  auto iter          = cuda::permutation_iterator{baseIter{buffer}, indexIter{offset}};
  assert(cuda::std::ranges::iter_move(iter) == buffer[*offset]);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::iter_move(iter)), int&&>);

  // The test iterators are not noexcept
  static_assert(!noexcept(cuda::std::ranges::iter_move(iter)));
  static_assert(noexcept(cuda::std::ranges::iter_move(cuda::permutation_iterator<int*, int*>())));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
