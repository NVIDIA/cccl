//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// reverse_iterator

// template <RandomAccessIterator Iterator>
// reverse_iterator<Iter> operator+(Iter::difference_type n, const reverse_iterator<Iter>& x); // constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ TEST_CONSTEXPR_CXX14 void
test(It i, typename cuda::std::iterator_traits<It>::difference_type n, It x)
{
  const cuda::std::reverse_iterator<It> r(i);
  cuda::std::reverse_iterator<It> rr = n + r;
  assert(rr.base() == x);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  const char* s = "1234567890";
  test(random_access_iterator<const char*>(s + 5), 5, random_access_iterator<const char*>(s));
  test(s + 5, 5, s);
  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER > 2011
  static_assert(tests(), "");
#endif
  return 0;
}
