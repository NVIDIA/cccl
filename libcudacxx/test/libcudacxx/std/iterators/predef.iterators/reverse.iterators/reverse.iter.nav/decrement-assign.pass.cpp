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

// requires RandomAccessIterator<Iter>
// reverse_iterator& operator-=(difference_type n); // constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ TEST_CONSTEXPR_CXX14 void
test(It i, typename cuda::std::iterator_traits<It>::difference_type n, It x)
{
  cuda::std::reverse_iterator<It> r(i);
  cuda::std::reverse_iterator<It>& rr = r -= n;
  assert(r.base() == x);
  assert(&rr == &r);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  const char* s = "1234567890";
  test(random_access_iterator<const char*>(s + 5), 5, random_access_iterator<const char*>(s + 10));
  test(s + 5, 5, s + 10);
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
