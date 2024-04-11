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

// reverse_iterator operator--(int); // constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test(It i, It x)
{
  cuda::std::reverse_iterator<It> r(i);
  cuda::std::reverse_iterator<It> rr = r--;
  assert(r.base() == x);
  assert(rr.base() == i);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  const char* s = "123";
  test(bidirectional_iterator<const char*>(s + 1), bidirectional_iterator<const char*>(s + 2));
  test(random_access_iterator<const char*>(s + 1), random_access_iterator<const char*>(s + 2));
  test(s + 1, s + 2);
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
