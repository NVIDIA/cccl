//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// template <InputIterator Iter>
//   Iter prev(Iter x, Iter::difference_type n = 1);

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ TEST_CONSTEXPR_CXX14 void
check_prev_n(It it, typename cuda::std::iterator_traits<It>::difference_type n, It result)
{
  static_assert(cuda::std::is_same<decltype(cuda::std::prev(it, n)), It>::value, "");
  assert(cuda::std::prev(it, n) == result);

  It (*prev_ptr)(It, typename cuda::std::iterator_traits<It>::difference_type) = cuda::std::prev;
  assert(prev_ptr(it, n) == result);
}

template <class It>
__host__ __device__ TEST_CONSTEXPR_CXX14 void check_prev_1(It it, It result)
{
  static_assert(cuda::std::is_same<decltype(cuda::std::prev(it)), It>::value, "");
  assert(cuda::std::prev(it) == result);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  const char* s = "1234567890";
  check_prev_n(forward_iterator<const char*>(s), -10, forward_iterator<const char*>(s + 10));
  check_prev_n(bidirectional_iterator<const char*>(s + 10), 10, bidirectional_iterator<const char*>(s));
  check_prev_n(bidirectional_iterator<const char*>(s), -10, bidirectional_iterator<const char*>(s + 10));
  check_prev_n(random_access_iterator<const char*>(s + 10), 10, random_access_iterator<const char*>(s));
  check_prev_n(random_access_iterator<const char*>(s), -10, random_access_iterator<const char*>(s + 10));
  check_prev_n(s + 10, 10, s);

  check_prev_1(bidirectional_iterator<const char*>(s + 1), bidirectional_iterator<const char*>(s));
  check_prev_1(random_access_iterator<const char*>(s + 1), random_access_iterator<const char*>(s));
  check_prev_1(s + 1, s);

  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER > 2014
  static_assert(tests(), "");
#endif
  return 0;
}
