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
//   Iter::difference_type
//   distance(Iter first, Iter last);
//
// template <RandomAccessIterator Iter>
//   Iter::difference_type
//   distance(Iter first, Iter last);

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ TEST_CONSTEXPR_CXX14 void
check_distance(It first, It last, typename cuda::std::iterator_traits<It>::difference_type dist)
{
  typedef typename cuda::std::iterator_traits<It>::difference_type Difference;
  static_assert(cuda::std::is_same<decltype(cuda::std::distance(first, last)), Difference>::value, "");
  assert(cuda::std::distance(first, last) == dist);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  const char* s = "1234567890";
  check_distance(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 10), 10);
  check_distance(forward_iterator<const char*>(s), forward_iterator<const char*>(s + 10), 10);
  check_distance(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s + 10), 10);
  check_distance(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s + 10), 10);
  check_distance(s, s + 10, 10);
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
