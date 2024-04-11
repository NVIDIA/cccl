//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

//   All of these became constexpr in C++17
//
// template <InputIterator Iter>
//   constexpr void advance(Iter& i, Iter::difference_type n);
//
// template <BidirectionalIterator Iter>
//   constexpr void advance(Iter& i, Iter::difference_type n);
//
// template <RandomAccessIterator Iter>
//   constexpr void advance(Iter& i, Iter::difference_type n);

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class Distance, class It>
__host__ __device__ TEST_CONSTEXPR_CXX14 void check_advance(It it, Distance n, It result)
{
  static_assert(cuda::std::is_same<decltype(cuda::std::advance(it, n)), void>::value, "");
  cuda::std::advance(it, n);
  assert(it == result);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  const char* s = "1234567890";

  // Check with iterator_traits::difference_type
  {
    typedef cuda::std::iterator_traits<const char*>::difference_type Distance;
    check_advance<Distance>(cpp17_input_iterator<const char*>(s), 10, cpp17_input_iterator<const char*>(s + 10));
    check_advance<Distance>(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s + 10));
    check_advance<Distance>(bidirectional_iterator<const char*>(s + 5), 5, bidirectional_iterator<const char*>(s + 10));
    check_advance<Distance>(bidirectional_iterator<const char*>(s + 5), -5, bidirectional_iterator<const char*>(s));
    check_advance<Distance>(random_access_iterator<const char*>(s + 5), 5, random_access_iterator<const char*>(s + 10));
    check_advance<Distance>(random_access_iterator<const char*>(s + 5), -5, random_access_iterator<const char*>(s));
    check_advance<Distance>(s + 5, 5, s + 10);
    check_advance<Distance>(s + 5, -5, s);
  }

  // Also check with other distance types
  {
    typedef int Distance;
    check_advance<Distance>(cpp17_input_iterator<const char*>(s), 10, cpp17_input_iterator<const char*>(s + 10));
    check_advance<Distance>(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s + 10));
    check_advance<Distance>(bidirectional_iterator<const char*>(s), 10, bidirectional_iterator<const char*>(s + 10));
    check_advance<Distance>(random_access_iterator<const char*>(s), 10, random_access_iterator<const char*>(s + 10));
  }

  // Check with unsigned distance types to catch signedness-change issues
  {
    typedef cuda::std::size_t Distance;
    check_advance<Distance>(cpp17_input_iterator<const char*>(s), 10u, cpp17_input_iterator<const char*>(s + 10));
    check_advance<Distance>(forward_iterator<const char*>(s), 10u, forward_iterator<const char*>(s + 10));
    check_advance<Distance>(bidirectional_iterator<const char*>(s), 10u, bidirectional_iterator<const char*>(s + 10));
    check_advance<Distance>(random_access_iterator<const char*>(s), 10u, random_access_iterator<const char*>(s + 10));
  }

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
