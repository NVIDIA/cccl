//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template <class ForwardIterator, class T>
//     void iota(ForwardIterator first, ForwardIterator last, T value);

#include <cuda/std/cassert>
#include <cuda/std/numeric>

#include "test_iterators.h"
#include "test_macros.h"

template <class InIter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int ia[]         = {1, 2, 3, 4, 5};
  int ir[]         = {5, 6, 7, 8, 9};
  const unsigned s = sizeof(ia) / sizeof(ia[0]);
  cuda::std::iota(InIter(ia), InIter(ia + s), 5);
  for (unsigned i = 0; i < s; ++i)
  {
    assert(ia[i] == ir[i]);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<int*>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014
  return 0;
}
