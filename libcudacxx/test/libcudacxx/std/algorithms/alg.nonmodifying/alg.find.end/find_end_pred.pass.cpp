//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter1, ForwardIterator Iter2,
//          Predicate<auto, Iter1::value_type, Iter2::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr Iter1  // constexpr after C++17
//   find_end(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Pred pred);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "test_iterators.h"
#include "test_macros.h"

struct count_equal
{
  unsigned& count_;

  __host__ __device__ constexpr count_equal(unsigned& count) noexcept
      : count_(count)
  {}

  template <class T>
  __host__ __device__ TEST_CONSTEXPR_CXX14 bool operator()(const T& x, const T& y) const noexcept
  {
    ++count_;
    return x == y;
  }
};

template <class Iter1, class Iter2>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  unsigned count_equal_count = 0;

  int ia[]          = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  int b[]           = {0};
  count_equal_count = 0;
  assert(cuda::std::find_end(Iter1(ia), Iter1(ia + sa), Iter2(b), Iter2(b + 1), count_equal{count_equal_count})
         == Iter1(ia + sa - 1));
  assert(count_equal_count <= 1 * (sa - 1 + 1));
  int c[]           = {0, 1};
  count_equal_count = 0;
  assert(cuda::std::find_end(Iter1(ia), Iter1(ia + sa), Iter2(c), Iter2(c + 2), count_equal{count_equal_count})
         == Iter1(ia + 18));
  assert(count_equal_count <= 2 * (sa - 2 + 1));
  int d[]           = {0, 1, 2};
  count_equal_count = 0;
  assert(cuda::std::find_end(Iter1(ia), Iter1(ia + sa), Iter2(d), Iter2(d + 3), count_equal{count_equal_count})
         == Iter1(ia + 15));
  assert(count_equal_count <= 3 * (sa - 3 + 1));
  int e[]           = {0, 1, 2, 3};
  count_equal_count = 0;
  assert(cuda::std::find_end(Iter1(ia), Iter1(ia + sa), Iter2(e), Iter2(e + 4), count_equal{count_equal_count})
         == Iter1(ia + 11));
  assert(count_equal_count <= 4 * (sa - 4 + 1));
  int f[]           = {0, 1, 2, 3, 4};
  count_equal_count = 0;
  assert(cuda::std::find_end(Iter1(ia), Iter1(ia + sa), Iter2(f), Iter2(f + 5), count_equal{count_equal_count})
         == Iter1(ia + 6));
  assert(count_equal_count <= 5 * (sa - 5 + 1));
  int g[]           = {0, 1, 2, 3, 4, 5};
  count_equal_count = 0;
  assert(cuda::std::find_end(Iter1(ia), Iter1(ia + sa), Iter2(g), Iter2(g + 6), count_equal{count_equal_count})
         == Iter1(ia));
  assert(count_equal_count <= 6 * (sa - 6 + 1));
  int h[]           = {0, 1, 2, 3, 4, 5, 6};
  count_equal_count = 0;
  assert(cuda::std::find_end(Iter1(ia), Iter1(ia + sa), Iter2(h), Iter2(h + 7), count_equal{count_equal_count})
         == Iter1(ia + sa));
  assert(count_equal_count <= 7 * (sa - 7 + 1));
  count_equal_count = 0;
  assert(cuda::std::find_end(Iter1(ia), Iter1(ia + sa), Iter2(b), Iter2(b), count_equal{count_equal_count})
         == Iter1(ia + sa));
  assert(count_equal_count <= 0);
  count_equal_count = 0;
  assert(cuda::std::find_end(Iter1(ia), Iter1(ia), Iter2(b), Iter2(b + 1), count_equal{count_equal_count})
         == Iter1(ia));
  assert(count_equal_count <= 0);

  return true;
}

int main(int, char**)
{
  test<forward_iterator<const int*>, forward_iterator<const int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<forward_iterator<const int*>, random_access_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<const int*>>();
  test<random_access_iterator<const int*>, forward_iterator<const int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<const int*>>();

#if TEST_STD_VER > 2011
  static_assert(test<forward_iterator<const int*>, forward_iterator<const int*>>(), "");
  static_assert(test<forward_iterator<const int*>, bidirectional_iterator<const int*>>(), "");
  static_assert(test<forward_iterator<const int*>, random_access_iterator<const int*>>(), "");
  static_assert(test<bidirectional_iterator<const int*>, forward_iterator<const int*>>(), "");
  static_assert(test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>>(), "");
  static_assert(test<bidirectional_iterator<const int*>, random_access_iterator<const int*>>(), "");
  static_assert(test<random_access_iterator<const int*>, forward_iterator<const int*>>(), "");
  static_assert(test<random_access_iterator<const int*>, bidirectional_iterator<const int*>>(), "");
  static_assert(test<random_access_iterator<const int*>, random_access_iterator<const int*>>(), "");
#endif

  return 0;
}
