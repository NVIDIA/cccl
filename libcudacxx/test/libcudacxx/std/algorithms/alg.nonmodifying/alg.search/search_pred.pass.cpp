//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter1, ForwardIterator Iter2>
//   requires HasEqualTo<Iter1::value_type, Iter2::value_type>
//   constexpr Iter1     // constexpr after C++17
//   search(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4018) // signed/unsigned mismatch
TEST_DIAG_SUPPRESS_GCC("-Wsign-compare")
TEST_DIAG_SUPPRESS_CLANG("-Wsign-compare")

struct count_equal
{
  __host__ __device__ constexpr count_equal(int& count) noexcept
      : count(count)
  {}

  template <class T>
  __host__ __device__ constexpr bool operator()(const T& x, const T& y)
  {
    ++count;
    return x == y;
  }

  int& count;
};

template <class Iter1, class Iter2>
__host__ __device__ constexpr void test()
{
  int ia[]              = {0, 1, 2, 3, 4, 5};
  const unsigned sa     = sizeof(ia) / sizeof(ia[0]);
  int count_equal_count = 0;
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia), Iter2(ia), count_equal{count_equal_count})
         == Iter1(ia));
  assert(count_equal_count <= 0);
  count_equal_count = 0;
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia), Iter2(ia + 1), count_equal{count_equal_count})
         == Iter1(ia));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 1), Iter2(ia + 2), count_equal{count_equal_count})
         == Iter1(ia + 1));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 2), Iter2(ia + 2), count_equal{count_equal_count})
         == Iter1(ia));
  assert(count_equal_count <= 0);
  count_equal_count = 0;
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 2), Iter2(ia + 3), count_equal{count_equal_count})
         == Iter1(ia + 2));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 2), Iter2(ia + 3), count_equal{count_equal_count})
         == Iter1(ia + 2));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search(Iter1(ia), Iter1(ia), Iter2(ia + 2), Iter2(ia + 3), count_equal{count_equal_count})
         == Iter1(ia));
  assert(count_equal_count <= 0);
  count_equal_count = 0;
  assert(
    cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia + sa - 1), Iter2(ia + sa), count_equal{count_equal_count})
    == Iter1(ia + sa - 1));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(
    cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia + sa - 3), Iter2(ia + sa), count_equal{count_equal_count})
    == Iter1(ia + sa - 3));
  assert(count_equal_count <= sa * 3);
  count_equal_count = 0;
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia), Iter2(ia + sa), count_equal{count_equal_count})
         == Iter1(ia));
  assert(count_equal_count <= sa * sa);
  count_equal_count = 0;
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa - 1), Iter2(ia), Iter2(ia + sa), count_equal{count_equal_count})
         == Iter1(ia + sa - 1));
  assert(count_equal_count <= (sa - 1) * sa);
  count_equal_count = 0;
  assert(cuda::std::search(Iter1(ia), Iter1(ia + 1), Iter2(ia), Iter2(ia + sa), count_equal{count_equal_count})
         == Iter1(ia + 1));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  int ib[]          = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4};
  const unsigned sb = sizeof(ib) / sizeof(ib[0]);
  int ic[]          = {1};
  assert(cuda::std::search(Iter1(ib), Iter1(ib + sb), Iter2(ic), Iter2(ic + 1), count_equal{count_equal_count})
         == Iter1(ib + 1));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  int id[]          = {1, 2};
  assert(cuda::std::search(Iter1(ib), Iter1(ib + sb), Iter2(id), Iter2(id + 2), count_equal{count_equal_count})
         == Iter1(ib + 1));
  assert(count_equal_count <= sb * 2);
  count_equal_count = 0;
  int ie[]          = {1, 2, 3};
  assert(cuda::std::search(Iter1(ib), Iter1(ib + sb), Iter2(ie), Iter2(ie + 3), count_equal{count_equal_count})
         == Iter1(ib + 4));
  assert(count_equal_count <= sb * 3);
  count_equal_count = 0;
  int ig[]          = {1, 2, 3, 4};
  assert(cuda::std::search(Iter1(ib), Iter1(ib + sb), Iter2(ig), Iter2(ig + 4), count_equal{count_equal_count})
         == Iter1(ib + 8));
  assert(count_equal_count <= sb * 4);
  count_equal_count = 0;
  int ih[]          = {0, 1, 1, 1, 1, 2, 3, 0, 1, 2, 3, 4};
  const unsigned sh = sizeof(ih) / sizeof(ih[0]);
  int ii[]          = {1, 1, 2};
  assert(cuda::std::search(Iter1(ih), Iter1(ih + sh), Iter2(ii), Iter2(ii + 3), count_equal{count_equal_count})
         == Iter1(ih + 3));
  assert(count_equal_count <= sh * 3);
}

__host__ __device__ constexpr bool test()
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

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
