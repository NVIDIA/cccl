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
//
//   template<class ForwardIterator, class Searcher>
//   ForwardIterator search(ForwardIterator first, ForwardIterator last,
//                          const Searcher& searcher); // C++17

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

#ifdef TEST_COMPILER_MSVC
#  pragma warning(disable : 4018) // signed/unsigned mismatch
#endif // TEST_COMPILER_MSVC

#ifdef TEST_COMPILER_GCC
#  pragma GCC diagnostic ignored "-Wsign-compare"
#endif // TEST_COMPILER_GCC

#ifdef TEST_COMPILER_CLANG
#  pragma clang diagnostic ignored "-Wsign-compare"
#endif // TEST_COMPILER_CLANG

struct MySearcherC
{
  template <typename Iterator>
  __host__ __device__ cuda::std::pair<Iterator, Iterator> TEST_CONSTEXPR_CXX14 operator()(Iterator b, Iterator e) const
  {
    return cuda::std::make_pair(b, e);
  }
};

struct MySearcher
{
  __host__ __device__ TEST_CONSTEXPR_CXX14 MySearcher(int& searcher_called) noexcept
      : searcher_called(searcher_called)
  {}

  template <typename Iterator>
  __host__ __device__ cuda::std::pair<Iterator, Iterator> TEST_CONSTEXPR_CXX14 operator()(Iterator b, Iterator e) const
  {
    ++searcher_called;
    return cuda::std::make_pair(b, e);
  }

  int& searcher_called;
};

namespace User
{
struct S
{
  __host__ __device__ constexpr S(int x)
      : x_(x)
  {}
  __host__ __device__ friend constexpr bool operator==(S lhs, S rhs) noexcept
  {
    return lhs.x_ == rhs.x_;
  }
  int x_;
};

template <class T, class U>
void make_pair(T&&, U&&) = delete;
} // namespace User

template <class Iter1, class Iter2>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int ia[]          = {0, 1, 2, 3, 4, 5};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia), Iter2(ia)) == Iter1(ia));
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia), Iter2(ia + 1)) == Iter1(ia));
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 1), Iter2(ia + 2)) == Iter1(ia + 1));
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 2), Iter2(ia + 2)) == Iter1(ia));
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 2), Iter2(ia + 3)) == Iter1(ia + 2));
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 2), Iter2(ia + 3)) == Iter1(ia + 2));
  assert(cuda::std::search(Iter1(ia), Iter1(ia), Iter2(ia + 2), Iter2(ia + 3)) == Iter1(ia));
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia + sa - 1), Iter2(ia + sa)) == Iter1(ia + sa - 1));
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia + sa - 3), Iter2(ia + sa)) == Iter1(ia + sa - 3));
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa), Iter2(ia), Iter2(ia + sa)) == Iter1(ia));
  assert(cuda::std::search(Iter1(ia), Iter1(ia + sa - 1), Iter2(ia), Iter2(ia + sa)) == Iter1(ia + sa - 1));
  assert(cuda::std::search(Iter1(ia), Iter1(ia + 1), Iter2(ia), Iter2(ia + sa)) == Iter1(ia + 1));
  int ib[]          = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4};
  const unsigned sb = sizeof(ib) / sizeof(ib[0]);
  int ic[]          = {1};
  assert(cuda::std::search(Iter1(ib), Iter1(ib + sb), Iter2(ic), Iter2(ic + 1)) == Iter1(ib + 1));
  int id[] = {1, 2};
  assert(cuda::std::search(Iter1(ib), Iter1(ib + sb), Iter2(id), Iter2(id + 2)) == Iter1(ib + 1));
  int ie[] = {1, 2, 3};
  assert(cuda::std::search(Iter1(ib), Iter1(ib + sb), Iter2(ie), Iter2(ie + 3)) == Iter1(ib + 4));
  int ig[] = {1, 2, 3, 4};
  assert(cuda::std::search(Iter1(ib), Iter1(ib + sb), Iter2(ig), Iter2(ig + 4)) == Iter1(ib + 8));
  int ih[]          = {0, 1, 1, 1, 1, 2, 3, 0, 1, 2, 3, 4};
  const unsigned sh = sizeof(ih) / sizeof(ih[0]);
  int ii[]          = {1, 1, 2};
  assert(cuda::std::search(Iter1(ih), Iter1(ih + sh), Iter2(ii), Iter2(ii + 3)) == Iter1(ih + 3));
  int ij[]          = {0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0};
  const unsigned sj = sizeof(ij) / sizeof(ij[0]);
  int ik[]          = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0};
  const unsigned sk = sizeof(ik) / sizeof(ik[0]);
  assert(cuda::std::search(Iter1(ij), Iter1(ij + sj), Iter2(ik), Iter2(ik + sk)) == Iter1(ij + 6));
}

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void adl_test()
{
  User::S ua[] = {1};
  assert(cuda::std::search(Iter(ua), Iter(ua), Iter(ua), Iter(ua)) == Iter(ua));
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
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

  adl_test<forward_iterator<User::S*>>();
  adl_test<random_access_iterator<User::S*>>();

  return true;
}

int main(int, char**)
{
  test();

#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014

#if TEST_STD_VER >= 2017
  {
    int searcher_called = 0;
    typedef int* RI;
    static_assert((cuda::std::is_same<RI, decltype(cuda::std::search(RI(), RI(), MySearcher{searcher_called}))>::value),
                  "");

    RI it(nullptr);
    assert(it == cuda::std::search(it, it, MySearcher{searcher_called}));
    assert(searcher_called == 1);
  }
#endif // TEST_STD_VER >= 2017

  return 0;
}
