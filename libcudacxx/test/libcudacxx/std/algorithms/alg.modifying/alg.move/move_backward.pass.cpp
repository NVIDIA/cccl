//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<BidirectionalIterator InIter, BidirectionalIterator OutIter>
//   requires OutputIterator<OutIter, RvalueOf<InIter::reference>::type>
//   OutIter
//   move_backward(InIter first, InIter last, OutIter result);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#if defined(_LIBCUDACXX_HAS_MEMORY)
#  include <cuda/std/memory>
#endif // _LIBCUDACXX_HAS_MEMORY

#include "test_iterators.h"
#include "test_macros.h"

struct NonTrivialMove
{
  int data                = 0;
  bool move_assigned_from = false;

  NonTrivialMove() = default;
  __host__ __device__ TEST_CONSTEXPR_CXX14 NonTrivialMove(const NonTrivialMove& other) noexcept
      : data(other.data)
      , move_assigned_from(false)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX14 NonTrivialMove(NonTrivialMove&& other) noexcept
      : data(other.data)
      , move_assigned_from(false)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX14 NonTrivialMove& operator=(const NonTrivialMove& other) noexcept
  {
    data               = other.data;
    move_assigned_from = false;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 NonTrivialMove& operator=(NonTrivialMove&& other) noexcept
  {
    data               = other.data;
    move_assigned_from = true;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 NonTrivialMove(const int val) noexcept
      : data(val)
      , move_assigned_from(false)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX14 NonTrivialMove& operator=(const int val) noexcept
  {
    data               = val;
    move_assigned_from = false;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 friend bool
  operator==(const NonTrivialMove& lhs, const NonTrivialMove& rhs) noexcept
  {
    // NOTE: This uses implicit knowledge that the right hand side has been moved from
    return lhs.data == rhs.data && !lhs.move_assigned_from && rhs.move_assigned_from;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 bool operator==(const int& other) const noexcept
  {
    return data == other;
  }
};

struct NonTrivialDestructor
{
  int data = 0;

  NonTrivialDestructor() = default;

  NonTrivialDestructor(NonTrivialDestructor&&) noexcept                 = default;
  NonTrivialDestructor(const NonTrivialDestructor&) noexcept            = default;
  NonTrivialDestructor& operator=(NonTrivialDestructor&&) noexcept      = default;
  NonTrivialDestructor& operator=(const NonTrivialDestructor&) noexcept = default;
  __host__ __device__ TEST_CONSTEXPR_CXX20 ~NonTrivialDestructor() noexcept {}

  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialDestructor(const int val) noexcept
      : data(val)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialDestructor& operator=(const int val) noexcept
  {
    data = val;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 friend bool
  operator==(const NonTrivialDestructor& lhs, const NonTrivialDestructor& rhs) noexcept
  {
    return lhs.data == rhs.data;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 bool operator==(const int& other) const noexcept
  {
    return data == other;
  }
};

template <class InIter, class OutIter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  using value_type = typename cuda::std::iterator_traits<InIter>::value_type;
  {
    constexpr int N  = 1000;
    value_type ia[N] = {0};
    for (int i = 0; i < N; ++i)
    {
      ia[i] = i;
    }
    value_type ib[N] = {0};

    OutIter r = cuda::std::move_backward(InIter(ia), InIter(ia + N), OutIter(ib + N));
    assert(base(r) == ib);
    for (int i = 0; i < N; ++i)
    {
      assert(ia[i] == ib[i]);
    }
  }
  {
    constexpr int N      = 5;
    value_type ia[N]     = {1, 2, 3, 4, 5};
    value_type ic[N + 2] = {6, 6, 6, 6, 6, 6, 6};

    auto p = cuda::std::move_backward(ia, ia + 5, ic + N);
    assert(p == ic);
    for (int i = 0; i < N; ++i)
    {
      assert(ia[i] == ic[i]);
    }

    for (int i = N; i < N + 2; ++i)
    {
      assert(ic[i] == 6);
    }
  }
  { // Ensure that we are properly perserving back elements when the ranges are overlapping
    constexpr int N      = 5;
    value_type ia[N + 2] = {1, 2, 3, 4, 5, 6, 7};
    int expected[N + 2]  = {1, 2, 1, 2, 3, 4, 5};

    auto p = cuda::std::move_backward(ia, ia + N, ia + N + 2);
    assert(p == ia + 2);
    for (int i = 0; i < N; ++i)
    {
      assert(ia[i] == expected[i]);
    }
  }
}

#if defined(_LIBCUDACXX_HAS_MEMORY)
template <class InIter, class OutIter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test1()
{
  const unsigned N = 100;
  cuda::std::unique_ptr<int> ia[N];
  for (unsigned i = 0; i < N; ++i)
  {
    ia[i].reset(new int(i));
  }
  cuda::std::unique_ptr<int> ib[N];

  OutIter r = cuda::std::move_backward(InIter(ia), InIter(ia + N), OutIter(ib + N));
  assert(base(r) == ib);
  for (unsigned i = 0; i < N; ++i)
  {
    assert(*ib[i] == static_cast<int>(i));
  }
}
#endif // _LIBCUDACXX_HAS_MEMORY

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test<bidirectional_iterator<int*>, bidirectional_iterator<int*>>();
  test<bidirectional_iterator<int*>, random_access_iterator<int*>>();
  test<bidirectional_iterator<int*>, int*>();

  test<random_access_iterator<int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>, random_access_iterator<int*>>();
  test<random_access_iterator<int*>, int*>();

  test<int*, bidirectional_iterator<int*>>();
  test<int*, random_access_iterator<int*>>();
  test<int*, int*>();

  test<NonTrivialMove*, NonTrivialMove*>();
  test<NonTrivialDestructor*, NonTrivialDestructor*>();

#if defined(_LIBCUDACXX_HAS_MEMORY)
  test1<bidirectional_iterator<cuda::std::unique_ptr<int>*>, bidirectional_iterator<cuda::std::unique_ptr<int>*>>();
  test1<bidirectional_iterator<cuda::std::unique_ptr<int>*>, random_access_iterator<cuda::std::unique_ptr<int>*>>();
  test1<bidirectional_iterator<cuda::std::unique_ptr<int>*>, cuda::std::unique_ptr<int>*>();

  test1<random_access_iterator<cuda::std::unique_ptr<int>*>, bidirectional_iterator<cuda::std::unique_ptr<int>*>>();
  test1<random_access_iterator<cuda::std::unique_ptr<int>*>, random_access_iterator<cuda::std::unique_ptr<int>*>>();
  test1<random_access_iterator<cuda::std::unique_ptr<int>*>, cuda::std::unique_ptr<int>*>();

  test1<cuda::std::unique_ptr<int>*, bidirectional_iterator<cuda::std::unique_ptr<int>*>>();
  test1<cuda::std::unique_ptr<int>*, random_access_iterator<cuda::std::unique_ptr<int>*>>();
  test1<cuda::std::unique_ptr<int>*, cuda::std::unique_ptr<int>*>();
#endif // _LIBCUDACXX_HAS_MEMORY

  return true;
}

int main(int, char**)
{
  test();

#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020

  return 0;
}
