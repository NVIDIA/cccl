//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter, OutputIterator<auto, InIter::reference> OutIter>
//   constexpr OutIter   // constexpr after C++17
//   copy(InIter first, InIter last, OutIter result);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct NonTrivialCopy
{
  int data                = 0;
  bool copy_assigned_from = false;

  NonTrivialCopy() = default;

  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialCopy(NonTrivialCopy&& other) noexcept
      : data(other.data)
      , copy_assigned_from(false)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialCopy(const NonTrivialCopy& other) noexcept
      : data(other.data)
      , copy_assigned_from(false)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialCopy& operator=(const NonTrivialCopy& other) noexcept
  {
    data               = other.data;
    copy_assigned_from = true;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialCopy& operator=(NonTrivialCopy&& other) noexcept
  {
    data               = other.data;
    copy_assigned_from = false;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialCopy(const int val) noexcept
      : data(val)
      , copy_assigned_from(false)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialCopy& operator=(const int val) noexcept
  {
    data               = val;
    copy_assigned_from = false;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 friend bool
  operator==(const NonTrivialCopy& lhs, const NonTrivialCopy& rhs) noexcept
  {
    // NOTE: This uses implicit knowledge that the right hand side has been copied from
    return lhs.data == rhs.data && !lhs.copy_assigned_from && rhs.copy_assigned_from;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 bool operator==(const int& other) const noexcept
  {
    // NOTE: This uses implicit knowledge that the only elements we compare against were untouched
    return data == other && !copy_assigned_from;
  }
};

template <class InIter, class OutIter>
TEST_CONSTEXPR_CXX20 __host__ __device__ void test()
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

    OutIter r = cuda::std::copy(InIter(ia), InIter(ia + N), OutIter(ib));
    assert(base(r) == ib + N);
    for (int i = 0; i < N; ++i)
    {
      assert(ia[i] == ib[i]);
    }
  }
  {
    constexpr int N      = 5;
    value_type ia[N]     = {1, 2, 3, 4, 5};
    value_type ic[N + 2] = {6, 6, 6, 6, 6, 6, 6};

    auto p = cuda::std::copy(ia, ia + 5, ic);
    assert(p == (ic + N));
    for (int i = 0; i < N; ++i)
    {
      assert(ia[i] == ic[i]);
    }

    for (int i = N; i < N + 2; ++i)
    {
      assert(ic[i] == 6);
    }
  }
}

TEST_CONSTEXPR_CXX20 __host__ __device__ bool test()
{
  test<cpp17_input_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, int*>();

  test<forward_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<forward_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<forward_iterator<const int*>, forward_iterator<int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<int*>>();
  test<forward_iterator<const int*>, random_access_iterator<int*>>();
  test<forward_iterator<const int*>, int*>();

  test<bidirectional_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<bidirectional_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<int*>>();
  test<bidirectional_iterator<const int*>, int*>();

  test<random_access_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<random_access_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<random_access_iterator<const int*>, forward_iterator<int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<int*>>();
  test<random_access_iterator<const int*>, int*>();

  test<const int*, cpp17_output_iterator<int*>>();
  test<const int*, cpp17_input_iterator<int*>>();
  test<const int*, forward_iterator<int*>>();
  test<const int*, bidirectional_iterator<int*>>();
  test<const int*, random_access_iterator<int*>>();
  test<const int*, int*>();

  test<const NonTrivialCopy*, NonTrivialCopy*>();

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
