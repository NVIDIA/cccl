//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare>
//   requires CopyConstructible<Compare>
//   Iter
//   min_element(Iter first, Iter last, Compare comp);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "cases.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test(const int (&input_data)[num_elements])
{
  Iter first{cuda::std::begin(input_data)};
  Iter last{cuda::std::end(input_data)};

  Iter i = cuda::std::min_element(first, last, cuda::std::greater<int>());
  if (first != last)
  {
    for (Iter j = first; j != last; ++j)
    {
      assert(!cuda::std::greater<int>()(*j, *i));
    }
  }
  else
  {
    assert(i == last);
  }
}

template <class Iter, class Pred>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_eq(Iter first, Iter last, Pred p)
{
  assert(first == cuda::std::min_element(first, last, p));
}

__host__ __device__ TEST_CONSTEXPR_CXX14 void test_eq()
{
  constexpr int N = 10;
  int a[N]        = {};
  for (int i = 0; i < N; ++i)
  {
    a[i] = 10; // all the same
  }
  test_eq(a, a + N, cuda::std::less<int>());
  test_eq(a, a + N, cuda::std::greater<int>());
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  constexpr int input_data[num_elements] = INPUT_DATA;
  test<forward_iterator<const int*>>(input_data);
  test<bidirectional_iterator<const int*>>(input_data);
  test<random_access_iterator<const int*>>(input_data);
  test<const int*>(input_data);
  test_eq();

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
