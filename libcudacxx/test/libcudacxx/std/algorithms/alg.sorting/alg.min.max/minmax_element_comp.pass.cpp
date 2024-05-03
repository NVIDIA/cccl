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
//   pair<Iter, Iter>
//   minmax_element(Iter first, Iter last, Compare comp);

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

  cuda::std::greater<int> comp{};
  cuda::std::pair<Iter, Iter> p = cuda::std::minmax_element(first, last, comp);
  if (first != last)
  {
    for (Iter j = first; j != last; ++j)
    {
      assert(!comp(*j, *p.first));
      assert(!comp(*p.second, *j));
    }
  }
  else
  {
    assert(p.first == last);
    assert(p.second == last);
  }
}

template <class Iter, class Pred>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_eq(Iter first, Iter last, Pred pred)
{
  cuda::std::pair<Iter, Iter> p = cuda::std::minmax_element(Iter(first), Iter(last), pred);
  assert(base(p.first) == first);
  assert(base(p.second) == last - 1);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 void test_eq()
{
  constexpr int N = 10;
  int a[N]        = {};
  for (int i = 0; i < N; ++i)
  {
    a[i] = 10; // all the same
  }
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
#if TEST_STD_VER >= 2014 && !defined(TEST_COMPILER_MSVC_2017)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014 && !TEST_COMPILER_MSVC_2017

  return 0;
}
