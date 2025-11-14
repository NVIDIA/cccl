//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, class T>
//   requires OutputIterator<Iter, Iter::reference>
//         && OutputIterator<Iter, const T&>
//         && HasEqualTo<Iter::value_type, T>
//   constexpr void      // constexpr after C++17
//   replace(Iter first, Iter last, const T& old_value, const T& new_value);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
constexpr __host__ __device__ void test()
{
  constexpr int N           = 5;
  int ia[N]                 = {0, 1, 2, 3, 4};
  constexpr int expected[N] = {0, 1, 5, 3, 4};
  cuda::std::replace(Iter(ia), Iter(ia + N), 2, 5);

  for (int i = 0; i < N; ++i)
  {
    assert(ia[i] == expected[i]);
  }
}

constexpr __host__ __device__ bool test()
{
  test<cpp17_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
