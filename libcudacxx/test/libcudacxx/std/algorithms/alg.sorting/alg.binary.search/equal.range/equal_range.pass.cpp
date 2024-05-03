//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, class T>
//   requires HasLess<T, Iter::value_type>
//         && HasLess<Iter::value_type, T>
//   constexpr pair<Iter, Iter>   // constexpr after c++17
//   equal_range(Iter first, Iter last, const T& value);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "../cases.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter, class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test(Iter first, Iter last, const T& value)
{
  cuda::std::pair<Iter, Iter> i = cuda::std::equal_range(first, last, value);
  for (Iter j = first; j != i.first; ++j)
  {
    assert(*j < value);
  }
  for (Iter j = i.first; j != last; ++j)
  {
    assert(!(*j < value));
  }
  for (Iter j = first; j != i.second; ++j)
  {
    assert(!(value < *j));
  }
  for (Iter j = i.second; j != last; ++j)
  {
    assert(value < *j);
  }
}

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  constexpr int M = 10;
  auto v          = get_data(M);
  for (int x = 0; x < M; ++x)
  {
    test(Iter(cuda::std::begin(v)), Iter(cuda::std::end(v)), x);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  int d[] = {0, 1, 2, 3};
  for (int* e = d; e < d + 4; ++e)
  {
    for (int x = -1; x < 4; ++x)
    {
      test(d, e, x);
    }
  }

  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();
  test<const int*>();

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
