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

// template<ForwardIterator Iter, class T, CopyConstructible Compare>
//   constexpr pair<Iter, Iter>   // constexpr after c++17
//   equal_range(Iter first, Iter last, const T& value, Compare comp);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/functional>

#include "../cases.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter, class T>
__host__ __device__ constexpr void test(Iter first, Iter last, const T& value)
{
  cuda::std::pair<Iter, Iter> i = cuda::std::equal_range(first, last, value, cuda::std::less<int>());
  for (Iter j = first; j != i.first; ++j)
  {
    assert(cuda::std::less<int>()(*j, value));
  }
  for (Iter j = i.first; j != last; ++j)
  {
    assert(!cuda::std::less<int>()(*j, value));
  }
  for (Iter j = first; j != i.second; ++j)
  {
    assert(!cuda::std::less<int>()(value, *j));
  }
  for (Iter j = i.second; j != last; ++j)
  {
    assert(cuda::std::less<int>()(value, *j));
  }
}

template <class Iter>
__host__ __device__ constexpr bool test()
{
  constexpr int M = 10;
  auto v          = get_data(M);
  for (int x = 0; x < M; ++x)
  {
    test(Iter(cuda::std::begin(v)), Iter(cuda::std::end(v)), x);
  }
  return true;
}

__host__ __device__ constexpr bool test()
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

  return true;
}

int main(int, char**)
{
  test();
  test<const int*>();
  static_assert(test(), "");
  static_assert(test<const int*>(), ""); // clang otherwise hits the evaluation limit

  return 0;
}
