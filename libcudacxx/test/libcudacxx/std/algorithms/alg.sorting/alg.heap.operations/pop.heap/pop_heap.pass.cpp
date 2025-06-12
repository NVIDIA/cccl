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

// template<RandomAccessIterator Iter>
//   requires ShuffleIterator<Iter> && LessThanComparable<Iter::value_type>
//   constexpr void  // constexpr in C++20
//   pop_heap(Iter first, Iter last);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T, class Iter>
__host__ __device__ constexpr void test()
{
  T orig[15] = {9, 6, 9, 5, 5, 8, 9, 1, 1, 3, 5, 3, 4, 7, 2};
  T work[15] = {9, 6, 9, 5, 5, 8, 9, 1, 1, 3, 5, 3, 4, 7, 2};
  assert(cuda::std::is_heap(orig, orig + 15));
  for (int i = 15; i >= 1; --i)
  {
    cuda::std::pop_heap(Iter(work), Iter(work + i));
    assert(cuda::std::is_heap(work, work + i - 1));
    assert(cuda::std::max_element(work, work + i - 1) == work);
    assert(cuda::std::is_permutation(work, work + 15, orig));
  }
  assert(cuda::std::is_sorted(work, work + 15));

  {
    T input[] = {5, 4, 1, 2, 3};
    assert(cuda::std::is_heap(input, input + 5));
    cuda::std::pop_heap(Iter(input), Iter(input + 5));
    assert(input[4] == 5);
    cuda::std::pop_heap(Iter(input), Iter(input + 4));
    assert(input[3] == 4);
    cuda::std::pop_heap(Iter(input), Iter(input + 3));
    assert(input[2] == 3);
    cuda::std::pop_heap(Iter(input), Iter(input + 2));
    assert(input[1] == 2);
    cuda::std::pop_heap(Iter(input), Iter(input + 1));
    assert(input[0] == 1);
  }
}

__host__ __device__ constexpr bool test()
{
  test<int, random_access_iterator<int*>>();
  test<int, int*>();
  test<MoveOnly, random_access_iterator<MoveOnly*>>();
  test<MoveOnly, MoveOnly*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
