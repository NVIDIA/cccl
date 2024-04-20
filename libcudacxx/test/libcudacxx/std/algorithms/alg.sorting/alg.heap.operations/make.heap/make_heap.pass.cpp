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
//   make_heap(Iter first, Iter last);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T, class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int orig[15] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
  T work[15]   = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
  for (int n = 0; n < 15; ++n)
  {
    cuda::std::make_heap(Iter(work), Iter(work + n));
    assert(cuda::std::is_heap(work, work + n));
    assert(cuda::std::is_permutation(work, work + n, orig));
    cuda::std::copy(orig, orig + n, work);
  }

  {
    T input[] = {3, 4, 1, 2, 5};
    cuda::std::make_heap(Iter(input), Iter(input + 5));
    assert(cuda::std::is_heap(input, input + 5));
    cuda::std::pop_heap(input, input + 5);
    assert(input[4] == 5);
    cuda::std::pop_heap(input, input + 4);
    assert(input[3] == 4);
    cuda::std::pop_heap(input, input + 3);
    assert(input[2] == 3);
    cuda::std::pop_heap(input, input + 2);
    assert(input[1] == 2);
    cuda::std::pop_heap(input, input + 1);
    assert(input[0] == 1);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
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
#if TEST_STD_VER >= 2014 && defined(_LIBCUDACXX_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014 && _LIBCUDACXX_IS_CONSTANT_EVALUATED

  return 0;
}
