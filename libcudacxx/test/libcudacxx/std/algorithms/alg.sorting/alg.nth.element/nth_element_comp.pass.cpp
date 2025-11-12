//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: true

// <algorithm>

// template<RandomAccessIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare>
//   requires ShuffleIterator<Iter> && CopyConstructible<Compare>
//   constexpr void  // constexpr in C++20
//   nth_element(Iter first, Iter nth, Iter last, Compare comp);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T, class Iter>
__host__ __device__ constexpr void test()
{
  int orig[15] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
  T work[15]   = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
  for (int n = 0; n < 15; ++n)
  {
    for (int m = 0; m < n; ++m)
    {
      cuda::std::nth_element(Iter(work), Iter(work + m), Iter(work + n), cuda::std::greater<T>());
      assert(cuda::std::is_permutation(work, work + n, orig));
      // No element to m's left is less than m.
      for (int i = 0; i < m; ++i)
      {
        assert(!(work[i] < work[m]));
      }
      // No element to m's right is greater than m.
      for (int i = m; i < n; ++i)
      {
        assert(!(work[i] > work[m]));
      }
      cuda::std::copy(orig, orig + 15, work);
    }
  }

  {
    T input[] = {3, 1, 4, 1, 5, 9, 2};
    cuda::std::nth_element(Iter(input), Iter(input + 4), Iter(input + 7), cuda::std::greater<T>());
    assert(input[4] == 2);
    assert(input[5] + input[6] == 1 + 1);
  }

  {
    T input[] = {0, 1, 2, 3, 4, 5, 7, 6};
    cuda::std::nth_element(Iter(input), Iter(input + 6), Iter(input + 8), cuda::std::greater<T>());
    assert(input[6] == 1);
    assert(input[7] == 0);
  }

  {
    T input[] = {1, 0, 2, 3, 4, 5, 6, 7};
    cuda::std::nth_element(Iter(input), Iter(input + 1), Iter(input + 8), cuda::std::greater<T>());
    assert(input[0] == 7);
    assert(input[1] == 6);
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
#ifdef _CCCL_BUILTIN_IS_CONSTANT_EVALUATED
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

  return 0;
}
