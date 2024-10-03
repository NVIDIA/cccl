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

// template<InputIterator InIter, RandomAccessIterator RAIter, class Compare>
//   requires ShuffleIterator<RAIter>
//         && OutputIterator<RAIter, InIter::reference>
//         && Predicate<Compare, InIter::value_type, RAIter::value_type>
//         && StrictWeakOrder<Compare, RAIter::value_type>}
//         && CopyConstructible<Compare>
//   constexpr RAIter  // constexpr in C++20
//   partial_sort_copy(InIter first, InIter last,
//                     RAIter result_first, RAIter result_last, Compare comp);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T, class Iter, class OutIter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int orig[15] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
  T work[15]   = {};
  for (int n = 0; n < 15; ++n)
  {
    for (int m = 0; m < 15; ++m)
    {
      OutIter it = cuda::std::partial_sort_copy(
        Iter(orig), Iter(orig + n), OutIter(work), OutIter(work + m), cuda::std::greater<T>());
      if (n <= m)
      {
        assert(it == OutIter(work + n));
        assert(cuda::std::is_permutation(OutIter(work), it, orig));
      }
      else
      {
        assert(it == OutIter(work + m));
      }
      assert(cuda::std::is_sorted(OutIter(work), it, cuda::std::greater<T>()));
      if (it != OutIter(work))
      {
        // At most m-1 elements in the input are greater than the biggest element in the result.
        int count = 0;
        for (int i = m; i < n; ++i)
        {
          count += (T(orig[i]) > *(it - 1));
        }
        assert(count < m);
      }
    }
  }

  {
    int input[] = {3, 4, 2, 5, 1};
    T output[]  = {0, 0, 0};
    cuda::std::partial_sort_copy(
      Iter(input), Iter(input + 5), OutIter(output), OutIter(output + 3), cuda::std::greater<T>());
    assert(output[0] == 5);
    assert(output[1] == 4);
    assert(output[2] == 3);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  int i = 42;
  int j = 75;
  cuda::std::partial_sort_copy(&i, &i, &j, &j,
                               cuda::std::greater<int>()); // no-op
  assert(i == 42);
  assert(j == 75);

  test<int, random_access_iterator<int*>, random_access_iterator<int*>>();
  if (!cuda::std::__libcpp_is_constant_evaluated()) // This breaks some compilers due to excessive constant folding
  {
    test<int, random_access_iterator<int*>, int*>();
    test<int, int*, random_access_iterator<int*>>();
    test<int, int*, int*>();
  }

  test<MoveOnly, random_access_iterator<int*>, random_access_iterator<MoveOnly*>>();
  if (!cuda::std::__libcpp_is_constant_evaluated()) // This breaks some compilers due to excessive constant folding
  {
    test<MoveOnly, random_access_iterator<int*>, MoveOnly*>();
    test<MoveOnly, int*, random_access_iterator<MoveOnly*>>();
    test<MoveOnly, int*, MoveOnly*>();
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014 && !defined(TEST_COMPILER_MSVC_2017)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014 && ! TEST_COMPILER_MSVC_2017

  return 0;
}
