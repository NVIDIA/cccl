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

// template<class ForwardIterator>
// constexpr ForwardIterator
//   shift_left(ForwardIterator first, ForwardIterator last,
//              typename iterator_traits<ForwardIterator>::difference_type n);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T, class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int orig[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
  T work[]   = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};

  for (int n = 0; n <= 15; ++n)
  {
    for (int k = 0; k <= n + 2; ++k)
    {
      cuda::std::copy(orig, orig + n, work);
      Iter it = cuda::std::shift_left(Iter(work), Iter(work + n), k);
      if (0 <= k && k < n)
      {
        assert(it == Iter(work + n - k));
        assert(cuda::std::equal(orig + k, orig + n, work, work + n - k));
      }
      else
      {
        assert(it == Iter(work));
        assert(cuda::std::equal(orig, orig + n, work, work + n));
      }
    }
  }

  // n == 0
  {
    T input[]          = {0, 1, 2};
    const T expected[] = {0, 1, 2};
    Iter b             = Iter(cuda::std::begin(input));
    Iter e             = Iter(cuda::std::end(input));
    Iter it            = cuda::std::shift_left(b, e, 0);
    assert(cuda::std::equal(cuda::std::begin(expected), cuda::std::end(expected), b, e));
    assert(it == e);
  }

  // n > 0 && n < len
  {
    T input[]          = {0, 1, 2};
    const T expected[] = {1, 2};
    Iter b             = Iter(cuda::std::begin(input));
    Iter e             = Iter(cuda::std::end(input));
    Iter it            = cuda::std::shift_left(b, e, 1);
    assert(cuda::std::equal(cuda::std::begin(expected), cuda::std::end(expected), b, it));
  }
  {
    T input[]          = {1, 2, 3, 4, 5, 6, 7, 8};
    const T expected[] = {3, 4, 5, 6, 7, 8};
    Iter b             = Iter(cuda::std::begin(input));
    Iter e             = Iter(cuda::std::end(input));
    Iter it            = cuda::std::shift_left(b, e, 2);
    assert(cuda::std::equal(cuda::std::begin(expected), cuda::std::end(expected), b, it));
  }
  {
    T input[]          = {1, 2, 3, 4, 5, 6, 7, 8};
    const T expected[] = {7, 8};
    Iter b             = Iter(cuda::std::begin(input));
    Iter e             = Iter(cuda::std::end(input));
    Iter it            = cuda::std::shift_left(b, e, 6);
    assert(cuda::std::equal(cuda::std::begin(expected), cuda::std::end(expected), b, it));
  }

  // n == len
  {
    constexpr int len     = 3;
    T input[len]          = {0, 1, 2};
    const T expected[len] = {0, 1, 2};
    Iter b                = Iter(cuda::std::begin(input));
    Iter e                = Iter(cuda::std::end(input));
    Iter it               = cuda::std::shift_left(b, e, len);
    assert(cuda::std::equal(cuda::std::begin(expected), cuda::std::end(expected), b, e));
    assert(it == b);
  }

  // n > len
  {
    constexpr int len     = 3;
    T input[len]          = {0, 1, 2};
    const T expected[len] = {0, 1, 2};
    Iter b                = Iter(cuda::std::begin(input));
    Iter e                = Iter(cuda::std::end(input));
    Iter it               = cuda::std::shift_left(b, e, len + 1);
    assert(cuda::std::equal(cuda::std::begin(expected), cuda::std::end(expected), b, e));
    assert(it == b);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test<int, forward_iterator<int*>>();
  test<int, bidirectional_iterator<int*>>();
  test<int, random_access_iterator<int*>>();
  test<int, int*>();
  test<MoveOnly, forward_iterator<MoveOnly*>>();
  test<MoveOnly, bidirectional_iterator<MoveOnly*>>();
  test<MoveOnly, random_access_iterator<MoveOnly*>>();
  test<MoveOnly, MoveOnly*>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014 && defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014 && _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

  return 0;
}
