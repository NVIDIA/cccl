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

// template<BidirectionalIterator Iter>
//   requires ShuffleIterator<Iter>
//         && LessThanComparable<Iter::value_type>
//   constexpr bool  // constexpr in C++20
//   prev_permutation(Iter first, Iter last);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 int factorial(int x)
{
  int r = 1;
  for (; x; --x)
  {
    r *= x;
  }
  return r;
}

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int ia[]     = {6, 5, 4, 3, 2, 1};
  const int sa = sizeof(ia) / sizeof(ia[0]);
  int prev[sa] = {};
  for (int e = 0; e <= sa; ++e)
  {
    int count = 0;
    bool x    = false;
    do
    {
      cuda::std::copy(ia, ia + e, prev);
      x = cuda::std::prev_permutation(Iter(ia), Iter(ia + e));
      if (e > 1)
      {
        if (x)
        {
          assert(cuda::std::lexicographical_compare(ia, ia + e, prev, prev + e));
        }
        else
        {
          assert(cuda::std::lexicographical_compare(prev, prev + e, ia, ia + e));
        }
      }
      ++count;
    } while (x);
    assert(count == factorial(e));
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<int*>();

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
