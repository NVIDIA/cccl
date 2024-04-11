//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class _Iterator>
// struct __bounded_iter;
//
// Comparison operators

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  int array[]                                 = {0, 1, 2, 3, 4};
  int* b                                      = array + 0;
  int* e                                      = array + 5;
  cuda::std::__bounded_iter<Iter> const iter1 = cuda::std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
  cuda::std::__bounded_iter<Iter> const iter2 = cuda::std::__make_bounded_iter(Iter(e), Iter(b), Iter(e));

  // operator==
  {
    assert(iter1 == iter1);
    assert(!(iter1 == iter2));
  }
  // operator!=
  {
    assert(iter1 != iter2);
    assert(!(iter1 != iter1));
  }
  // operator<
  {
    assert(iter1 < iter2);
    assert(!(iter2 < iter1));
    assert(!(iter1 < iter1));
  }
  // operator>
  {
    assert(iter2 > iter1);
    assert(!(iter1 > iter2));
    assert(!(iter1 > iter1));
  }
  // operator<=
  {
    assert(iter1 <= iter2);
    assert(!(iter2 <= iter1));
    assert(iter1 <= iter1);
  }
  // operator>=
  {
    assert(iter2 >= iter1);
    assert(!(iter1 >= iter2));
    assert(iter1 >= iter1);
  }

  return true;
}

int main(int, char**)
{
  tests<int*>();
#if TEST_STD_VER > 2011
  static_assert(tests<int*>(), "");
#endif

#if TEST_STD_VER > 2017
  tests<contiguous_iterator<int*>>();
  static_assert(tests<contiguous_iterator<int*>>(), "");
#endif

  return 0;
}
