//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// reverse_iterator

// template <RandomAccessIterator Iter1, RandomAccessIterator Iter2>
//   requires HasMinus<Iter2, Iter1>
// auto operator-(const reverse_iterator<Iter1>& x, const reverse_iterator<Iter2>& y) // constexpr in C++17
//  -> decltype(y.base() - x.base());

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

template <class, class, class = void>
struct HasMinus : cuda::std::false_type
{};
template <class R1, class R2>
struct HasMinus<R1, R2, decltype((R1() - R2(), void()))> : cuda::std::true_type
{};

template <class It1, class It2>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test(It1 l, It2 r, cuda::std::ptrdiff_t x)
{
  const cuda::std::reverse_iterator<It1> r1(l);
  const cuda::std::reverse_iterator<It2> r2(r);
  assert((r1 - r2) == x);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  using PC  = const char*;
  char s[3] = {0};

  // Test same base iterator type
  test(s, s, 0);
  test(s, s + 1, 1);
  test(s + 1, s, -1);

  // Test different (but subtractable) base iterator types
  test(PC(s), s, 0);
  test(PC(s), s + 1, 1);
  test(PC(s + 1), s, -1);

  // Test non-subtractable base iterator types
  static_assert(HasMinus<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<int*>>::value, "");
  static_assert(HasMinus<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<const int*>>::value, "");
  static_assert(!HasMinus<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<char*>>::value, "");
  static_assert(!HasMinus<cuda::std::reverse_iterator<bidirectional_iterator<int*>>,
                          cuda::std::reverse_iterator<bidirectional_iterator<int*>>>::value,
                "");

  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER > 2011
  static_assert(tests(), "");
#endif
  return 0;
}
