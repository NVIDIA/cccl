//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// move_iterator

// move_iterator& operator--();
//
//  constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ void test(It i, It x)
{
  cuda::std::move_iterator<It> r(i);
  cuda::std::move_iterator<It>& rr = --r;
  assert(r.base() == x);
  assert(&rr == &r);
}

int main(int, char**)
{
  char s[] = "123";
  test(bidirectional_iterator<char*>(s + 1), bidirectional_iterator<char*>(s));
  test(random_access_iterator<char*>(s + 1), random_access_iterator<char*>(s));
  test(s + 1, s);

#if TEST_STD_VER > 2011
  {
    constexpr const char* p = "123456789";
    typedef cuda::std::move_iterator<const char*> MI;
    constexpr MI it1 = cuda::std::make_move_iterator(p);
    constexpr MI it2 = cuda::std::make_move_iterator(p + 1);
    static_assert(it1 != it2, "");
    constexpr MI it3 = --cuda::std::make_move_iterator(p + 1);
    static_assert(it1 == it3, "");
    static_assert(it2 != it3, "");
  }
#endif

  return 0;
}
