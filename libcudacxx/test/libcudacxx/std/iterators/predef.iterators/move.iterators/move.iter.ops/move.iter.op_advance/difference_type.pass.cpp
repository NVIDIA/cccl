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

// requires RandomAccessIterator<Iter>
//   move_iterator& operator+=(difference_type n);
//
//  constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ void test(It i, typename cuda::std::iterator_traits<It>::difference_type n, It x)
{
  cuda::std::move_iterator<It> r(i);
  cuda::std::move_iterator<It>& rr = r += n;
  assert(r.base() == x);
  assert(&rr == &r);
}

int main(int, char**)
{
  const char* s = "1234567890";
  test(random_access_iterator<const char*>(s + 5), 5, random_access_iterator<const char*>(s + 10));
  test(s + 5, 5, s + 10);

#if TEST_STD_VER > 2011
  {
    constexpr const char* p = "123456789";
    typedef cuda::std::move_iterator<const char*> MI;
    constexpr MI it1 = cuda::std::make_move_iterator(p);
    constexpr MI it2 = cuda::std::make_move_iterator(p + 5);
#  ifndef TEST_COMPILER_ICC
    constexpr MI it3 = cuda::std::make_move_iterator(p) += 5;
#  endif // TEST_COMPILER_ICC
    static_assert(it1 != it2, "");
#  ifndef TEST_COMPILER_ICC
    static_assert(it1 != it3, "");
    static_assert(it2 == it3, "");
#  endif // TEST_COMPILER_ICC
  }
#endif

  return 0;
}
