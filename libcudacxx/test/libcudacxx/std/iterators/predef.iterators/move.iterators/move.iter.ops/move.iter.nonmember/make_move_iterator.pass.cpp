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

// template <InputIterator Iter>
//   move_iterator<Iter>
//   make_move_iterator(const Iter& i);
//
//  constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ void test(It i)
{
  const cuda::std::move_iterator<It> r(i);
  assert(cuda::std::make_move_iterator(i) == r);
}

int main(int, char**)
{
  {
    char s[] = "1234567890";
    test(cpp17_input_iterator<char*>(s + 5));
    test(forward_iterator<char*>(s + 5));
    test(bidirectional_iterator<char*>(s + 5));
    test(random_access_iterator<char*>(s + 5));
    test(s + 5);
  }
  {
    int a[] = {1, 2, 3, 4};
    TEST_IGNORE_NODISCARD cuda::std::make_move_iterator(a + 4);
    TEST_IGNORE_NODISCARD cuda::std::make_move_iterator(a); // test for LWG issue 2061
  }

#if TEST_STD_VER > 2011
  {
    constexpr const char* p = "123456789";
    constexpr auto iter     = cuda::std::make_move_iterator<const char*>(p);
    static_assert(iter.base() == p, "");
  }
#endif

  return 0;
}
