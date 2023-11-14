//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, class T>
//   requires OutputIterator<Iter, const T&>
//   constexpr void      // constexpr after C++17
//   fill(Iter first, Iter last, const T& value);

#include <cuda/std/__algorithm>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_char()
{
    const unsigned n = 4;
    char ca[n] = {0};
    cuda::std::fill(Iter(ca), Iter(ca+n), char(1));
    assert(ca[0] == 1);
    assert(ca[1] == 1);
    assert(ca[2] == 1);
    assert(ca[3] == 1);
}

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_int()
{
    const unsigned n = 4;
    int ia[n] = {0};
    cuda::std::fill(Iter(ia), Iter(ia+n), 1);
    assert(ia[0] == 1);
    assert(ia[1] == 1);
    assert(ia[2] == 1);
    assert(ia[3] == 1);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test() {
    test_char<forward_iterator<char*> >();
    test_char<bidirectional_iterator<char*> >();
    test_char<random_access_iterator<char*> >();
    test_char<char*>();

    test_int<forward_iterator<int*> >();
    test_int<bidirectional_iterator<int*> >();
    test_int<random_access_iterator<int*> >();
    test_int<int*>();

    return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 14
  static_assert(test(), "");
#endif // TEST_STD_VER >= 14

  return 0;
}
