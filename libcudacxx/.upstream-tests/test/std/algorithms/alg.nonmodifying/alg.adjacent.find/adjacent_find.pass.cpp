//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter>
//   requires EqualityComparable<Iter::value_type>
//   constexpr Iter  // constexpr after C++17
//   adjacent_find(Iter first, Iter last);

#include <cuda/std/__algorithm>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test() {
    int ia[] = {0, 1, 2, 2, 0, 1, 2, 3};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    assert(cuda::std::adjacent_find(forward_iterator<const int*>(ia),
                                    forward_iterator<const int*>(ia + sa)) ==
                                    forward_iterator<const int*>(ia+2));
    assert(cuda::std::adjacent_find(forward_iterator<const int*>(ia),
                                    forward_iterator<const int*>(ia)) ==
                                    forward_iterator<const int*>(ia));
    assert(cuda::std::adjacent_find(forward_iterator<const int*>(ia+3),
                                    forward_iterator<const int*>(ia + sa)) ==
                                    forward_iterator<const int*>(ia+sa));

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER > 11
    static_assert(test(), "");
#endif

  return 0;
}
