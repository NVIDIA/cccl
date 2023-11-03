//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter1, ForwardIterator Iter2>
//   requires HasEqualTo<Iter1::value_type, Iter2::value_type>
//   constexpr Iter1  // constexpr after C++17
//   find_first_of(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2);

#include <cuda/std/__algorithm>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test() {
    int ia[] = {0, 1, 2, 3, 0, 1, 2, 3};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    int ib[] = {1, 3, 5, 7};
    const unsigned sb = sizeof(ib)/sizeof(ib[0]);
    assert(cuda::std::find_first_of(cpp17_input_iterator<const int*>(ia),
                                    cpp17_input_iterator<const int*>(ia + sa),
                                    forward_iterator<const int*>(ib),
                                    forward_iterator<const int*>(ib + sb)) ==
                                    cpp17_input_iterator<const int*>(ia+1));
    int ic[] = {7};
    assert(cuda::std::find_first_of(cpp17_input_iterator<const int*>(ia),
                                    cpp17_input_iterator<const int*>(ia + sa),
                                    forward_iterator<const int*>(ic),
                                    forward_iterator<const int*>(ic + 1)) ==
                                    cpp17_input_iterator<const int*>(ia+sa));
    assert(cuda::std::find_first_of(cpp17_input_iterator<const int*>(ia),
                                    cpp17_input_iterator<const int*>(ia + sa),
                                    forward_iterator<const int*>(ic),
                                    forward_iterator<const int*>(ic)) ==
                                    cpp17_input_iterator<const int*>(ia+sa));
    assert(cuda::std::find_first_of(cpp17_input_iterator<const int*>(ia),
                                    cpp17_input_iterator<const int*>(ia),
                                    forward_iterator<const int*>(ic),
                                    forward_iterator<const int*>(ic+1)) ==
                                    cpp17_input_iterator<const int*>(ia));

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
