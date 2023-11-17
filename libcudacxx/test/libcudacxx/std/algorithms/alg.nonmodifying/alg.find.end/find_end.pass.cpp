//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter1, ForwardIterator Iter2>
//   requires HasEqualTo<Iter1::value_type, Iter2::value_type>
//   constexpr Iter1  // constexpr after C++17
//   find_end(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2);

#include <cuda/std/__algorithm>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter1, class Iter2>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
    int ia[] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    int b[] = {0};
    assert(cuda::std::find_end(Iter1(ia), Iter1(ia+sa), Iter2(b), Iter2(b+1)) == Iter1(ia+sa-1));
    int c[] = {0, 1};
    assert(cuda::std::find_end(Iter1(ia), Iter1(ia+sa), Iter2(c), Iter2(c+2)) == Iter1(ia+18));
    int d[] = {0, 1, 2};
    assert(cuda::std::find_end(Iter1(ia), Iter1(ia+sa), Iter2(d), Iter2(d+3)) == Iter1(ia+15));
    int e[] = {0, 1, 2, 3};
    assert(cuda::std::find_end(Iter1(ia), Iter1(ia+sa), Iter2(e), Iter2(e+4)) == Iter1(ia+11));
    int f[] = {0, 1, 2, 3, 4};
    assert(cuda::std::find_end(Iter1(ia), Iter1(ia+sa), Iter2(f), Iter2(f+5)) == Iter1(ia+6));
    int g[] = {0, 1, 2, 3, 4, 5};
    assert(cuda::std::find_end(Iter1(ia), Iter1(ia+sa), Iter2(g), Iter2(g+6)) == Iter1(ia));
    int h[] = {0, 1, 2, 3, 4, 5, 6};
    assert(cuda::std::find_end(Iter1(ia), Iter1(ia+sa), Iter2(h), Iter2(h+7)) == Iter1(ia+sa));
    assert(cuda::std::find_end(Iter1(ia), Iter1(ia+sa), Iter2(b), Iter2(b)) == Iter1(ia+sa));
    assert(cuda::std::find_end(Iter1(ia), Iter1(ia), Iter2(b), Iter2(b+1)) == Iter1(ia));

    return true;
}

int main(int, char**)
{
    test<forward_iterator<const int*>, forward_iterator<const int*> >();
    test<forward_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<forward_iterator<const int*>, random_access_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, forward_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*> >();
    test<random_access_iterator<const int*>, forward_iterator<const int*> >();
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*>, random_access_iterator<const int*> >();

#if TEST_STD_VER > 11
    static_assert(test<forward_iterator<const int*>, forward_iterator<const int*> >(), "");
    static_assert(test<forward_iterator<const int*>, bidirectional_iterator<const int*> >(), "");
    static_assert(test<forward_iterator<const int*>, random_access_iterator<const int*> >(), "");
    static_assert(test<bidirectional_iterator<const int*>, forward_iterator<const int*> >(), "");
    static_assert(test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*> >(), "");
    static_assert(test<bidirectional_iterator<const int*>, random_access_iterator<const int*> >(), "");
    static_assert(test<random_access_iterator<const int*>, forward_iterator<const int*> >(), "");
    static_assert(test<random_access_iterator<const int*>, bidirectional_iterator<const int*> >(), "");
    static_assert(test<random_access_iterator<const int*>, random_access_iterator<const int*> >(), "");
#endif

  return 0;
}
