//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// reverse_iterator

// template <class U>
//   requires HasAssign<Iter, const U&>
//   constexpr reverse_iterator&
//   operator=(const reverse_iterator<U>& u);
//
//   constexpr in C++17

#include <cuda/std/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It, class U>
__host__ __device__
void
test(U u)
{
    const cuda::std::reverse_iterator<U> r2(u);
    cuda::std::reverse_iterator<It> r1;
    cuda::std::reverse_iterator<It>& rr = r1 = r2;
    assert(r1.base() == u);
    assert(&rr == &r1);
}

struct Base {};
struct Derived : Base {};

int main(int, char**)
{
    Derived d;

    test<bidirectional_iterator<Base*> >(bidirectional_iterator<Derived*>(&d));
    test<random_access_iterator<const Base*> >(random_access_iterator<Derived*>(&d));
    test<Base*>(&d);

#if TEST_STD_VER > 14
    {
        using BaseIter    = cuda::std::reverse_iterator<const Base *>;
        using DerivedIter = cuda::std::reverse_iterator<const Derived *>;
        constexpr const Derived *p = nullptr;
        constexpr DerivedIter     it1 = cuda::std::make_reverse_iterator(p);
        constexpr BaseIter        it2 = (BaseIter{nullptr} = it1);
        static_assert(it2.base() == p, "");
    }
#endif

  return 0;
}
