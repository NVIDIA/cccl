//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// void swap(pair& p);

#include <cuda/std/utility>
#include <cuda/std/cassert>

#include "test_macros.h"

struct S {
    int i;
    TEST_HOST_DEVICE S() : i(0) {}
    TEST_HOST_DEVICE S(int j) : i(j) {}
    TEST_HOST_DEVICE S * operator& () { assert(false); return this; }
    TEST_HOST_DEVICE S const * operator& () const { assert(false); return this; }
    TEST_HOST_DEVICE bool operator==(int x) const { return i == x; }
};

int main(int, char**)
{
    {
        typedef cuda::std::pair<int, short> P1;
        P1 p1(3, static_cast<short>(4));
        P1 p2(5, static_cast<short>(6));
        p1.swap(p2);
        assert(p1.first == 5);
        assert(p1.second == 6);
        assert(p2.first == 3);
        assert(p2.second == 4);
    }
    {
        typedef cuda::std::pair<int, S> P1;
        P1 p1(3, S(4));
        P1 p2(5, S(6));
        p1.swap(p2);
        assert(p1.first == 5);
        assert(p1.second == 6);
        assert(p2.first == 3);
        assert(p2.second == 4);
    }

  return 0;
}
