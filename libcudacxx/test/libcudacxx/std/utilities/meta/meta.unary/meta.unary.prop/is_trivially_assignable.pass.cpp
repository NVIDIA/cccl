//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivially_assignable

// XFAIL: gcc-4.8, gcc-4.9

#include <cuda/std/type_traits>
#include "test_macros.h"

template <class T, class U>
TEST_HOST_DEVICE
void test_is_trivially_assignable()
{
    static_assert(( cuda::std::is_trivially_assignable<T, U>::value), "");
#if TEST_STD_VER > 2011
    static_assert(( cuda::std::is_trivially_assignable_v<T, U>), "");
#endif
}

template <class T, class U>
TEST_HOST_DEVICE
void test_is_not_trivially_assignable()
{
    static_assert((!cuda::std::is_trivially_assignable<T, U>::value), "");
#if TEST_STD_VER > 2011
    static_assert((!cuda::std::is_trivially_assignable_v<T, U>), "");
#endif
}

struct A
{
};

struct B
{
    TEST_HOST_DEVICE
    void operator=(A);
};

struct C
{
    TEST_HOST_DEVICE
    void operator=(C&);  // not const
};

int main(int, char**)
{
    test_is_trivially_assignable<int&, int&> ();
    test_is_trivially_assignable<int&, int> ();
    test_is_trivially_assignable<int&, double> ();

    test_is_not_trivially_assignable<int, int&> ();
    test_is_not_trivially_assignable<int, int> ();
    test_is_not_trivially_assignable<B, A> ();
    test_is_not_trivially_assignable<A, B> ();
    test_is_not_trivially_assignable<C&, C&> ();

  return 0;
}
