//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// template <class T, class... Args>
//   struct is_nothrow_constructible;

#include <cuda/std/type_traits>
#include "test_macros.h"

template <class T>
TEST_HOST_DEVICE
void test_is_nothrow_constructible()
{
    static_assert(( cuda::std::is_nothrow_constructible<T>::value), "");
#if TEST_STD_VER > 2011
    static_assert(( cuda::std::is_nothrow_constructible_v<T>), "");
#endif
}

template <class T, class A0>
TEST_HOST_DEVICE
void test_is_nothrow_constructible()
{
    static_assert(( cuda::std::is_nothrow_constructible<T, A0>::value), "");
#if TEST_STD_VER > 2011
    static_assert(( cuda::std::is_nothrow_constructible_v<T, A0>), "");
#endif
}

template <class T>
TEST_HOST_DEVICE
void test_is_not_nothrow_constructible()
{
    static_assert((!cuda::std::is_nothrow_constructible<T>::value), "");
#if TEST_STD_VER > 2011
    static_assert((!cuda::std::is_nothrow_constructible_v<T>), "");
#endif
}

template <class T, class A0>
TEST_HOST_DEVICE
void test_is_not_nothrow_constructible()
{
    static_assert((!cuda::std::is_nothrow_constructible<T, A0>::value), "");
#if TEST_STD_VER > 2011
    static_assert((!cuda::std::is_nothrow_constructible_v<T, A0>), "");
#endif
}

template <class T, class A0, class A1>
TEST_HOST_DEVICE
void test_is_not_nothrow_constructible()
{
    static_assert((!cuda::std::is_nothrow_constructible<T, A0, A1>::value), "");
#if TEST_STD_VER > 2011
    static_assert((!cuda::std::is_nothrow_constructible_v<T, A0, A1>), "");
#endif
}

class Empty
{
};

class NotEmpty
{
    TEST_HOST_DEVICE
    virtual ~NotEmpty();
};

union Union {};

struct bit_zero
{
    int :  0;
};

class Abstract
{
    TEST_HOST_DEVICE
    virtual ~Abstract() = 0;
};

struct A
{
    TEST_HOST_DEVICE
    A(const A&);
};

struct C
{
    TEST_HOST_DEVICE
    C(C&);  // not const
    TEST_HOST_DEVICE
    void operator=(C&);  // not const
};

struct Tuple {
    TEST_HOST_DEVICE
    Tuple(Empty&&) noexcept {}
};

int main(int, char**)
{
    test_is_nothrow_constructible<int> ();
    test_is_nothrow_constructible<int, const int&> ();
    test_is_nothrow_constructible<Empty> ();
    test_is_nothrow_constructible<Empty, const Empty&> ();

    test_is_not_nothrow_constructible<A, int> ();
    test_is_not_nothrow_constructible<A, int, double> ();
    test_is_not_nothrow_constructible<A> ();
    test_is_not_nothrow_constructible<C> ();
    test_is_nothrow_constructible<Tuple &&, Empty> (); // See bug #19616.

    static_assert(!cuda::std::is_constructible<Tuple&, Empty>::value, "");
    test_is_not_nothrow_constructible<Tuple &, Empty> (); // See bug #19616.

  return 0;
}
