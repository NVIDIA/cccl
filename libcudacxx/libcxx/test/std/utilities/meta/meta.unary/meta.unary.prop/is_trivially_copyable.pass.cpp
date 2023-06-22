//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivially_copyable

// These compilers have not implemented Core 2094 which makes volatile
// qualified types trivially copyable.
// XFAIL: gcc-6, gcc-7, gcc-8, gcc-9

// If we're just building the test and not executing it, it should pass.
// UNSUPPORTED: no_execute

#include <type_traits>
#include <cassert>
#include "test_macros.h"

template <class T>
void test_is_trivially_copyable()
{
    static_assert( std::is_trivially_copyable<T>::value, "");
    static_assert( std::is_trivially_copyable<const T>::value, "");
    static_assert( std::is_trivially_copyable<volatile T>::value, "");
    static_assert( std::is_trivially_copyable<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_trivially_copyable_v<T>, "");
    static_assert( std::is_trivially_copyable_v<const T>, "");
    static_assert( std::is_trivially_copyable_v<volatile T>, "");
    static_assert( std::is_trivially_copyable_v<const volatile T>, "");
#endif
}

template <class T>
void test_is_not_trivially_copyable()
{
    static_assert(!std::is_trivially_copyable<T>::value, "");
    static_assert(!std::is_trivially_copyable<const T>::value, "");
    static_assert(!std::is_trivially_copyable<volatile T>::value, "");
    static_assert(!std::is_trivially_copyable<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!std::is_trivially_copyable_v<T>, "");
    static_assert(!std::is_trivially_copyable_v<const T>, "");
    static_assert(!std::is_trivially_copyable_v<volatile T>, "");
    static_assert(!std::is_trivially_copyable_v<const volatile T>, "");
#endif
}

struct A
{
    int i_;
};

struct B
{
    int i_;
    ~B() {assert(i_ == 0);}
};

class C
{
public:
    C();
};

int main(int, char**)
{
    test_is_trivially_copyable<int> ();
    test_is_trivially_copyable<const int> ();
    test_is_trivially_copyable<A> ();
    test_is_trivially_copyable<const A> ();
    test_is_trivially_copyable<C> ();

    test_is_not_trivially_copyable<int&> ();
    test_is_not_trivially_copyable<const A&> ();
    test_is_not_trivially_copyable<B> ();

  return 0;
}
