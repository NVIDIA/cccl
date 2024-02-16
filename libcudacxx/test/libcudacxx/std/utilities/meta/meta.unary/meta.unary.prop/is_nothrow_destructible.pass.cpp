//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_destructible

// Prevent warning when testing the Abstract test type.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdelete-non-virtual-dtor"
#endif

#include <cuda/std/type_traits>
#include "test_macros.h"

template <class T>
TEST_HOST_DEVICE
void test_is_nothrow_destructible()
{
    static_assert( cuda::std::is_nothrow_destructible<T>::value, "");
    static_assert( cuda::std::is_nothrow_destructible<const T>::value, "");
    static_assert( cuda::std::is_nothrow_destructible<volatile T>::value, "");
    static_assert( cuda::std::is_nothrow_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 2011
    static_assert( cuda::std::is_nothrow_destructible_v<T>, "");
    static_assert( cuda::std::is_nothrow_destructible_v<const T>, "");
    static_assert( cuda::std::is_nothrow_destructible_v<volatile T>, "");
    static_assert( cuda::std::is_nothrow_destructible_v<const volatile T>, "");
#endif
}

template <class T>
TEST_HOST_DEVICE
void test_is_not_nothrow_destructible()
{
    static_assert(!cuda::std::is_nothrow_destructible<T>::value, "");
    static_assert(!cuda::std::is_nothrow_destructible<const T>::value, "");
    static_assert(!cuda::std::is_nothrow_destructible<volatile T>::value, "");
    static_assert(!cuda::std::is_nothrow_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 2011
    static_assert(!cuda::std::is_nothrow_destructible_v<T>, "");
    static_assert(!cuda::std::is_nothrow_destructible_v<const T>, "");
    static_assert(!cuda::std::is_nothrow_destructible_v<volatile T>, "");
    static_assert(!cuda::std::is_nothrow_destructible_v<const volatile T>, "");
#endif
}


struct PublicDestructor           { public:     TEST_HOST_DEVICE ~PublicDestructor() {}};
struct ProtectedDestructor        { protected:  TEST_HOST_DEVICE ~ProtectedDestructor() {}};
struct PrivateDestructor          { private:    TEST_HOST_DEVICE ~PrivateDestructor() {}};

struct VirtualPublicDestructor           { public:    TEST_HOST_DEVICE virtual ~VirtualPublicDestructor() {}};
struct VirtualProtectedDestructor        { protected: TEST_HOST_DEVICE virtual ~VirtualProtectedDestructor() {}};
struct VirtualPrivateDestructor          { private:   TEST_HOST_DEVICE virtual ~VirtualPrivateDestructor() {}};

struct PurePublicDestructor              { public:    TEST_HOST_DEVICE virtual ~PurePublicDestructor() = 0; };
struct PureProtectedDestructor           { protected: TEST_HOST_DEVICE virtual ~PureProtectedDestructor() = 0; };
struct PurePrivateDestructor             { private:   TEST_HOST_DEVICE virtual ~PurePrivateDestructor() = 0; };

class Empty
{
};


union Union {};

struct bit_zero
{
    int :  0;
};

class Abstract
{
    TEST_HOST_DEVICE
    virtual void foo() = 0;
};


int main(int, char**)
{
    test_is_not_nothrow_destructible<void>();
    test_is_not_nothrow_destructible<char[]>();
    test_is_not_nothrow_destructible<char[][3]>();

    test_is_nothrow_destructible<int&>();
    test_is_nothrow_destructible<int>();
    test_is_nothrow_destructible<double>();
    test_is_nothrow_destructible<int*>();
    test_is_nothrow_destructible<const int*>();
    test_is_nothrow_destructible<char[3]>();

    // requires noexcept. These are all destructible.
    test_is_nothrow_destructible<PublicDestructor>();
    test_is_nothrow_destructible<VirtualPublicDestructor>();
    test_is_nothrow_destructible<PurePublicDestructor>();
    test_is_nothrow_destructible<bit_zero>();
    test_is_nothrow_destructible<Abstract>();
    test_is_nothrow_destructible<Empty>();
    test_is_nothrow_destructible<Union>();

    // requires access control
    test_is_not_nothrow_destructible<ProtectedDestructor>();
    test_is_not_nothrow_destructible<PrivateDestructor>();
    test_is_not_nothrow_destructible<VirtualProtectedDestructor>();
    test_is_not_nothrow_destructible<VirtualPrivateDestructor>();
    test_is_not_nothrow_destructible<PureProtectedDestructor>();
    test_is_not_nothrow_destructible<PurePrivateDestructor>();


  return 0;
}
