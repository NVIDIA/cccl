//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: apple-clang-9

// type_traits

// is_trivially_destructible

// Prevent warning when testing the Abstract test type.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdelete-non-virtual-dtor"
#endif

#include <cuda/std/type_traits>
#include "test_macros.h"

template <class T>
TEST_HOST_DEVICE
void test_is_trivially_destructible()
{
    static_assert( cuda::std::is_trivially_destructible<T>::value, "");
    static_assert( cuda::std::is_trivially_destructible<const T>::value, "");
    static_assert( cuda::std::is_trivially_destructible<volatile T>::value, "");
    static_assert( cuda::std::is_trivially_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 2011
    static_assert( cuda::std::is_trivially_destructible_v<T>, "");
    static_assert( cuda::std::is_trivially_destructible_v<const T>, "");
    static_assert( cuda::std::is_trivially_destructible_v<volatile T>, "");
    static_assert( cuda::std::is_trivially_destructible_v<const volatile T>, "");
#endif
}

template <class T>
TEST_HOST_DEVICE
void test_is_not_trivially_destructible()
{
    static_assert(!cuda::std::is_trivially_destructible<T>::value, "");
    static_assert(!cuda::std::is_trivially_destructible<const T>::value, "");
    static_assert(!cuda::std::is_trivially_destructible<volatile T>::value, "");
    static_assert(!cuda::std::is_trivially_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 2011
    static_assert(!cuda::std::is_trivially_destructible_v<T>, "");
    static_assert(!cuda::std::is_trivially_destructible_v<const T>, "");
    static_assert(!cuda::std::is_trivially_destructible_v<volatile T>, "");
    static_assert(!cuda::std::is_trivially_destructible_v<const volatile T>, "");
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

class AbstractDestructor
{
    TEST_HOST_DEVICE
    virtual ~AbstractDestructor() = 0;
};

struct A
{
    TEST_HOST_DEVICE
    ~A();
};

int main(int, char**)
{
    test_is_not_trivially_destructible<void>();
    test_is_not_trivially_destructible<A>();
    test_is_not_trivially_destructible<char[]>();
    test_is_not_trivially_destructible<VirtualPublicDestructor>();
    test_is_not_trivially_destructible<PurePublicDestructor>();

    test_is_trivially_destructible<Abstract>();
    test_is_trivially_destructible<Union>();
    test_is_trivially_destructible<Empty>();
    test_is_trivially_destructible<int&>();
    test_is_trivially_destructible<int>();
    test_is_trivially_destructible<double>();
    test_is_trivially_destructible<int*>();
    test_is_trivially_destructible<const int*>();
    test_is_trivially_destructible<char[3]>();
    test_is_trivially_destructible<bit_zero>();

    // requires access control sfinae
    test_is_not_trivially_destructible<ProtectedDestructor>();
    test_is_not_trivially_destructible<PrivateDestructor>();
    test_is_not_trivially_destructible<VirtualProtectedDestructor>();
    test_is_not_trivially_destructible<VirtualPrivateDestructor>();
    test_is_not_trivially_destructible<PureProtectedDestructor>();
    test_is_not_trivially_destructible<PurePrivateDestructor>();

#if TEST_HAS_BUILTIN_IDENTIFIER(_Atomic)
    test_is_trivially_destructible<_Atomic int>();
    test_is_trivially_destructible<_Atomic float>();
    test_is_trivially_destructible<_Atomic int*>();
#endif

  return 0;
}
