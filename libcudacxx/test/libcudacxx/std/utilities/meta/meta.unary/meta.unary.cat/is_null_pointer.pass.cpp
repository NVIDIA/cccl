//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_null_pointer

// UNSUPPORTED: c++98, c++03, c++11

#include <cuda/std/type_traits>
#include <cuda/std/cstddef>        // for cuda::std::nullptr_t
#include "test_macros.h"

template <class T>
TEST_HOST_DEVICE
void test_is_null_pointer()
{
    static_assert( cuda::std::is_null_pointer<T>::value, "");
    static_assert( cuda::std::is_null_pointer<const T>::value, "");
    static_assert( cuda::std::is_null_pointer<volatile T>::value, "");
    static_assert( cuda::std::is_null_pointer<const volatile T>::value, "");
#if TEST_STD_VER > 2011
    static_assert( cuda::std::is_null_pointer_v<T>, "");
    static_assert( cuda::std::is_null_pointer_v<const T>, "");
    static_assert( cuda::std::is_null_pointer_v<volatile T>, "");
    static_assert( cuda::std::is_null_pointer_v<const volatile T>, "");
#endif
}

template <class T>
TEST_HOST_DEVICE
void test_is_not_null_pointer()
{
    static_assert(!cuda::std::is_null_pointer<T>::value, "");
    static_assert(!cuda::std::is_null_pointer<const T>::value, "");
    static_assert(!cuda::std::is_null_pointer<volatile T>::value, "");
    static_assert(!cuda::std::is_null_pointer<const volatile T>::value, "");
#if TEST_STD_VER > 2011
    static_assert(!cuda::std::is_null_pointer_v<T>, "");
    static_assert(!cuda::std::is_null_pointer_v<const T>, "");
    static_assert(!cuda::std::is_null_pointer_v<volatile T>, "");
    static_assert(!cuda::std::is_null_pointer_v<const volatile T>, "");
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

enum Enum {zero, one};
struct incomplete_type;

typedef void (*FunctionPtr)();

int main(int, char**)
{
    test_is_null_pointer<cuda::std::nullptr_t>();

    test_is_not_null_pointer<void>();
    test_is_not_null_pointer<int>();
    test_is_not_null_pointer<int&>();
    test_is_not_null_pointer<int&&>();
    test_is_not_null_pointer<int*>();
    test_is_not_null_pointer<double>();
    test_is_not_null_pointer<const int*>();
    test_is_not_null_pointer<char[3]>();
    test_is_not_null_pointer<char[]>();
    test_is_not_null_pointer<Union>();
    test_is_not_null_pointer<Enum>();
    test_is_not_null_pointer<FunctionPtr>();
    test_is_not_null_pointer<Empty>();
    test_is_not_null_pointer<bit_zero>();
    test_is_not_null_pointer<NotEmpty>();
    test_is_not_null_pointer<Abstract>();
    test_is_not_null_pointer<incomplete_type>();

  return 0;
}
