//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_floating_point

#include <cuda/std/type_traits>
#include <cuda/std/cstddef>        // for cuda::std::nullptr_t
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(cuda_demote_unsupported_floating_point)

template <class T>
__host__ __device__
void test_is_floating_point()
{
    static_assert( cuda::std::is_floating_point<T>::value, "");
    static_assert( cuda::std::is_floating_point<const T>::value, "");
    static_assert( cuda::std::is_floating_point<volatile T>::value, "");
    static_assert( cuda::std::is_floating_point<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( cuda::std::is_floating_point_v<T>, "");
    static_assert( cuda::std::is_floating_point_v<const T>, "");
    static_assert( cuda::std::is_floating_point_v<volatile T>, "");
    static_assert( cuda::std::is_floating_point_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_is_not_floating_point()
{
    static_assert(!cuda::std::is_floating_point<T>::value, "");
    static_assert(!cuda::std::is_floating_point<const T>::value, "");
    static_assert(!cuda::std::is_floating_point<volatile T>::value, "");
    static_assert(!cuda::std::is_floating_point<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!cuda::std::is_floating_point_v<T>, "");
    static_assert(!cuda::std::is_floating_point_v<const T>, "");
    static_assert(!cuda::std::is_floating_point_v<volatile T>, "");
    static_assert(!cuda::std::is_floating_point_v<const volatile T>, "");
#endif
}

class Empty
{
};

class NotEmpty
{
    __host__ __device__
    virtual ~NotEmpty();
};

union Union {};

struct bit_zero
{
    int :  0;
};

class Abstract
{
    __host__ __device__
    virtual ~Abstract() = 0;
};

enum Enum {zero, one};
struct incomplete_type;

typedef void (*FunctionPtr)();

int main(int, char**)
{
    test_is_floating_point<float>();
    test_is_floating_point<double>();
    test_is_floating_point<long double>();

    test_is_not_floating_point<short>();
    test_is_not_floating_point<unsigned short>();
    test_is_not_floating_point<int>();
    test_is_not_floating_point<unsigned int>();
    test_is_not_floating_point<long>();
    test_is_not_floating_point<unsigned long>();

    test_is_not_floating_point<cuda::std::nullptr_t>();
    test_is_not_floating_point<void>();
    test_is_not_floating_point<int&>();
    test_is_not_floating_point<int&&>();
    test_is_not_floating_point<int*>();
    test_is_not_floating_point<const int*>();
    test_is_not_floating_point<char[3]>();
    test_is_not_floating_point<char[]>();
    test_is_not_floating_point<Union>();
    test_is_not_floating_point<Empty>();
    test_is_not_floating_point<bit_zero>();
    test_is_not_floating_point<NotEmpty>();
    test_is_not_floating_point<Abstract>();
    test_is_not_floating_point<Enum>();
    test_is_not_floating_point<FunctionPtr>();
    test_is_not_floating_point<incomplete_type>();

  return 0;
}
