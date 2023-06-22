//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_default_constructible

#include <cuda/std/type_traits>
#include "test_macros.h"

template <class T>
__host__ __device__
void test_is_nothrow_default_constructible()
{
    static_assert( cuda::std::is_nothrow_default_constructible<T>::value, "");
    static_assert( cuda::std::is_nothrow_default_constructible<const T>::value, "");
    static_assert( cuda::std::is_nothrow_default_constructible<volatile T>::value, "");
    static_assert( cuda::std::is_nothrow_default_constructible<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( cuda::std::is_nothrow_default_constructible_v<T>, "");
    static_assert( cuda::std::is_nothrow_default_constructible_v<const T>, "");
    static_assert( cuda::std::is_nothrow_default_constructible_v<volatile T>, "");
    static_assert( cuda::std::is_nothrow_default_constructible_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_has_not_nothrow_default_constructor()
{
    static_assert(!cuda::std::is_nothrow_default_constructible<T>::value, "");
    static_assert(!cuda::std::is_nothrow_default_constructible<const T>::value, "");
    static_assert(!cuda::std::is_nothrow_default_constructible<volatile T>::value, "");
    static_assert(!cuda::std::is_nothrow_default_constructible<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!cuda::std::is_nothrow_default_constructible_v<T>, "");
    static_assert(!cuda::std::is_nothrow_default_constructible_v<const T>, "");
    static_assert(!cuda::std::is_nothrow_default_constructible_v<volatile T>, "");
    static_assert(!cuda::std::is_nothrow_default_constructible_v<const volatile T>, "");
#endif
}

class Empty
{
};

union Union {};

struct bit_zero
{
    int :  0;
};

struct A
{
    __host__ __device__
    A();
};

#if TEST_STD_VER >= 11
struct DThrows
{
    __host__ __device__
    DThrows()  noexcept(true) {}
    __host__ __device__
    ~DThrows() noexcept(false) {}
};
#endif

int main(int, char**)
{
    test_has_not_nothrow_default_constructor<void>();
    test_has_not_nothrow_default_constructor<int&>();
#ifndef TEST_COMPILER_NVHPC
    test_has_not_nothrow_default_constructor<A>();
#if TEST_STD_VER >= 11 && !defined(__INTEL_COMPILER) && (!defined(TEST_COMPILER_C1XX) || 1920 <= _MSC_VER)
    test_has_not_nothrow_default_constructor<DThrows>(); // This is LWG2116
#endif
#endif // TEST_COMPILER_NVHPC

    test_is_nothrow_default_constructible<Union>();
    test_is_nothrow_default_constructible<Empty>();
    test_is_nothrow_default_constructible<int>();
    test_is_nothrow_default_constructible<double>();
    test_is_nothrow_default_constructible<int*>();
    test_is_nothrow_default_constructible<const int*>();
    test_is_nothrow_default_constructible<char[3]>();
    test_is_nothrow_default_constructible<bit_zero>();

  return 0;
}
