//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SUPPORT_USER_DEFINED_INTEGRAL_H
#define SUPPORT_USER_DEFINED_INTEGRAL_H

template <class T>
struct UserDefinedIntegral
{
    TEST_HOST_DEVICE constexpr UserDefinedIntegral() : value(0) {}
    TEST_HOST_DEVICE constexpr UserDefinedIntegral(T v) : value(v) {}
    TEST_HOST_DEVICE constexpr operator T() const { return value; }
    T value;
};

// Poison the arithmetic and comparison operations
template <class T, class U>
TEST_HOST_DEVICE constexpr void operator+(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
TEST_HOST_DEVICE constexpr void operator-(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
TEST_HOST_DEVICE constexpr void operator*(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
TEST_HOST_DEVICE constexpr void operator/(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
TEST_HOST_DEVICE constexpr void operator==(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
TEST_HOST_DEVICE constexpr void operator!=(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
TEST_HOST_DEVICE constexpr void operator<(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
TEST_HOST_DEVICE constexpr void operator>(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
TEST_HOST_DEVICE constexpr void operator<=(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
TEST_HOST_DEVICE constexpr void operator>=(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

#endif // SUPPORT_USER_DEFINED_INTEGRAL_H
