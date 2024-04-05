//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// template <class U>
//   optional(const optional<U>& rhs);

#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

using cuda::std::optional;

template <class T, class U>
__host__ __device__
TEST_CONSTEXPR_CXX14 void test(const optional<U>& rhs)
{
    bool rhs_engaged = static_cast<bool>(rhs);
    optional<T> lhs = rhs;
    assert(static_cast<bool>(lhs) == rhs_engaged);
    if (rhs_engaged) {
        assert(*lhs == *rhs);
    }
}

#ifndef TEST_HAS_NO_EXCEPTIONS
template <class T, class U>
__host__ __device__
void test_exception(const optional<U>& rhs)
{
    try
    {
        optional<T> lhs = rhs;
        unused(lhs);
        assert(false);
    }
    catch (int i)
    {
        assert(i == 6);
    }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

class X
{
    int i_;
public:
    __host__ __device__
    constexpr X(int i) : i_(i) {}
    __host__ __device__
    constexpr X(const X& x) : i_(x.i_) {}
    __host__ __device__
    TEST_CONSTEXPR_CXX20 ~X() {i_ = 0;}
    __host__ __device__
    friend constexpr bool operator==(const X& x, const X& y) {return x.i_ == y.i_;}
};

class Y
{
    int i_;
public:
    __host__ __device__
    constexpr Y(int i) : i_(i) {}

    __host__ __device__
    friend constexpr bool operator==(const Y& x, const Y& y) {return x.i_ == y.i_;}
};

class Z
{
    int i_;
public:
    __host__ __device__
    Z(int i) : i_(i) {TEST_THROW(6);}

    __host__ __device__
    friend bool operator==(const Z& x, const Z& y) {return x.i_ == y.i_;}
};

template<class T, class U>
__host__ __device__
constexpr bool test_all()
{
  {
    optional<U> rhs;
    test<T>(rhs);
  }
  {
    optional<U> rhs(U{3});
    test<T>(rhs);
  }
  return true;
}

int main(int, char**)
{
    test_all<int, short>();
    test_all<X, int>();
    test_all<Y, int>();
#if TEST_STD_VER > 2017 && defined(_LIBCUDACXX_ADDRESSOF)
    static_assert(test_all<int, short>());
    static_assert(test_all<X, int>());
    static_assert(test_all<Y, int>());
#endif
    {
        typedef Z T;
        typedef int U;
        optional<U> rhs;
        test<T>(rhs);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        typedef Z T;
        typedef int U;
        optional<U> rhs(U{3});
        test_exception<T>(rhs);
    }
#endif // !TEST_HAS_NO_EXCEPTIONS

    static_assert(!(cuda::std::is_constructible<optional<X>, const optional<Y>&>::value), "");

  return 0;
}
