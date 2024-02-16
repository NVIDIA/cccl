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
//   explicit optional(const optional<U>& rhs);

#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

using cuda::std::optional;

template <class T, class U>
TEST_HOST_DEVICE
TEST_CONSTEXPR_CXX14 void
test(const optional<U>& rhs, bool is_going_to_throw = false)
{
    static_assert(!(cuda::std::is_convertible<const optional<U>&, optional<T>>::value), "");
    bool rhs_engaged = static_cast<bool>(rhs);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        optional<T> lhs(rhs);
        assert(is_going_to_throw == false);
        assert(static_cast<bool>(lhs) == rhs_engaged);
        if (rhs_engaged)
            assert(*lhs == T(*rhs));
    }
    catch (int i)
    {
        assert(i == 6);
    }
#else
    if (is_going_to_throw) return;
    optional<T> lhs(rhs);
    assert(static_cast<bool>(lhs) == rhs_engaged);
    if (rhs_engaged)
        assert(*lhs == T(*rhs));
#endif
}

class X
{
    int i_;
public:
    TEST_HOST_DEVICE
    constexpr explicit X(int i) : i_(i) {}
    TEST_HOST_DEVICE
    constexpr X(const X& x) : i_(x.i_) {}
    TEST_HOST_DEVICE
    TEST_CONSTEXPR_CXX20 ~X() {i_ = 0;}
    TEST_HOST_DEVICE
    friend constexpr bool operator==(const X& x, const X& y) {return x.i_ == y.i_;}
};

class Y
{
    int i_;
public:
    TEST_HOST_DEVICE
    constexpr explicit Y(int i) : i_(i) {}

    TEST_HOST_DEVICE
    friend constexpr bool operator==(const Y& x, const Y& y) {return x.i_ == y.i_;}
};

class Z
{
    int i_;
public:
    TEST_HOST_DEVICE
    explicit Z(int i) : i_(i) {TEST_THROW(6);}

    TEST_HOST_DEVICE
    friend bool operator==(const Z& x, const Z& y) {return x.i_ == y.i_;}
};

template<class T, class U>
TEST_HOST_DEVICE
constexpr bool test_all()
{
  {
    optional<U> rhs;
    test<T>(rhs);
  }
  {
    optional<U> rhs(3);
    test<T>(rhs);
  }
  return true;
}


int main(int, char**)
{
    test_all<X, int>();
    test_all<Y, int>();
#if TEST_STD_VER > 2017 && defined(_LIBCUDACXX_ADDRESSOF)
    static_assert(test_all<X, int>());
    static_assert(test_all<Y, int>());
#endif
    {
        typedef Z T;
        typedef int U;
        optional<U> rhs;
        test<T>(rhs);
    }
    {
        typedef Z T;
        typedef int U;
        optional<U> rhs(3);
        test<T>(rhs, true);
    }

  return 0;
}
