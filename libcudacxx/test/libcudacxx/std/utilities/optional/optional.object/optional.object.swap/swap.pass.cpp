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

// void swap(optional&)
//     noexcept(is_nothrow_move_constructible<T>::value &&
//              is_nothrow_swappable<T>::value)

#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "archetypes.h"

using cuda::std::optional;

class X
{
    int i_;
public:
    STATIC_MEMBER_VAR(dtor_called, unsigned);
    __host__ __device__
    X(int i) : i_(i) {}
    X(X&& x) = default;
    X& operator=(X&&) = default;
    __host__ __device__
    ~X() {++dtor_called();}

    __host__ __device__
    friend bool operator==(const X& x, const X& y) {return x.i_ == y.i_;}
};

class Y
{
    int i_;
public:
    STATIC_MEMBER_VAR(dtor_called, unsigned);
    __host__ __device__
    Y(int i) : i_(i) {}
    Y(Y&&) = default;
    __host__ __device__
    ~Y() {++dtor_called();}

    __host__ __device__
    friend constexpr bool operator==(const Y& x, const Y& y) {return x.i_ == y.i_;}
    __host__ __device__
    friend void swap(Y& x, Y& y) {cuda::std::swap(x.i_, y.i_);}
};

class Z
{
    int i_;
public:
    __host__ __device__
    Z(int i) : i_(i) {}
    __host__ __device__
    Z(Z&&) {TEST_THROW(7);}

    __host__ __device__
    friend constexpr bool operator==(const Z& x, const Z& y) {return x.i_ == y.i_;}
    __host__ __device__
    friend void swap(Z&, Z&) {TEST_THROW(6);}
};

class W
{
    int i_;
public:
    __host__ __device__
    constexpr W(int i) : i_(i) {}

    __host__ __device__
    friend constexpr bool operator==(const W& x, const W& y) {return x.i_ == y.i_;}
    __host__ __device__
    friend TEST_CONSTEXPR_CXX14 void swap(W& x, W& y) noexcept {cuda::std::swap(x.i_, y.i_);}
};

template<class T>
__host__ __device__
TEST_CONSTEXPR_CXX14 bool check_swap()
{
    {
        optional<T> opt1;
        optional<T> opt2;
        static_assert(noexcept(opt1.swap(opt2)) == true, "");
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == false);
        opt1.swap(opt2);
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == false);
    }
    {
        optional<T> opt1(1);
        optional<T> opt2;
        static_assert(noexcept(opt1.swap(opt2)) == true, "");
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 1);
        assert(static_cast<bool>(opt2) == false);
        opt1.swap(opt2);
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 1);
    }
    {
        optional<T> opt1;
        optional<T> opt2(2);
        static_assert(noexcept(opt1.swap(opt2)) == true, "");
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 2);
        opt1.swap(opt2);
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 2);
        assert(static_cast<bool>(opt2) == false);
    }
    {
        optional<T> opt1(1);
        optional<T> opt2(2);
        static_assert(noexcept(opt1.swap(opt2)) == true, "");
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 1);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 2);
        opt1.swap(opt2);
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 2);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 1);
    }
    return true;
}

int main(int, char**)
{
    check_swap<int>();
    check_swap<W>();
#if TEST_STD_VER > 17 && defined(_LIBCUDACXX_ADDRESSOF)
    static_assert(check_swap<int>());
    static_assert(check_swap<W>());
#endif
    {
        optional<X> opt1;
        optional<X> opt2;
        static_assert(noexcept(opt1.swap(opt2)) == true, "");
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == false);
        opt1.swap(opt2);
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == false);
        assert(X::dtor_called() == 0);
    }
    {
        optional<X> opt1(1);
        optional<X> opt2;
        static_assert(noexcept(opt1.swap(opt2)) == true, "");
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 1);
        assert(static_cast<bool>(opt2) == false);
        X::dtor_called() = 0;
        opt1.swap(opt2);
        assert(X::dtor_called() == 1);
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 1);
    }
    {
        optional<X> opt1;
        optional<X> opt2(2);
        static_assert(noexcept(opt1.swap(opt2)) == true, "");
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 2);
        X::dtor_called() = 0;
        opt1.swap(opt2);
        assert(X::dtor_called() == 1);
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 2);
        assert(static_cast<bool>(opt2) == false);
    }
    {
        optional<X> opt1(1);
        optional<X> opt2(2);
        static_assert(noexcept(opt1.swap(opt2)) == true, "");
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 1);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 2);
        X::dtor_called() = 0;
        opt1.swap(opt2);
        assert(X::dtor_called() == 1);  // from inside cuda::std::swap
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 2);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 1);
    }
    {
        optional<Y> opt1;
        optional<Y> opt2;
#ifndef TEST_COMPILER_ICC
        static_assert(noexcept(opt1.swap(opt2)) == false, "");
#endif // TEST_COMPILER_ICC
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == false);
        opt1.swap(opt2);
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == false);
        assert(Y::dtor_called() == 0);
    }
    {
        optional<Y> opt1(1);
        optional<Y> opt2;
#ifndef TEST_COMPILER_ICC
        static_assert(noexcept(opt1.swap(opt2)) == false, "");
#endif // TEST_COMPILER_ICC
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 1);
        assert(static_cast<bool>(opt2) == false);
        Y::dtor_called() = 0;
        opt1.swap(opt2);
        assert(Y::dtor_called() == 1);
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 1);
    }
    {
        optional<Y> opt1;
        optional<Y> opt2(2);
#ifndef TEST_COMPILER_ICC
        static_assert(noexcept(opt1.swap(opt2)) == false, "");
#endif // TEST_COMPILER_ICC
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 2);
        Y::dtor_called() = 0;
        opt1.swap(opt2);
        assert(Y::dtor_called() == 1);
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 2);
        assert(static_cast<bool>(opt2) == false);
    }
    {
        optional<Y> opt1(1);
        optional<Y> opt2(2);
#ifndef TEST_COMPILER_ICC
        static_assert(noexcept(opt1.swap(opt2)) == false, "");
#endif // TEST_COMPILER_ICC
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 1);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 2);
        Y::dtor_called() = 0;
        opt1.swap(opt2);
        assert(Y::dtor_called() == 0);
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 2);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 1);
    }
    {
        optional<Z> opt1;
        optional<Z> opt2;
#ifndef TEST_COMPILER_ICC
        static_assert(noexcept(opt1.swap(opt2)) == false, "");
#endif // TEST_COMPILER_ICC
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == false);
        opt1.swap(opt2);
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == false);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        optional<Z> opt1;
        opt1.emplace(1);
        optional<Z> opt2;
        static_assert(noexcept(opt1.swap(opt2)) == false, "");
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 1);
        assert(static_cast<bool>(opt2) == false);
        try
        {
            opt1.swap(opt2);
            assert(false);
        }
        catch (int i)
        {
            assert(i == 7);
        }
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 1);
        assert(static_cast<bool>(opt2) == false);
    }
    {
        optional<Z> opt1;
        optional<Z> opt2;
        opt2.emplace(2);
        static_assert(noexcept(opt1.swap(opt2)) == false, "");
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 2);
        try
        {
            opt1.swap(opt2);
            assert(false);
        }
        catch (int i)
        {
            assert(i == 7);
        }
        assert(static_cast<bool>(opt1) == false);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 2);
    }
    {
        optional<Z> opt1;
        opt1.emplace(1);
        optional<Z> opt2;
        opt2.emplace(2);
        static_assert(noexcept(opt1.swap(opt2)) == false, "");
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 1);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 2);
        try
        {
            opt1.swap(opt2);
            assert(false);
        }
        catch (int i)
        {
            assert(i == 6);
        }
        assert(static_cast<bool>(opt1) == true);
        assert(*opt1 == 1);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 2);
    }
#endif

  return 0;
}
