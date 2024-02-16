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

// template <class U, class... Args>
//     constexpr
//     explicit optional(in_place_t, initializer_list<U> il, Args&&... args);

#include <cuda/std/optional>
#include <cuda/std/type_traits>
#ifdef _LIBCUDACXX_HAS_VECTOR
#include <cuda/std/vector>
#endif
#include <cuda/std/cassert>

#include "test_macros.h"

using cuda::std::optional;
using cuda::std::in_place_t;
using cuda::std::in_place;

class X
{
    int i_;
    int j_ = 0;
public:
    TEST_HOST_DEVICE
    X() : i_(0) {}
    TEST_HOST_DEVICE
    X(int i) : i_(i) {}
    TEST_HOST_DEVICE
    X(int i, int j) : i_(i), j_(j) {}

    TEST_HOST_DEVICE
    ~X() {}

    TEST_HOST_DEVICE
    friend bool operator==(const X& x, const X& y)
        {return x.i_ == y.i_ && x.j_ == y.j_;}
};

class Y
{
    int i_;
    int j_ = 0;
public:
    TEST_HOST_DEVICE
    constexpr Y() : i_(0) {}
    TEST_HOST_DEVICE
    constexpr Y(int i) : i_(i) {}
    TEST_HOST_DEVICE
    constexpr Y(cuda::std::initializer_list<int> il) : i_(il.begin()[0]), j_(il.begin()[1]) {}

    TEST_HOST_DEVICE
    friend constexpr bool operator==(const Y& x, const Y& y)
        {return x.i_ == y.i_ && x.j_ == y.j_;}
};

class Z
{
    int i_;
    int j_ = 0;
public:
    TEST_HOST_DEVICE
    Z() : i_(0) {}
    TEST_HOST_DEVICE
    Z(int i) : i_(i) {}
    TEST_HOST_DEVICE
    Z(cuda::std::initializer_list<int> il) : i_(il.begin()[0]), j_(il.begin()[1])
        {TEST_THROW(6);}

    TEST_HOST_DEVICE
    friend bool operator==(const Z& x, const Z& y)
        {return x.i_ == y.i_ && x.j_ == y.j_;}
};

int main(int, char**)
{
    {
        static_assert(!cuda::std::is_constructible<X, cuda::std::initializer_list<int>&>::value, "");
        static_assert(!cuda::std::is_constructible<optional<X>, cuda::std::initializer_list<int>&>::value, "");
    }
#ifdef _LIBCUDACXX_HAS_VECTOR
    {
        optional<cuda::std::vector<int>> opt(in_place, {3, 1});
        assert(static_cast<bool>(opt) == true);
        assert((*opt == cuda::std::vector<int>{3, 1}));
        assert(opt->size() == 2);
    }
    {
        optional<cuda::std::vector<int>> opt(in_place, {3, 1}, cuda::std::allocator<int>());
        assert(static_cast<bool>(opt) == true);
        assert((*opt == cuda::std::vector<int>{3, 1}));
        assert(opt->size() == 2);
    }
#endif
    {
        static_assert(cuda::std::is_constructible<optional<Y>, cuda::std::initializer_list<int>&>::value, "");

        {
            optional<Y> opt(in_place, {3, 1});
            assert(static_cast<bool>(opt) == true);
            assert((*opt == Y{3, 1}));
        }
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
        {
            constexpr optional<Y> opt(in_place, {3, 1});
            static_assert(static_cast<bool>(opt) == true, "");
            static_assert(*opt == Y{3, 1}, "");
        }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))

        struct test_constexpr_ctor
            : public optional<Y>
        {
            TEST_HOST_DEVICE
            constexpr test_constexpr_ctor(in_place_t, cuda::std::initializer_list<int> i)
                : optional<Y>(in_place, i) {}
        };

    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        static_assert(cuda::std::is_constructible<optional<Z>, cuda::std::initializer_list<int>&>::value, "");
        try
        {
            optional<Z> opt(in_place, {3, 1});
            assert(false);
        }
        catch (int i)
        {
            assert(i == 6);
        }
    }
#endif

  return 0;
}
