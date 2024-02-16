//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef MOVEONLY_H
#define MOVEONLY_H

#include "test_macros.h"

#include <cuda/std/cstddef>
// #include <functional>

class MoveOnly
{
    int data_;
public:
    TEST_HOST_DEVICE TEST_CONSTEXPR MoveOnly(int data = 1) : data_(data) {}

    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;

    TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 MoveOnly(MoveOnly&& x)
        : data_(x.data_) {x.data_ = 0;}
    TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 MoveOnly& operator=(MoveOnly&& x)
        {data_ = x.data_; x.data_ = 0; return *this;}

    TEST_HOST_DEVICE TEST_CONSTEXPR int get() const {return data_;}

    TEST_HOST_DEVICE friend TEST_CONSTEXPR bool operator==(const MoveOnly& x, const MoveOnly& y)
        { return x.data_ == y.data_; }
    TEST_HOST_DEVICE friend TEST_CONSTEXPR bool operator!=(const MoveOnly& x, const MoveOnly& y)
        { return x.data_ != y.data_; }
    TEST_HOST_DEVICE friend TEST_CONSTEXPR bool operator< (const MoveOnly& x, const MoveOnly& y)
        { return x.data_ <  y.data_; }
    TEST_HOST_DEVICE friend TEST_CONSTEXPR bool operator<=(const MoveOnly& x, const MoveOnly& y)
        { return x.data_ <= y.data_; }
    TEST_HOST_DEVICE friend TEST_CONSTEXPR bool operator> (const MoveOnly& x, const MoveOnly& y)
        { return x.data_ >  y.data_; }
    TEST_HOST_DEVICE friend TEST_CONSTEXPR bool operator>=(const MoveOnly& x, const MoveOnly& y)
        { return x.data_ >= y.data_; }

#if TEST_STD_VER > 2017 && !defined(TEST_HAS_NO_SPACESHIP_OPERATOR)
    TEST_HOST_DEVICE friend constexpr auto operator<=>(const MoveOnly&, const MoveOnly&) = default;
#endif // TEST_STD_VER > 2017 && !defined(TEST_HAS_NO_SPACESHIP_OPERATOR)

    TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 MoveOnly operator+(const MoveOnly& x) const
        { return MoveOnly(data_ + x.data_); }
    TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 MoveOnly operator*(const MoveOnly& x) const
        { return MoveOnly(data_ * x.data_); }

    template <class T>
    void operator,(T const &) = delete;
};

/*
template <>
struct cuda::std::hash<MoveOnly>
{
    typedef MoveOnly argument_type;
    typedef size_t result_type;
    TEST_HOST_DEVICE TEST_CONSTEXPR size_t operator()(const MoveOnly& x) const {return x.get();}
};
*/

#endif // MOVEONLY_H
