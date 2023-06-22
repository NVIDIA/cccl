//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MOVEONLY_H
#define MOVEONLY_H

#include "test_macros.h"

#if TEST_STD_VER >= 11

#include <cuda/std/cstddef>
// #include <functional>

class MoveOnly
{
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;

    int data_;
public:
    __host__ __device__ TEST_CONSTEXPR_CXX14 MoveOnly(int data = 1) : data_(data) {}
    __host__ __device__ TEST_CONSTEXPR_CXX14 MoveOnly(MoveOnly&& x)
        : data_(x.data_) {x.data_ = 0;}
    __host__ __device__ TEST_CONSTEXPR_CXX14 MoveOnly& operator=(MoveOnly&& x)
        {data_ = x.data_; x.data_ = 0; return *this;}

    __host__ __device__ TEST_CONSTEXPR_CXX14 int get() const {return data_;}

    __host__ __device__ TEST_CONSTEXPR_CXX14 bool operator==(const MoveOnly& x) const {return data_ == x.data_;}
    __host__ __device__ TEST_CONSTEXPR_CXX14 bool operator< (const MoveOnly& x) const {return data_ <  x.data_;}
    __host__ __device__ TEST_CONSTEXPR_CXX14 MoveOnly operator+(const MoveOnly& x) const { return MoveOnly{data_ + x.data_}; }
    __host__ __device__ TEST_CONSTEXPR_CXX14 MoveOnly operator*(const MoveOnly& x) const { return MoveOnly{data_ * x.data_}; }
};

/*
namespace std {
template <>
struct hash<MoveOnly>
{
    typedef MoveOnly argument_type;
    typedef size_t result_type;
    std::size_t operator()(const MoveOnly& x) const {return x.get();}
};
}
*/

#endif  // TEST_STD_VER >= 11

#endif  // MOVEONLY_H
