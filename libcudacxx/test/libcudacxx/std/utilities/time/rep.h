//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef REP_H
#define REP_H

#include "test_macros.h"

class Rep
{
    int data_;
public:
    TEST_HOST_DEVICE
    TEST_CONSTEXPR Rep() : data_(-1) {}
    TEST_HOST_DEVICE
    explicit TEST_CONSTEXPR Rep(int i) : data_(i) {}

    TEST_HOST_DEVICE
    bool TEST_CONSTEXPR operator==(int i) const {return data_ == i;}
    TEST_HOST_DEVICE
    bool TEST_CONSTEXPR operator==(const Rep& r) const {return data_ == r.data_;}

    TEST_HOST_DEVICE
    Rep& operator*=(Rep x) {data_ *= x.data_; return *this;}
    TEST_HOST_DEVICE
    Rep& operator/=(Rep x) {data_ /= x.data_; return *this;}
};

#endif  // REP_H
