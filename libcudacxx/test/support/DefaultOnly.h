//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DEFAULTONLY_H
#define DEFAULTONLY_H

#include <cuda/std/cassert>

class DefaultOnly
{
    int data_;

    TEST_HOST_DEVICE DefaultOnly(const DefaultOnly&);
    TEST_HOST_DEVICE DefaultOnly& operator=(const DefaultOnly&);
public:
    STATIC_MEMBER_VAR(count, int);

    TEST_HOST_DEVICE DefaultOnly() : data_(-1) {++count();}
    TEST_HOST_DEVICE ~DefaultOnly() {data_ = 0; --count();}

    TEST_HOST_DEVICE friend bool operator==(const DefaultOnly& x, const DefaultOnly& y)
        {return x.data_ == y.data_;}
    TEST_HOST_DEVICE friend bool operator< (const DefaultOnly& x, const DefaultOnly& y)
        {return x.data_ < y.data_;}
};

#endif  // DEFAULTONLY_H
