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

#include "test_macros.h"

class DefaultOnly
{
  int data_;

  TEST_FUNC DefaultOnly(const DefaultOnly&);
  TEST_FUNC DefaultOnly& operator=(const DefaultOnly&);

public:
  STATIC_MEMBER_VAR(count, int)

  TEST_FUNC DefaultOnly()
      : data_(-1)
  {
    ++count();
  }
  TEST_FUNC ~DefaultOnly()
  {
    data_ = 0;
    --count();
  }

  TEST_FUNC friend bool operator==(const DefaultOnly& x, const DefaultOnly& y)
  {
    return x.data_ == y.data_;
  }
  TEST_FUNC friend bool operator<(const DefaultOnly& x, const DefaultOnly& y)
  {
    return x.data_ < y.data_;
  }
};

#endif // DEFAULTONLY_H
