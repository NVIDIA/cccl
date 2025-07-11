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

  __host__ __device__ DefaultOnly(const DefaultOnly&);
  __host__ __device__ DefaultOnly& operator=(const DefaultOnly&);

public:
  STATIC_MEMBER_VAR(count, int)

  __host__ __device__ DefaultOnly()
      : data_(-1)
  {
    ++count();
  }
  __host__ __device__ ~DefaultOnly()
  {
    data_ = 0;
    --count();
  }

  __host__ __device__ friend bool operator==(const DefaultOnly& x, const DefaultOnly& y)
  {
    return x.data_ == y.data_;
  }
  __host__ __device__ friend bool operator<(const DefaultOnly& x, const DefaultOnly& y)
  {
    return x.data_ < y.data_;
  }
};

#endif // DEFAULTONLY_H
