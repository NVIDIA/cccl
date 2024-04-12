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
  __host__ __device__ TEST_CONSTEXPR Rep()
      : data_(-1)
  {}
  __host__ __device__ explicit TEST_CONSTEXPR Rep(int i)
      : data_(i)
  {}

  __host__ __device__ bool TEST_CONSTEXPR operator==(int i) const
  {
    return data_ == i;
  }
  __host__ __device__ bool TEST_CONSTEXPR operator==(const Rep& r) const
  {
    return data_ == r.data_;
  }

  __host__ __device__ Rep& operator*=(Rep x)
  {
    data_ *= x.data_;
    return *this;
  }
  __host__ __device__ Rep& operator/=(Rep x)
  {
    data_ /= x.data_;
    return *this;
  }
};

#endif // REP_H
