//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... UTypes>
//   explicit tuple(UTypes&&... u);

// UNSUPPORTED: c++98, c++03

/*
    This is testing an extension whereby only Types having an explicit conversion
    from UTypes are bound by the explicit tuple constructor.
*/

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

class MoveOnly
{
  MoveOnly(const MoveOnly&);
  MoveOnly& operator=(const MoveOnly&);

  int data_;

public:
  __host__ __device__ explicit MoveOnly(int data = 1)
      : data_(data)
  {}
  __host__ __device__ MoveOnly(MoveOnly&& x)
      : data_(x.data_)
  {
    x.data_ = 0;
  }
  __host__ __device__ MoveOnly& operator=(MoveOnly&& x)
  {
    data_   = x.data_;
    x.data_ = 0;
    return *this;
  }

  __host__ __device__ int get() const
  {
    return data_;
  }

  __host__ __device__ bool operator==(const MoveOnly& x) const
  {
    return data_ == x.data_;
  }
  __host__ __device__ bool operator<(const MoveOnly& x) const
  {
    return data_ < x.data_;
  }
};

int main(int, char**)
{
  {
    cuda::std::tuple<MoveOnly> t = 1;
    unused(t);
  }

  return 0;
}
