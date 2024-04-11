//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// pair(const T1& x, const T2& y);

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

class A
{
  int data_;

public:
  __host__ __device__ A(int data)
      : data_(data)
  {}

  __host__ __device__ bool operator==(const A& a) const
  {
    return data_ == a.data_;
  }
};

int main(int, char**)
{
  {
    typedef cuda::std::pair<float, short*> P;
    P p(3.5f, 0);
    assert(p.first == 3.5f);
    assert(p.second == nullptr);
  }
  {
    typedef cuda::std::pair<A, int> P;
    P p(1, 2);
    assert(p.first == A(1));
    assert(p.second == 2);
  }

  return 0;
}
