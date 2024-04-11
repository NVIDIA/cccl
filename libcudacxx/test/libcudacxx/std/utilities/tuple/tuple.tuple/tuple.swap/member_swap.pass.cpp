//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// void swap(tuple& rhs);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "MoveOnly.h"
#include "test_macros.h"

int main(int, char**)
{
  {
    typedef cuda::std::tuple<> T;
    T t0;
    T t1;
    t0.swap(t1);
  }
  {
    typedef cuda::std::tuple<MoveOnly> T;
    T t0(MoveOnly(0));
    T t1(MoveOnly(1));
    t0.swap(t1);
    assert(cuda::std::get<0>(t0) == 1);
    assert(cuda::std::get<0>(t1) == 0);
  }
  {
    typedef cuda::std::tuple<MoveOnly, MoveOnly> T;
    T t0(MoveOnly(0), MoveOnly(1));
    T t1(MoveOnly(2), MoveOnly(3));
    t0.swap(t1);
    assert(cuda::std::get<0>(t0) == 2);
    assert(cuda::std::get<1>(t0) == 3);
    assert(cuda::std::get<0>(t1) == 0);
    assert(cuda::std::get<1>(t1) == 1);
  }
  {
    typedef cuda::std::tuple<MoveOnly, MoveOnly, MoveOnly> T;
    T t0(MoveOnly(0), MoveOnly(1), MoveOnly(2));
    T t1(MoveOnly(3), MoveOnly(4), MoveOnly(5));
    t0.swap(t1);
    assert(cuda::std::get<0>(t0) == 3);
    assert(cuda::std::get<1>(t0) == 4);
    assert(cuda::std::get<2>(t0) == 5);
    assert(cuda::std::get<0>(t1) == 0);
    assert(cuda::std::get<1>(t1) == 1);
    assert(cuda::std::get<2>(t1) == 2);
  }
  return 0;
}
