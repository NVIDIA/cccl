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
//   tuple& operator=(const tuple<UTypes...>& u);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

struct B
{
  int id_;

  __host__ __device__ explicit B(int i = 0)
      : id_(i)
  {}
};

struct D : B
{
  __host__ __device__ explicit D(int i = 0)
      : B(i)
  {}
};

int main(int, char**)
{
  {
    typedef cuda::std::tuple<long> T0;
    typedef cuda::std::tuple<long long> T1;
    T0 t0(2);
    T1 t1;
    t1 = t0;
    assert(cuda::std::get<0>(t1) == 2);
  }
  {
    typedef cuda::std::tuple<long, char> T0;
    typedef cuda::std::tuple<long long, int> T1;
    T0 t0(2, 'a');
    T1 t1;
    t1 = t0;
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
  }
  {
    typedef cuda::std::tuple<long, char, D> T0;
    typedef cuda::std::tuple<long long, int, B> T1;
    T0 t0(2, 'a', D(3));
    T1 t1;
    t1 = t0;
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
    assert(cuda::std::get<2>(t1).id_ == 3);
  }
  {
    D d(3);
    D d2(2);
    typedef cuda::std::tuple<long, char, D&> T0;
    typedef cuda::std::tuple<long long, int, B&> T1;
    T0 t0(2, 'a', d2);
    T1 t1(1, 'b', d);
    t1 = t0;
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
    assert(cuda::std::get<2>(t1).id_ == 2);
  }
  {
    // Test that tuple evaluates correctly applies an lvalue reference
    // before evaluating is_assignable (ie 'is_assignable<int&, int&>')
    // instead of evaluating 'is_assignable<int&&, int&>' which is false.
    int x = 42;
    int y = 43;
    cuda::std::tuple<int&&> t(cuda::std::move(x));
    cuda::std::tuple<int&> t2(y);
    t = t2;
    assert(cuda::std::get<0>(t) == 43);
    assert(&cuda::std::get<0>(t) == &x);
  }
  return 0;
}
