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
//   tuple& operator=(tuple<UTypes...>&& u);

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"

struct B
{
  int id_;

  __host__ __device__ explicit B(int i = 0)
      : id_(i)
  {}

  __host__ __device__ virtual ~B() {}
};

struct D : B
{
  __host__ __device__ explicit D(int i)
      : B(i)
  {}
};

struct E
{
  E() = default;
  __host__ __device__ E& operator=(int)
  {
    return *this;
  }
};

int main(int, char**)
{
  {
    using T0 = cuda::std::tuple<long>;
    using T1 = cuda::std::tuple<long long>;
    T0 t0(2);
    T1 t1;
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 2);
  }
  {
    using T0 = cuda::std::tuple<long, char>;
    using T1 = cuda::std::tuple<long long, int>;
    T0 t0(2, 'a');
    T1 t1;
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
  }
  {
    using T0 = cuda::std::tuple<long, char, D>;
    using T1 = cuda::std::tuple<long long, int, B>;
    T0 t0(2, 'a', D(3));
    T1 t1;
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
    assert(cuda::std::get<2>(t1).id_ == 3);
  }
  {
    D d(3);
    D d2(2);
    using T0 = cuda::std::tuple<long, char, D&>;
    using T1 = cuda::std::tuple<long long, int, B&>;
    T0 t0(2, 'a', d2);
    T1 t1(1, 'b', d);
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
    assert(cuda::std::get<2>(t1).id_ == 2);
  }

  {
    using T0 = cuda::std::tuple<long, char, cuda::std::unique_ptr<D>>;
    using T1 = cuda::std::tuple<long long, int, cuda::std::unique_ptr<B>>;
    T0 t0(2, 'a', cuda::std::unique_ptr<D>(new D(3)));
    T1 t1;
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
    assert(cuda::std::get<2>(t1)->id_ == 3);
  }

  {
    // Test that tuple evaluates correctly applies an lvalue reference
    // before evaluating is_assignable (ie 'is_assignable<int&, int&&>')
    // instead of evaluating 'is_assignable<int&&, int&&>' which is false.
    int x = 42;
    int y = 43;
    cuda::std::tuple<int&&, E> t(cuda::std::move(x), E{});
    cuda::std::tuple<int&&, int> t2(cuda::std::move(y), 44);
    t = cuda::std::move(t2);
    assert(cuda::std::get<0>(t) == 43);
    assert(&cuda::std::get<0>(t) == &x);
  }
  return 0;
}
