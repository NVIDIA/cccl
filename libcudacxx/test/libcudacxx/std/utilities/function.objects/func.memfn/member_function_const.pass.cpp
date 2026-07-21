//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile
// error: function-to-pointer decay is unsupported in tile code
// error: taking address of a function is unsupported in tile code

// <cuda/std/functional>

// template<Returnable R, class T, CopyConstructible... Args>
//   unspecified mem_fn(R (T::* pm)(Args...) const);

#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "test_macros.h"

struct A
{
  TEST_FUNC char test0() const
  {
    return 'a';
  }
  TEST_FUNC char test1(int) const
  {
    return 'b';
  }
  TEST_FUNC char test2(int, double) const
  {
    return 'c';
  }
};

template <class F>
TEST_FUNC void test0(F f)
{
  {
    A a;
    assert(f(a) == 'a');
    A* ap = &a;
    assert(f(ap) == 'a');
    const A* cap = &a;
    assert(f(cap) == 'a');
    const F& cf = f;
    assert(cf(ap) == 'a');
  }
}

template <class F>
TEST_FUNC void test1(F f)
{
  {
    A a;
    assert(f(a, 1) == 'b');
    A* ap = &a;
    assert(f(ap, 2) == 'b');
    const A* cap = &a;
    assert(f(cap, 2) == 'b');
    const F& cf = f;
    assert(cf(ap, 2) == 'b');
  }
}

template <class F>
TEST_FUNC void test2(F f)
{
  {
    A a;
    assert(f(a, 1, 2) == 'c');
    A* ap = &a;
    assert(f(ap, 2, 3.5) == 'c');
    const A* cap = &a;
    assert(f(cap, 2, 3.5) == 'c');
    const F& cf = f;
    assert(cf(ap, 2, 3.5) == 'c');
  }
}

int main(int, char**)
{
  test0(cuda::std::mem_fn(&A::test0));
  test1(cuda::std::mem_fn(&A::test1));
  test2(cuda::std::mem_fn(&A::test2));

  return 0;
}
