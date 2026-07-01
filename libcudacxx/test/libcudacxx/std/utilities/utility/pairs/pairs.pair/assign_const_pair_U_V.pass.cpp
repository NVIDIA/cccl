//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template<class U, class V> pair& operator=(const pair<U, V>& p);

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "archetypes.h"
#include "test_macros.h"

struct CopyAssignableInt
{
  TEST_FUNC constexpr CopyAssignableInt& operator=(int&)
  {
    return *this;
  }
};

struct Unrelated
{};

TEST_FUNC constexpr bool test()
{
  {
    typedef cuda::std::pair<int, short> P1;
    typedef cuda::std::pair<double, long> P2;
    P1 p1(3, static_cast<short>(4));
    P2 p2;
    p2 = p1;
    assert(p2.first == 3);
    assert(p2.second == 4);
  }
  {
    using C = ConstexprTestTypes::TestType;
    using P = cuda::std::pair<int, C>;
    using T = cuda::std::pair<long, C>;
    const T t(42, -42);
    P p(101, 101);
    p = t;
    assert(p.first == 42);
    assert(p.second.value == -42);
  }
  {
    int i = 0, j = 0;
    cuda::std::pair<int&, int&> p(i, j);
    const cuda::std::pair<const int, const int> from(11, 12);
    p = from;
    assert(i == 11);
    assert(j == 12);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
