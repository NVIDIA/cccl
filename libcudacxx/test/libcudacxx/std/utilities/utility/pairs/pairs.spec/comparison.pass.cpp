//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template <class T1, class T2> bool operator==(const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator!=(const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator< (const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator> (const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator>=(const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator<=(const pair<T1,T2>&, const pair<T1,T2>&);

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  {
    using P = cuda::std::pair<int, short>;
    P p1(3, static_cast<short>(4));
    P p2(3, static_cast<short>(4));
    assert((p1 == p2));
    assert(!(p1 != p2));
    assert(!(p1 < p2));
    assert((p1 <= p2));
    assert(!(p1 > p2));
    assert((p1 >= p2));
  }
  {
    using P = cuda::std::pair<int, short>;
    P p1(2, static_cast<short>(4));
    P p2(3, static_cast<short>(4));
    assert(!(p1 == p2));
    assert((p1 != p2));
    assert((p1 < p2));
    assert((p1 <= p2));
    assert(!(p1 > p2));
    assert(!(p1 >= p2));
  }
  {
    using P = cuda::std::pair<int, short>;
    P p1(3, static_cast<short>(2));
    P p2(3, static_cast<short>(4));
    assert(!(p1 == p2));
    assert((p1 != p2));
    assert((p1 < p2));
    assert((p1 <= p2));
    assert(!(p1 > p2));
    assert(!(p1 >= p2));
  }
  {
    using P = cuda::std::pair<int, short>;
    P p1(3, static_cast<short>(4));
    P p2(2, static_cast<short>(4));
    assert(!(p1 == p2));
    assert((p1 != p2));
    assert(!(p1 < p2));
    assert(!(p1 <= p2));
    assert((p1 > p2));
    assert((p1 >= p2));
  }
  {
    using P = cuda::std::pair<int, short>;
    P p1(3, static_cast<short>(4));
    P p2(3, static_cast<short>(2));
    assert(!(p1 == p2));
    assert((p1 != p2));
    assert(!(p1 < p2));
    assert(!(p1 <= p2));
    assert((p1 > p2));
    assert((p1 >= p2));
  }

  {
    using P = cuda::std::pair<int, short>;
    constexpr P p1(3, static_cast<short>(4));
    constexpr P p2(3, static_cast<short>(2));
    static_assert(!(p1 == p2), "");
    static_assert((p1 != p2), "");
    static_assert(!(p1 < p2), "");
    static_assert(!(p1 <= p2), "");
    static_assert((p1 > p2), "");
    static_assert((p1 >= p2), "");
  }

  return 0;
}
