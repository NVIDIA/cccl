//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template<class... TTypes, class... UTypes>
//   bool
//   operator<(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
//
// template<class... TTypes, class... UTypes>
//   bool
//   operator>(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
//
// template<class... TTypes, class... UTypes>
//   bool
//   operator<=(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
//
// template<class... TTypes, class... UTypes>
//   bool
//   operator>=(const tuple<TTypes...>& t, const tuple<UTypes...>& u);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/tuple>
// cuda::std::string not supported
// #include <cuda/std/string>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  {
    typedef cuda::std::tuple<> T1;
    typedef cuda::std::tuple<> T2;
    const T1 t1;
    const T2 t2;
    assert(!(t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long> T1;
    typedef cuda::std::tuple<double> T2;
    const T1 t1(1);
    const T2 t2(1);
    assert(!(t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long> T1;
    typedef cuda::std::tuple<double> T2;
    const T1 t1(1);
    const T2 t2(0.9);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long> T1;
    typedef cuda::std::tuple<double> T2;
    const T1 t1(1);
    const T2 t2(1.1);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long, int> T1;
    typedef cuda::std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1, 2);
    assert(!(t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long, int> T1;
    typedef cuda::std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(0.9, 2);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long, int> T1;
    typedef cuda::std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1.1, 2);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long, int> T1;
    typedef cuda::std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1, 1);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long, int> T1;
    typedef cuda::std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1, 3);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long, int, double> T1;
    typedef cuda::std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 2, 3);
    assert(!(t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long, int, double> T1;
    typedef cuda::std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(0.9, 2, 3);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long, int, double> T1;
    typedef cuda::std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1.1, 2, 3);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long, int, double> T1;
    typedef cuda::std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 1, 3);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long, int, double> T1;
    typedef cuda::std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 3, 3);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long, int, double> T1;
    typedef cuda::std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 2, 2);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef cuda::std::tuple<long, int, double> T1;
    typedef cuda::std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 2, 4);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
#if TEST_STD_VER > 2011
  {
    typedef cuda::std::tuple<long, int, double> T1;
    typedef cuda::std::tuple<double, long, int> T2;
    constexpr T1 t1(1, 2, 3);
    constexpr T2 t2(1, 2, 4);
    static_assert((t1 < t2), "");
    static_assert((t1 <= t2), "");
    static_assert(!(t1 > t2), "");
    static_assert(!(t1 >= t2), "");
  }
#endif

  return 0;
}
