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

#include <cuda/std/tuple>
// cuda::std::string not supported
// #include <cuda/std/string>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  {
    using T1 = cuda::std::tuple<>;
    using T2 = cuda::std::tuple<>;
    const T1 t1;
    const T2 t2;
    assert(!(t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert((t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long>;
    using T2 = cuda::std::tuple<double>;
    const T1 t1(1);
    const T2 t2(1);
    assert(!(t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert((t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long>;
    using T2 = cuda::std::tuple<double>;
    const T1 t1(1);
    const T2 t2(0.9);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long>;
    using T2 = cuda::std::tuple<double>;
    const T1 t1(1);
    const T2 t2(1.1);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long, int>;
    using T2 = cuda::std::tuple<double, long>;
    const T1 t1(1, 2);
    const T2 t2(1, 2);
    assert(!(t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert((t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long, int>;
    using T2 = cuda::std::tuple<double, long>;
    const T1 t1(1, 2);
    const T2 t2(0.9, 2);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long, int>;
    using T2 = cuda::std::tuple<double, long>;
    const T1 t1(1, 2);
    const T2 t2(1.1, 2);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long, int>;
    using T2 = cuda::std::tuple<double, long>;
    const T1 t1(1, 2);
    const T2 t2(1, 1);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long, int>;
    using T2 = cuda::std::tuple<double, long>;
    const T1 t1(1, 2);
    const T2 t2(1, 3);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long, int, double>;
    using T2 = cuda::std::tuple<double, long, int>;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 2, 3);
    assert(!(t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert((t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long, int, double>;
    using T2 = cuda::std::tuple<double, long, int>;
    const T1 t1(1, 2, 3);
    const T2 t2(0.9, 2, 3);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long, int, double>;
    using T2 = cuda::std::tuple<double, long, int>;
    const T1 t1(1, 2, 3);
    const T2 t2(1.1, 2, 3);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long, int, double>;
    using T2 = cuda::std::tuple<double, long, int>;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 1, 3);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long, int, double>;
    using T2 = cuda::std::tuple<double, long, int>;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 3, 3);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long, int, double>;
    using T2 = cuda::std::tuple<double, long, int>;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 2, 2);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long, int, double>;
    using T2 = cuda::std::tuple<double, long, int>;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 2, 4);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    using T1 = cuda::std::tuple<long, int, double>;
    using T2 = cuda::std::tuple<double, long, int>;
    constexpr T1 t1(1, 2, 3);
    constexpr T2 t2(1, 2, 4);
    static_assert((t1 < t2), "");
    static_assert((t1 <= t2), "");
    static_assert(!(t1 > t2), "");
    static_assert(!(t1 >= t2), "");
  }

  return 0;
}
