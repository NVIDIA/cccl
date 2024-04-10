//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// template <class Rep1, class Period1, class Rep2, class Period2>
//   constexpr
//   typename common_type<duration<Rep1, Period1>, duration<Rep2, Period2>>::type
//   operator-(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::chrono::seconds s1(3);
    cuda::std::chrono::seconds s2(5);
    cuda::std::chrono::seconds r = s1 - s2;
    assert(r.count() == -2);
  }
  {
    cuda::std::chrono::seconds s1(3);
    cuda::std::chrono::microseconds s2(5);
    cuda::std::chrono::microseconds r = s1 - s2;
    assert(r.count() == 2999995);
  }
  {
    cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(3);
    cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>> s2(5);
    cuda::std::chrono::duration<int, cuda::std::ratio<1, 15>> r = s1 - s2;
    assert(r.count() == -15);
  }
  {
    cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(3);
    cuda::std::chrono::duration<double, cuda::std::ratio<3, 5>> s2(5);
    cuda::std::chrono::duration<double, cuda::std::ratio<1, 15>> r = s1 - s2;
    assert(r.count() == -15);
  }
  {
    constexpr cuda::std::chrono::seconds s1(3);
    constexpr cuda::std::chrono::seconds s2(5);
    constexpr cuda::std::chrono::seconds r = s1 - s2;
    static_assert(r.count() == -2, "");
  }
  {
    constexpr cuda::std::chrono::seconds s1(3);
    constexpr cuda::std::chrono::microseconds s2(5);
    constexpr cuda::std::chrono::microseconds r = s1 - s2;
    static_assert(r.count() == 2999995, "");
  }
  {
    constexpr cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(3);
    constexpr cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>> s2(5);
    constexpr cuda::std::chrono::duration<int, cuda::std::ratio<1, 15>> r = s1 - s2;
    static_assert(r.count() == -15, "");
  }
  {
    constexpr cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(3);
    constexpr cuda::std::chrono::duration<double, cuda::std::ratio<3, 5>> s2(5);
    constexpr cuda::std::chrono::duration<double, cuda::std::ratio<1, 15>> r = s1 - s2;
    static_assert(r.count() == -15, "");
  }

  return 0;
}
