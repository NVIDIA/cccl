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
//   typename common_type<Rep1, Rep2>::type
//   operator/(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(cuda_demote_unsupported_floating_point)

#include "truncate_fp.h"

int main(int, char**)
{
  {
    cuda::std::chrono::nanoseconds ns1(15);
    cuda::std::chrono::nanoseconds ns2(5);
    assert(ns1 / ns2 == 3);
  }
  {
    cuda::std::chrono::microseconds us1(15);
    cuda::std::chrono::nanoseconds ns2(5);
    assert(us1 / ns2 == 3000);
  }
  {
    cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(30);
    cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>> s2(5);
    assert(s1 / s2 == 6);
  }
  {
    cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(30);
    cuda::std::chrono::duration<double, cuda::std::ratio<3, 5>> s2(5);
    assert(s1 / s2 == truncate_fp(20. / 3));
  }
  {
    constexpr cuda::std::chrono::nanoseconds ns1(15);
    constexpr cuda::std::chrono::nanoseconds ns2(5);
    static_assert(ns1 / ns2 == 3, "");
  }
  {
    constexpr cuda::std::chrono::microseconds us1(15);
    constexpr cuda::std::chrono::nanoseconds ns2(5);
    static_assert(us1 / ns2 == 3000, "");
  }
  {
    constexpr cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(30);
    constexpr cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>> s2(5);
    static_assert(s1 / s2 == 6, "");
  }
  {
    constexpr cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(30);
    constexpr cuda::std::chrono::duration<double, cuda::std::ratio<3, 5>> s2(5);
    static_assert(s1 / s2 == 20. / 3, "");
  }

  return 0;
}
