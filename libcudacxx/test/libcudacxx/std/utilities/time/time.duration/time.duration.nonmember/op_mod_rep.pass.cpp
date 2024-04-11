//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// template <class Rep1, class Period, class Rep2>
//   constexpr
//   duration<typename common_type<Rep1, Rep2>::type, Period>
//   operator%(const duration<Rep1, Period>& d, const Rep2& s)

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::chrono::nanoseconds ns(15);
    ns = ns % 6;
    assert(ns.count() == 3);
  }
  {
    constexpr cuda::std::chrono::nanoseconds ns(15);
    constexpr cuda::std::chrono::nanoseconds ns2 = ns % 6;
    static_assert(ns2.count() == 3, "");
  }

  return 0;
}
