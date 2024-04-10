//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// template <class Rep2, class Period2>
//   duration(const duration<Rep2, Period2>& d);

//  conversions from integral to floating point durations allowed

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::chrono::duration<int> i(3);
    cuda::std::chrono::duration<double, cuda::std::milli> d = i;
    assert(d.count() == 3000);
  }
  {
    constexpr cuda::std::chrono::duration<int> i(3);
    constexpr cuda::std::chrono::duration<double, cuda::std::milli> d = i;
    static_assert(d.count() == 3000, "");
  }

  return 0;
}
