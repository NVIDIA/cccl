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

// inexact conversions allowed for floating point reps

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::chrono::duration<double, cuda::std::micro> us(1);
    cuda::std::chrono::duration<double, cuda::std::milli> ms = us;
    assert(ms.count() == 1. / 1000);
  }
  {
    constexpr cuda::std::chrono::duration<double, cuda::std::micro> us(1);
    constexpr cuda::std::chrono::duration<double, cuda::std::milli> ms = us;
    static_assert(ms.count() == 1. / 1000, "");
  }

  return 0;
}
