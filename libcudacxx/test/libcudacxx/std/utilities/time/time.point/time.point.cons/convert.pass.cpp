//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// template <class Duration2>
//   time_point(const time_point<clock, Duration2>& t);

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::microseconds Duration1;
  typedef cuda::std::chrono::milliseconds Duration2;
  {
    cuda::std::chrono::time_point<Clock, Duration2> t2(Duration2(3));
    cuda::std::chrono::time_point<Clock, Duration1> t1 = t2;
    assert(t1.time_since_epoch() == Duration1(3000));
  }
#if TEST_STD_VER > 2011
  {
    constexpr cuda::std::chrono::time_point<Clock, Duration2> t2(Duration2(3));
    constexpr cuda::std::chrono::time_point<Clock, Duration1> t1 = t2;
    static_assert(t1.time_since_epoch() == Duration1(3000), "");
  }
#endif

  return 0;
}
