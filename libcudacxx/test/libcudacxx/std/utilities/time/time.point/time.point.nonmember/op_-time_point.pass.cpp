//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// template <class Clock, class Duration1, class Duration2>
//   typename common_type<Duration1, Duration2>::type
//   operator-(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::milliseconds Duration1;
  typedef cuda::std::chrono::microseconds Duration2;
  {
    cuda::std::chrono::time_point<Clock, Duration1> t1(Duration1(3));
    cuda::std::chrono::time_point<Clock, Duration2> t2(Duration2(5));
    assert((t1 - t2) == Duration2(2995));
  }
#if TEST_STD_VER > 2011
  {
    constexpr cuda::std::chrono::time_point<Clock, Duration1> t1(Duration1(3));
    constexpr cuda::std::chrono::time_point<Clock, Duration2> t2(Duration2(5));
    static_assert((t1 - t2) == Duration2(2995), "");
  }
#endif

  return 0;
}
