//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// explicit time_point(const duration& d);

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::milliseconds Duration;
  {
    cuda::std::chrono::time_point<Clock, Duration> t(Duration(3));
    assert(t.time_since_epoch() == Duration(3));
  }
  {
    cuda::std::chrono::time_point<Clock, Duration> t(cuda::std::chrono::seconds(3));
    assert(t.time_since_epoch() == Duration(3000));
  }
#if TEST_STD_VER > 2011
  {
    constexpr cuda::std::chrono::time_point<Clock, Duration> t(Duration(3));
    static_assert(t.time_since_epoch() == Duration(3), "");
  }
  {
    constexpr cuda::std::chrono::time_point<Clock, Duration> t(cuda::std::chrono::seconds(3));
    static_assert(t.time_since_epoch() == Duration(3000), "");
  }
#endif

  return 0;
}
