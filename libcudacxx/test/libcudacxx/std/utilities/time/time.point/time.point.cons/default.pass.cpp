//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// time_point();

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "../../rep.h"
#include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::duration<Rep, cuda::std::milli> Duration;
  {
    cuda::std::chrono::time_point<Clock, Duration> t;
    assert(t.time_since_epoch() == Duration::zero());
  }
#if TEST_STD_VER > 2011
  {
    constexpr cuda::std::chrono::time_point<Clock, Duration> t;
    static_assert(t.time_since_epoch() == Duration::zero(), "");
  }
#endif

  return 0;
}
