//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// template <class ToDuration, class Rep, class Period>
//   constexpr
//   ToDuration
//   duration_cast(const duration<Rep, Period>& d);

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class ToDuration, class FromDuration>
__host__ __device__ void test(const FromDuration& f, const ToDuration& d)
{
  {
    typedef decltype(cuda::std::chrono::duration_cast<ToDuration>(f)) R;
    static_assert((cuda::std::is_same<R, ToDuration>::value), "");
    assert(cuda::std::chrono::duration_cast<ToDuration>(f) == d);
  }
}

int main(int, char**)
{
  test(cuda::std::chrono::milliseconds(7265000), cuda::std::chrono::hours(2));
  test(cuda::std::chrono::milliseconds(7265000), cuda::std::chrono::minutes(121));
  test(cuda::std::chrono::milliseconds(7265000), cuda::std::chrono::seconds(7265));
  test(cuda::std::chrono::milliseconds(7265000), cuda::std::chrono::milliseconds(7265000));
  test(cuda::std::chrono::milliseconds(7265000), cuda::std::chrono::microseconds(7265000000LL));
  test(cuda::std::chrono::milliseconds(7265000), cuda::std::chrono::nanoseconds(7265000000000LL));
  test(cuda::std::chrono::milliseconds(7265000),
       cuda::std::chrono::duration<double, cuda::std::ratio<3600>>(7265. / 3600));
  test(cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>>(9),
       cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>>(10));
  {
    constexpr cuda::std::chrono::hours h =
      cuda::std::chrono::duration_cast<cuda::std::chrono::hours>(cuda::std::chrono::milliseconds(7265000));
    static_assert(h.count() == 2, "");
  }

  return 0;
}
