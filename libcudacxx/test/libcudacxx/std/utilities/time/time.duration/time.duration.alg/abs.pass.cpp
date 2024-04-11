//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <cuda/std/chrono>

// abs

// template <class Rep, class Period>
//   constexpr duration<Rep, Period> abs(duration<Rep, Period> d)

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

template <class Duration>
__host__ __device__ void test(const Duration& f, const Duration& d)
{
  {
    typedef decltype(cuda::std::chrono::abs(f)) R;
    static_assert((cuda::std::is_same<R, Duration>::value), "");
    assert(cuda::std::chrono::abs(f) == d);
  }
}

int main(int, char**)
{
  //  7290000ms is 2 hours, 1 minute, and 30 seconds
  test(cuda::std::chrono::milliseconds(7290000), cuda::std::chrono::milliseconds(7290000));
  test(cuda::std::chrono::milliseconds(-7290000), cuda::std::chrono::milliseconds(7290000));
  test(cuda::std::chrono::minutes(122), cuda::std::chrono::minutes(122));
  test(cuda::std::chrono::minutes(-122), cuda::std::chrono::minutes(122));
  test(cuda::std::chrono::hours(0), cuda::std::chrono::hours(0));

  {
    //  9000000ms is 2 hours and 30 minutes
    constexpr cuda::std::chrono::hours h1 = cuda::std::chrono::abs(cuda::std::chrono::hours(-3));
    static_assert(h1.count() == 3, "");
    constexpr cuda::std::chrono::hours h2 = cuda::std::chrono::abs(cuda::std::chrono::hours(3));
    static_assert(h2.count() == 3, "");
  }

  return 0;
}
