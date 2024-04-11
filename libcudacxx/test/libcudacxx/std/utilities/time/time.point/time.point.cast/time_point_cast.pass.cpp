//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// template <class ToDuration, class Clock, class Duration>
//   time_point<Clock, ToDuration>
//   time_point_cast(const time_point<Clock, Duration>& t);

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class FromDuration, class ToDuration>
__host__ __device__ void test(const FromDuration& df, const ToDuration& d)
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::time_point<Clock, FromDuration> FromTimePoint;
  typedef cuda::std::chrono::time_point<Clock, ToDuration> ToTimePoint;
  {
    FromTimePoint f(df);
    ToTimePoint t(d);
    typedef decltype(cuda::std::chrono::time_point_cast<ToDuration>(f)) R;
    static_assert((cuda::std::is_same<R, ToTimePoint>::value), "");
    assert(cuda::std::chrono::time_point_cast<ToDuration>(f) == t);
  }
}

#if TEST_STD_VER > 2011

template <class FromDuration, long long From, class ToDuration, long long To>
__host__ __device__ void test_constexpr()
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::time_point<Clock, FromDuration> FromTimePoint;
  typedef cuda::std::chrono::time_point<Clock, ToDuration> ToTimePoint;
  {
    constexpr FromTimePoint f{FromDuration{From}};
    constexpr ToTimePoint t{ToDuration{To}};
    static_assert(cuda::std::chrono::time_point_cast<ToDuration>(f) == t, "");
  }
}

#endif

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
#if TEST_STD_VER > 2011
  {
    test_constexpr<cuda::std::chrono::milliseconds, 7265000, cuda::std::chrono::hours, 2>();
    test_constexpr<cuda::std::chrono::milliseconds, 7265000, cuda::std::chrono::minutes, 121>();
    test_constexpr<cuda::std::chrono::milliseconds, 7265000, cuda::std::chrono::seconds, 7265>();
    test_constexpr<cuda::std::chrono::milliseconds, 7265000, cuda::std::chrono::milliseconds, 7265000>();
    test_constexpr<cuda::std::chrono::milliseconds, 7265000, cuda::std::chrono::microseconds, 7265000000LL>();
    test_constexpr<cuda::std::chrono::milliseconds, 7265000, cuda::std::chrono::nanoseconds, 7265000000000LL>();
    typedef cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>> T1;
    test_constexpr<cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>>, 9, T1, 10>();
  }
#endif

  return 0;
}
