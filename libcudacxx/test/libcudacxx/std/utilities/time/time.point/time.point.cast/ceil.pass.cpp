//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <cuda/std/chrono>

// ceil

// template <class ToDuration, class Clock, class Duration>
//   time_point<Clock, ToDuration>
//   ceil(const time_point<Clock, Duration>& t);

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

template <class FromDuration, class ToDuration>
__host__ __device__ void test(const FromDuration& df, const ToDuration& d)
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::time_point<Clock, FromDuration> FromTimePoint;
  typedef cuda::std::chrono::time_point<Clock, ToDuration> ToTimePoint;
  {
    FromTimePoint f(df);
    ToTimePoint t(d);
    typedef decltype(cuda::std::chrono::ceil<ToDuration>(f)) R;
    static_assert((cuda::std::is_same<R, ToTimePoint>::value), "");
    assert(cuda::std::chrono::ceil<ToDuration>(f) == t);
  }
}

template <class FromDuration, long long From, class ToDuration, long long To>
__host__ __device__ void test_constexpr()
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::time_point<Clock, FromDuration> FromTimePoint;
  typedef cuda::std::chrono::time_point<Clock, ToDuration> ToTimePoint;
  {
    constexpr FromTimePoint f{FromDuration{From}};
    constexpr ToTimePoint t{ToDuration{To}};
    static_assert(cuda::std::chrono::ceil<ToDuration>(f) == t, "");
  }
}

int main(int, char**)
{
  //  7290000ms is 2 hours, 1 minute, and 30 seconds
  test(cuda::std::chrono::milliseconds(7290000), cuda::std::chrono::hours(3));
  test(cuda::std::chrono::milliseconds(-7290000), cuda::std::chrono::hours(-2));
  test(cuda::std::chrono::milliseconds(7290000), cuda::std::chrono::minutes(122));
  test(cuda::std::chrono::milliseconds(-7290000), cuda::std::chrono::minutes(-121));

  //  9000000ms is 2 hours and 30 minutes
  test_constexpr<cuda::std::chrono::milliseconds, 9000000, cuda::std::chrono::hours, 3>();
  test_constexpr<cuda::std::chrono::milliseconds, -9000000, cuda::std::chrono::hours, -2>();
  test_constexpr<cuda::std::chrono::milliseconds, 9000001, cuda::std::chrono::minutes, 151>();
  test_constexpr<cuda::std::chrono::milliseconds, -9000001, cuda::std::chrono::minutes, -150>();

  test_constexpr<cuda::std::chrono::milliseconds, 9000000, cuda::std::chrono::seconds, 9000>();
  test_constexpr<cuda::std::chrono::milliseconds, -9000000, cuda::std::chrono::seconds, -9000>();

  return 0;
}
