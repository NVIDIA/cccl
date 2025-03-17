//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++17

// <cuda/std/chrono>

// template<class Duration>
//   using sys_time  = time_point<system_clock, Duration>;
// using sys_seconds = sys_time<seconds>;
// using sys_days    = sys_time<days>;

// [Example:
//   sys_seconds{sys_days{1970y/January/1}}.time_since_epoch() is 0s.
//   sys_seconds{sys_days{2000y/January/1}}.time_since_epoch() is 946’684’800s, which is 10’957 * 86’400s.
// —end example]

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  using system_clock = cuda::std::chrono::system_clock;
  using year         = cuda::std::chrono::year;

  using seconds = cuda::std::chrono::seconds;
  using minutes = cuda::std::chrono::minutes;
  using days    = cuda::std::chrono::days;

  using sys_seconds = cuda::std::chrono::sys_seconds;
  using sys_minutes = cuda::std::chrono::sys_time<minutes>;
  using sys_days    = cuda::std::chrono::sys_days;

  constexpr cuda::std::chrono::month January = cuda::std::chrono::January;

  static_assert(cuda::std::is_same_v<cuda::std::chrono::sys_time<seconds>, sys_seconds>);
  static_assert(cuda::std::is_same_v<cuda::std::chrono::sys_time<days>, sys_days>);

  //  Test the long form, too
  static_assert(cuda::std::is_same_v<cuda::std::chrono::time_point<system_clock, seconds>, sys_seconds>);
  static_assert(cuda::std::is_same_v<cuda::std::chrono::time_point<system_clock, minutes>, sys_minutes>);
  static_assert(cuda::std::is_same_v<cuda::std::chrono::time_point<system_clock, days>, sys_days>);

  //  Test some well known values
  sys_days d0 = sys_days{year{1970} / January / 1};
  sys_days d1 = sys_days{year{2000} / January / 1};
  static_assert(cuda::std::is_same_v<decltype(d0.time_since_epoch()), days>);
  assert(d0.time_since_epoch().count() == 0);
  assert(d1.time_since_epoch().count() == 10957);

  sys_seconds s0{d0};
  sys_seconds s1{d1};
  static_assert(cuda::std::is_same_v<decltype(s0.time_since_epoch()), seconds>);
  assert(s0.time_since_epoch().count() == 0);
  assert(s1.time_since_epoch().count() == 946684800L);

  return 0;
}
