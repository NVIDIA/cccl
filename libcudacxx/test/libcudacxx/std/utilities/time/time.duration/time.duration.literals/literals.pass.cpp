//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <cuda/std/chrono>

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using namespace cuda::std::literals::chrono_literals;

  //    Make sure the types are right
  static_assert(cuda::std::is_same<decltype(3h), cuda::std::chrono::hours>::value, "");
  static_assert(cuda::std::is_same<decltype(3min), cuda::std::chrono::minutes>::value, "");
  static_assert(cuda::std::is_same<decltype(3s), cuda::std::chrono::seconds>::value, "");
  static_assert(cuda::std::is_same<decltype(3ms), cuda::std::chrono::milliseconds>::value, "");
  static_assert(cuda::std::is_same<decltype(3us), cuda::std::chrono::microseconds>::value, "");
  static_assert(cuda::std::is_same<decltype(3ns), cuda::std::chrono::nanoseconds>::value, "");

  cuda::std::chrono::hours h = 4h;
  assert(h == cuda::std::chrono::hours(4));
  auto h2 = 4.0h;
  assert(h == h2);

  cuda::std::chrono::minutes min = 36min;
  assert(min == cuda::std::chrono::minutes(36));
  auto min2 = 36.0min;
  assert(min == min2);

  cuda::std::chrono::seconds s = 24s;
  assert(s == cuda::std::chrono::seconds(24));
  auto s2 = 24.0s;
  assert(s == s2);

  cuda::std::chrono::milliseconds ms = 247ms;
  assert(ms == cuda::std::chrono::milliseconds(247));
  auto ms2 = 247.0ms;
  assert(ms == ms2);

  cuda::std::chrono::microseconds us = 867us;
  assert(us == cuda::std::chrono::microseconds(867));
  auto us2 = 867.0us;
  assert(us == us2);

  cuda::std::chrono::nanoseconds ns = 645ns;
  assert(ns == cuda::std::chrono::nanoseconds(645));
  auto ns2 = 645.ns;
  assert(ns == ns2);

  return 0;
}
