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

// ToDuration shall be an instantiation of duration.

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC

#include <cuda/std/chrono>

int main(int, char**)
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::time_point<Clock, cuda::std::chrono::milliseconds> FromTimePoint;
  typedef cuda::std::chrono::time_point<Clock, cuda::std::chrono::minutes> ToTimePoint;
  cuda::std::chrono::time_point_cast<ToTimePoint>(FromTimePoint(cuda::std::chrono::milliseconds(3)));

  return 0;
}
