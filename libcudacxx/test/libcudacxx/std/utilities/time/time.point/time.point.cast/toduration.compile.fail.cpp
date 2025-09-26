//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// template <cuda/std/class ToDuration, class Clock, class Duration>
//   time_point<cuda/std/Clock, ToDuration>
//   time_point_cast(const time_point<cuda/std/Clock, Duration>& t);

// ToDuration shall be an instantiation of duration.

#include <cuda/std/chrono>

int main(int, char**)
{
  using Clock         = cuda::std::chrono::system_clock;
  using FromTimePoint = cuda::std::chrono::time_point<Clock, cuda::std::chrono::milliseconds>;
  using ToTimePoint   = cuda::std::chrono::time_point<Clock, cuda::std::chrono::minutes>;
  cuda::std::chrono::time_point_cast<ToTimePoint>(FromTimePoint(cuda::std::chrono::milliseconds(3)));

  return 0;
}
