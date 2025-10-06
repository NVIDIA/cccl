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

// template <cuda/std/class Duration2>
//   time_point(const time_point<cuda/std/clock, Duration2>& t);

// Duration2 shall be implicitly convertible to duration.

#include <cuda/std/chrono>

int main(int, char**)
{
  using Clock     = cuda::std::chrono::system_clock;
  using Duration1 = cuda::std::chrono::milliseconds;
  using Duration2 = cuda::std::chrono::microseconds;
  {
    cuda::std::chrono::time_point<Clock, Duration2> t2(Duration2(3));
    cuda::std::chrono::time_point<Clock, Duration1> t1 = t2;
  }

  return 0;
}
