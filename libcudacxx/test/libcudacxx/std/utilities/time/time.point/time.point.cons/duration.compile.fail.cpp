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

// explicit time_point(const duration& d);

// test for explicit

#include <cuda/std/chrono>

int main(int, char**)
{
  using Clock                                      = cuda::std::chrono::system_clock;
  using Duration                                   = cuda::std::chrono::milliseconds;
  cuda::std::chrono::time_point<Clock, Duration> t = Duration(3);

  return 0;
}
