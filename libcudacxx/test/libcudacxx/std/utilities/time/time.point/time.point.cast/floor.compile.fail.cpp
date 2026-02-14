//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// floor

// template <cuda/std/class ToDuration, class Clock, class Duration>
//   time_point<cuda/std/Clock, ToDuration>
//   floor(const time_point<cuda/std/Clock, Duration>& t);

// ToDuration shall be an instantiation of duration.

#include <cuda/std/chrono>

int main(int, char**)
{
  cuda::std::chrono::floor<int>(cuda::std::chrono::system_clock::now());

  return 0;
}
