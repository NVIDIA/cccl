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

// static constexpr time_point min(); // noexcept after C++17

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using Clock    = cuda::std::chrono::system_clock;
  using Duration = cuda::std::chrono::milliseconds;
  using TP       = cuda::std::chrono::time_point<Clock, Duration>;

  static_assert(noexcept(TP::min()));
  assert(TP::min() == TP(Duration::min()));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
