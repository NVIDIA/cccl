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

// time_point();

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/ratio>

#include "../../rep.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using Clock    = cuda::std::chrono::system_clock;
  using Duration = cuda::std::chrono::duration<Rep, cuda::std::milli>;

  cuda::std::chrono::time_point<Clock, Duration> t;
  assert(t.time_since_epoch() == Duration::zero());

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
