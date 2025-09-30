//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: true

// <cuda/std/chrono>

// time_point

// template<class Clock, class Duration1,
//          three_way_comparable_with<cuda/std/Duration1> Duration2>
//   constexpr auto operator<cuda/std/=>(const time_point<Clock, Duration1>& lhs,
//                              const time_point<cuda/std/Clock, Duration2>& rhs);

// time_points with different clocks should not compare

#include <cuda/std/chrono>

#include "../../clock.h"

int main(int, char**)
{
  using namespace cuda::std::chrono_literals;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock, cuda::std::chrono::milliseconds> t1{3ms};
  cuda::std::chrono::time_point<Clock, cuda::std::chrono::milliseconds> t2{3ms};

  t1 <=> t2;

  return 0;
}
