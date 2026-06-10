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

// Test default template arg:

// template <cuda/std/class Clock, class Duration = typename Clock::duration>
//   class time_point;

#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::is_same_v<cuda::std::chrono::system_clock::duration,
                                     cuda::std::chrono::time_point<cuda::std::chrono::system_clock>::duration>);

  return 0;
}
