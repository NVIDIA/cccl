//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// Test default template arg:

// template <class Rep, class Period = ratio<1>>
// class duration;

#include <cuda/std/chrono>
#include <cuda/std/ratio>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(
    cuda::std::is_same_v<cuda::std::chrono::duration<int, cuda::std::ratio<1>>, cuda::std::chrono::duration<int>>);

  return 0;
}
