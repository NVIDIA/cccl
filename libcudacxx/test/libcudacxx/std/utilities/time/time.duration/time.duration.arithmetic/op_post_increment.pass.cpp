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

// constexpr duration operator++(int);  // constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  cuda::std::chrono::hours h1(3);
  cuda::std::chrono::hours h2 = h1++;
  assert(h1.count() == 4);
  assert(h2.count() == 3);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
