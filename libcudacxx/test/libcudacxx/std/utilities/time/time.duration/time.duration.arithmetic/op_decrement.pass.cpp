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

// constexpr duration& operator--();  // constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  cuda::std::chrono::hours h(3);
  cuda::std::chrono::hours& href = --h;
  assert(&href == &h);
  assert(h.count() == 2);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
