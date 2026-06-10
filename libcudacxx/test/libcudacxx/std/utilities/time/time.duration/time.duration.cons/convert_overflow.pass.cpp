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

// template <class Rep2, class Period2>
//   duration(const duration<Rep2, Period2>& d);

// overflow should SFINAE instead of error out, LWG 2094

// XFAIL: enable-tile
// error: a non-__tile__ variable cannot be used in tile code

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/ratio>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE bool called = false;

TEST_FUNC void f(cuda::std::chrono::milliseconds);
TEST_FUNC void f(cuda::std::chrono::seconds)
{
  called = true;
}

int main(int, char**)
{
  {
    cuda::std::chrono::duration<int, cuda::std::exa> r(1);
    f(r);
    assert(called);
  }

  return 0;
}
