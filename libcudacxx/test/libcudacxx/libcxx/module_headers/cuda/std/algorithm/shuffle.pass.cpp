//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile
// error: a non-__tile__ variable cannot be used in tile code

#include <cuda/std/algorithm.shuffle.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct urbg
{
  using result_type = unsigned;
  unsigned s        = 1;

  TEST_FUNC static constexpr result_type min()
  {
    return 0;
  }
  TEST_FUNC static constexpr result_type max()
  {
    return 0xFFFFFFFFu;
  }
  TEST_FUNC result_type operator()()
  {
    s = s * 48271u % 2147483647u;
    return s;
  }
};

TEST_FUNC bool test()
{
  int a[] = {1, 2, 3, 4};
  urbg g{};
  cuda::std::shuffle(a, a + 4, g);

  return true;
}

int main(int, char**)
{
  test();

  return 0;
}
