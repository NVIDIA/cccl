//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: dynamic memory allocation is unsupported in tile code

#include <cuda/std/cassert>
#include <cuda/std/random>

#include "test_macros.h"

TEST_FUNC void test()
{
  static_assert(!cuda::std::is_copy_assignable_v<cuda::std::seed_seq>);
  static_assert(!cuda::std::is_move_assignable_v<cuda::std::seed_seq>);
  static_assert(!cuda::std::is_copy_constructible_v<cuda::std::seed_seq>);
  static_assert(!cuda::std::is_move_constructible_v<cuda::std::seed_seq>);
}

int main(int, char**)
{
  test();
  return 0;
}
