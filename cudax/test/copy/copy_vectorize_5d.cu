//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// 5D vectorization tests (3 tests) in a separate TU to avoid a GCC false-positive -Warray-bounds warning when rank-2
// and rank-5 template instantiations coexist

#include "copy_common.cuh"

// 5D layouts with 48 contiguous elements that should coalesce into a (48):(1) layout
// and generate 16B vectorized copies.

static constexpr int N = 48;

/***********************************************************************************************************************
 * 5D Vectorization Tests
 **********************************************************************************************************************/

// src: (2,2,2,2,3):(2,1,4,8,16)
// dst: (2,2,2,2,3):(2,1,4,8,16)
TEST_CASE("copy d2d vectorize (2,2,2,2,3):(2,1,4,8,16)", "[copy][d2d][vectorize][5d]")
{
  test_copy_strided(
    make_iota<int>(N), cuda::std::array<int, 5>{2, 2, 2, 2, 3}, cuda::std::array<int, 5>{2, 1, 4, 8, 16});
}

// src: (2,2,2,2,3):(8,1,4,2,16)
// dst: (2,2,2,2,3):(8,1,4,2,16)
TEST_CASE("copy d2d vectorize (2,2,2,2,3):(8,1,4,2,16)", "[copy][d2d][vectorize][5d]")
{
  test_copy_strided(
    make_iota<int>(N), cuda::std::array<int, 5>{2, 2, 2, 2, 3}, cuda::std::array<int, 5>{8, 1, 4, 2, 16});
}

// src: (2,2,3,2,2):(6,3,1,24,12)
// dst: (2,2,3,2,2):(6,3,1,24,12)
TEST_CASE("copy d2d vectorize (2,2,3,2,2):(6,3,1,24,12)", "[copy][d2d][vectorize][5d]")
{
  test_copy_strided(
    make_iota<int>(N), cuda::std::array<int, 5>{2, 2, 3, 2, 2}, cuda::std::array<int, 5>{6, 3, 1, 24, 12});
}
