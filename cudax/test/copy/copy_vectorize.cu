//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "copy_common.cuh"

// All layouts in this file have 48 contiguous elements that should coalesce into a (48):(1) layout
// and generate 16B vectorized copies.

static constexpr int N = 48;

/***********************************************************************************************************************
 * 1D Vectorization Test
 **********************************************************************************************************************/

// src: (48):(1)
// dst: (48):(1)
TEST_CASE("copy d2d vectorize 48:1", "[copy][d2d][vectorize][1d]")
{
  test_copy<layout_right>(make_iota<int>(N), N);
}

/***********************************************************************************************************************
 * 2D Vectorization Tests
 **********************************************************************************************************************/

// src: (6,8):(8,1)
// dst: (6,8):(8,1)
TEST_CASE("copy d2d vectorize (6,8):(8,1)", "[copy][d2d][vectorize][2d]")
{
  test_copy<layout_right>(make_iota<int>(N), 6, 8);
}

// src: (8,6):(1,8)
// dst: (8,6):(1,8)
TEST_CASE("copy d2d vectorize (8,6):(1,8)", "[copy][d2d][vectorize][2d]")
{
  test_copy<layout_left>(make_iota<int>(N), 8, 6);
}

// src: (6,8):(1,6)
// dst: (6,8):(1,6)
TEST_CASE("copy d2d vectorize (6,8):(1,6)", "[copy][d2d][vectorize][2d]")
{
  test_copy<layout_left>(make_iota<int>(N), 6, 8);
}

// src: (8,6):(6,1)
// dst: (8,6):(6,1)
TEST_CASE("copy d2d vectorize (8,6):(6,1)", "[copy][d2d][vectorize][2d]")
{
  test_copy<layout_right>(make_iota<int>(N), 8, 6);
}

/***********************************************************************************************************************
 * 2D Non-Square Vectorization Tests
 **********************************************************************************************************************/

// src: (3,16):(1,3)
// dst: (3,16):(1,3)
TEST_CASE("copy d2d vectorize (3,16):(1,3)", "[copy][d2d][vectorize][2d]")
{
  test_copy<layout_left>(make_iota<int>(N), 3, 16);
}

// src: (3,16):(16,1)
// dst: (3,16):(16,1)
TEST_CASE("copy d2d vectorize (3,16):(16,1)", "[copy][d2d][vectorize][2d]")
{
  test_copy<layout_right>(make_iota<int>(N), 3, 16);
}

// src: (16,3):(1,16)
// dst: (16,3):(1,16)
TEST_CASE("copy d2d vectorize (16,3):(1,16)", "[copy][d2d][vectorize][2d]")
{
  test_copy<layout_left>(make_iota<int>(N), 16, 3);
}

// src: (16,3):(3,1)
// dst: (16,3):(3,1)
TEST_CASE("copy d2d vectorize (16,3):(3,1)", "[copy][d2d][vectorize][2d]")
{
  test_copy<layout_right>(make_iota<int>(N), 16, 3);
}
