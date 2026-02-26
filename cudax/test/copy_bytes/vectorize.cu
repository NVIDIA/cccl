//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/host_vector.h>

#include <cute/layout.hpp>

#include "copy_bytes_common.cuh"

// Tested Layouts -> must generate 16B vectorized copy and (48):(1) layout
// --------------------------
// (6,8):(8,1)
// (8,6):(1,8)
// (6,8):(1,6)
// (8,6):(6,1)
// 48:1
// (3,16):(1,3)
// (3,16):(16,1)
// (16,3):(1,16)
// (16,3):(3,1)
// (2,2,2,2,3):(2,1,4,8,16)
// (2,2,2,2,3):(8,1,4,2,16)
// (2,2,3,2,2):(6,3,1,24,12)

static constexpr int N = 48;

static thrust::host_vector<int> make_data()
{
  thrust::host_vector<int> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = i;
  }
  return data;
}

/***********************************************************************************************************************
 * 2D Vectorization Tests
 **********************************************************************************************************************/

// (6,8):(8,1) — row-major 6x8
TEST_CASE("vectorize (6,8):(8,1)", "[vectorize][2d]")
{
  using namespace cute;
  test_impl(make_data(), make_layout(make_shape(6, 8), make_stride(8, 1)));
}

// (8,6):(1,8) — column-major 8x6
TEST_CASE("vectorize (8,6):(1,8)", "[vectorize][2d]")
{
  using namespace cute;
  test_impl(make_data(), make_layout(make_shape(8, 6), make_stride(1, 8)));
}

// (6,8):(1,6) — column-major 6x8
TEST_CASE("vectorize (6,8):(1,6)", "[vectorize][2d]")
{
  using namespace cute;
  test_impl(make_data(), make_layout(make_shape(6, 8), make_stride(1, 6)));
}

// (8,6):(6,1) — row-major 8x6
TEST_CASE("vectorize (8,6):(6,1)", "[vectorize][2d]")
{
  using namespace cute;
  test_impl(make_data(), make_layout(make_shape(8, 6), make_stride(6, 1)));
}

/***********************************************************************************************************************
 * 1D Vectorization Test
 **********************************************************************************************************************/

// 48:1 — 1D contiguous
TEST_CASE("vectorize 48:1", "[vectorize][1d]")
{
  using namespace cute;
  test_impl(make_data(), make_layout(make_shape(N), make_stride(1)));
}

/***********************************************************************************************************************
 * 2D Non-Square Vectorization Tests
 **********************************************************************************************************************/

// (3,16):(1,3) — column-major 3x16
TEST_CASE("vectorize (3,16):(1,3)", "[vectorize][2d]")
{
  using namespace cute;
  test_impl(make_data(), make_layout(make_shape(3, 16), make_stride(1, 3)));
}

// (3,16):(16,1) — row-major 3x16
TEST_CASE("vectorize (3,16):(16,1)", "[vectorize][2d]")
{
  using namespace cute;
  test_impl(make_data(), make_layout(make_shape(3, 16), make_stride(16, 1)));
}

// (16,3):(1,16) — column-major 16x3
TEST_CASE("vectorize (16,3):(1,16)", "[vectorize][2d]")
{
  using namespace cute;
  test_impl(make_data(), make_layout(make_shape(16, 3), make_stride(1, 16)));
}

// (16,3):(3,1) — row-major 16x3
TEST_CASE("vectorize (16,3):(3,1)", "[vectorize][2d]")
{
  using namespace cute;
  test_impl(make_data(), make_layout(make_shape(16, 3), make_stride(3, 1)));
}

/***********************************************************************************************************************
 * 5D Vectorization Tests
 **********************************************************************************************************************/

// (2,2,2,2,3):(2,1,4,8,16)
TEST_CASE("vectorize (2,2,2,2,3):(2,1,4,8,16)", "[vectorize][5d]")
{
  using namespace cute;
  test_impl(make_data(), make_layout(make_shape(2, 2, 2, 2, 3), make_stride(2, 1, 4, 8, 16)));
}

// (2,2,2,2,3):(8,1,4,2,16)
TEST_CASE("vectorize (2,2,2,2,3):(8,1,4,2,16)", "[vectorize][5d]")
{
  using namespace cute;
  test_impl(make_data(), make_layout(make_shape(2, 2, 2, 2, 3), make_stride(8, 1, 4, 2, 16)));
}

// (2,2,3,2,2):(6,3,1,24,12)
TEST_CASE("vectorize (2,2,3,2,2):(6,3,1,24,12)", "[vectorize][5d]")
{
  using namespace cute;
  test_impl(make_data(), make_layout(make_shape(2, 2, 3, 2, 2), make_stride(6, 3, 1, 24, 12)));
}
