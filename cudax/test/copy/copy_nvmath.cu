//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include "copy_common.cuh"

using data_t = int8_t;

/***********************************************************************************************************************
 * nvmath minimal offset test (device-to-device)
 **********************************************************************************************************************/

// src: (3,4):(4,1), offset=4, alloc=16
// dst: (3,4):(4,1), offset=0, alloc=12
TEST_CASE("copy d2d nvmath minimal offset", "[copy][d2d][nvmath][debug]")
{
  cuda::std::array<int, 2> shape{3, 4};
  cuda::std::array<int, 2> src_strides{4, 1};
  cuda::std::array<int, 2> dst_strides{4, 1};
  test_copy_stride_relaxed<data_t>(16, 4, shape, src_strides, 12, 0, dst_strides);
}

// src: (4,256):(512,1), offset=256, alloc=2048+256
// dst: (4,256):(256,1), offset=0, alloc=1024
TEST_CASE("copy d2d nvmath medium offset", "[copy][d2d][nvmath][debug]")
{
  cuda::std::array<int, 2> shape{4, 256};
  cuda::std::array<int, 2> src_strides{512, 1};
  cuda::std::array<int, 2> dst_strides{256, 1};
  test_copy_stride_relaxed<data_t>(2048 + 256, 256, shape, src_strides, 1024, 0, dst_strides);
}

// src: (35,255,10,24):(61440,240,24,1), offset=240, alloc=35*256*10*24 — same as sliced_vec but smaller
// dst: (3,255,10,24):(61200,240,24,1), offset=0, alloc=3*255*10*24
TEST_CASE("copy d2d nvmath small sliced_vec", "[copy][d2d][nvmath][debug]")
{
  constexpr int src_alloc  = 3 * 256 * 10 * 24;
  constexpr int dst_alloc  = 3 * 255 * 10 * 24;
  constexpr int src_offset = 240;
  cuda::std::array<int, 4> shape{3, 255, 10, 24};
  cuda::std::array<int, 4> src_strides{61440, 240, 24, 1};
  cuda::std::array<int, 4> dst_strides{61200, 240, 24, 1};
  test_copy_stride_relaxed<data_t>(src_alloc, src_offset, shape, src_strides, dst_alloc, 0, dst_strides);
}

/***********************************************************************************************************************
 * nvmath memcpy test cases (device-to-device)
 **********************************************************************************************************************/

// src: (70,90,80,80):(576000,6400,80,1)
// dst: (70,90,80,80):(576000,6400,80,1)
TEST_CASE("copy d2d nvmath memcpy_layout_0", "[copy][d2d][nvmath][memcpy]")
{
  test_copy_iota<data_t>(70, 90, 80, 80);
}

// src: (70,90,80,80):(90,1,6300,504000)
// dst: (70,90,80,80):(90,1,6300,504000)
TEST_CASE("copy d2d nvmath memcpy_layout_1", "[copy][d2d][nvmath][memcpy]")
{
  constexpr int alloc = 70 * 90 * 80 * 80;
  test_copy_strided(
    make_iota<data_t>(alloc), cuda::std::array<int, 4>{70, 90, 80, 80}, cuda::std::array<int, 4>{90, 1, 6300, 504000});
}

// src: (1001,1007,3,31):(31217,1,31248217,1007), offset=0, alloc=1001*1007*3*31
// dst: (1001,1007,3,31):(31217,1,31248217,1007), offset=31248217, alloc=1001*1007*5*31
TEST_CASE("copy d2d nvmath memcpy_layout_2", "[copy][d2d][nvmath][memcpy]")
{
  constexpr int src_alloc  = 1001 * 1007 * 3 * 31;
  constexpr int dst_alloc  = 1001 * 1007 * 5 * 31;
  constexpr int dst_offset = 31248217;
  cuda::std::array<int, 4> shape{1001, 1007, 3, 31};
  cuda::std::array<int, 4> strides{31217, 1, 31248217, 1007};
  test_copy_stride_relaxed<data_t>(src_alloc, 0, shape, strides, dst_alloc, dst_offset, strides);
}

// src: (57,71,1,1007,1):(71497,1,12225987,71,4075329), offset=12225987+4075329, alloc=57*71*3*1007*3
// dst: (57,71,1,1007,1):(71497,1,4075329,71,20376645), offset=3*4075329, alloc=57*71*5*1007*1
TEST_CASE("copy d2d nvmath memcpy_layout_3", "[copy][d2d][nvmath][memcpy]")
{
  constexpr int src_alloc  = 57 * 71 * 3 * 1007 * 3;
  constexpr int src_offset = 12225987 + 4075329;
  constexpr int dst_alloc  = 57 * 71 * 5 * 1007 * 1;
  constexpr int dst_offset = 3 * 4075329;
  cuda::std::array<int, 5> shape{57, 71, 1, 1007, 1};
  cuda::std::array<int, 5> src_strides{71497, 1, 12225987, 71, 4075329};
  cuda::std::array<int, 5> dst_strides{71497, 1, 4075329, 71, 20376645};
  test_copy_stride_relaxed<data_t>(src_alloc, src_offset, shape, src_strides, dst_alloc, dst_offset, dst_strides);
}

// src: (63,70,1001):(1001,63063,-1), offset=1000
// dst: (63,70,1001):(1001,63063,-1), offset=1000
TEST_CASE("copy d2d nvmath memcpy_neg", "[copy][d2d][nvmath][memcpy]")
{
  constexpr int alloc  = 63 * 70 * 1001;
  constexpr int offset = 1000;
  cuda::std::array<int, 3> shape{63, 70, 1001};
  cuda::std::array<int, 3> strides{1001, 63063, -1};
  test_copy_stride_relaxed<data_t>(alloc, offset, shape, strides);
}

/***********************************************************************************************************************
 * nvmath reorder strides test case (device-to-device)
 **********************************************************************************************************************/

// src: (8,100019,4):(1100209,11,3), offset=0, alloc=8*100019*11
// dst: (8,100019,4):(1,8,800152), offset=0, alloc=8*100019*4
TEST_CASE("copy d2d nvmath reorder_strides", "[copy][d2d][nvmath][reorder_strides]")
{
  constexpr int src_alloc = 8 * 100019 * 11;
  constexpr int dst_alloc = 8 * 100019 * 4;
  cuda::std::array<int, 3> shape{8, 100019, 4};
  cuda::std::array<int, 3> src_strides{1100209, 11, 3};
  cuda::std::array<int, 3> dst_strides{1, 8, 800152};
  test_copy_stride_relaxed<data_t>(src_alloc, 0, shape, src_strides, dst_alloc, 0, dst_strides);
}

/***********************************************************************************************************************
 * nvmath negative strides test case (device-to-device)
 **********************************************************************************************************************/

// src: (63,70,1001):(-1001,-63063,-1), offset=alloc-1
// dst: (63,70,1001):(70070,1001,1), offset=0
TEST_CASE("copy d2d nvmath src_neg_stride", "[copy][d2d][nvmath][neg_stride]")
{
  constexpr int alloc      = 63 * 70 * 1001;
  constexpr int src_offset = alloc - 1;
  cuda::std::array<int, 3> shape{63, 70, 1001};
  cuda::std::array<int, 3> src_strides{-1001, -63063, -1};
  cuda::std::array<int, 3> dst_strides{70070, 1001, 1};
  test_copy_stride_relaxed<data_t>(alloc, src_offset, shape, src_strides, alloc, 0, dst_strides);
}

/***********************************************************************************************************************
 * nvmath squeezing and flattening test cases (device-to-device)
 **********************************************************************************************************************/

static cuda::std::array<int, 23> make_flatten_common_src_strides()
{
  return {1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 14, 1 << 13, 1 << 12, 1 << 11,
          1 << 10, 1 << 9,  1 << 8,  1 << 7,  1 << 6,  1 << 5,  1 << 4,  1 << 3,  1 << 2,  1 << 0,  1 << 1};
}

static cuda::std::array<int, 23> make_flatten_common_dst_strides()
{
  return {1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 14, 1 << 13, 1 << 12, 1 << 11,
          1 << 10, 1 << 9,  1 << 8,  1 << 7,  1 << 6,  1 << 5,  1 << 4,  1 << 3,  1 << 1,  1 << 2,  1 << 0};
}

// src: (2,)^23, bit-permuted strides
// dst: (2,)^23, different bit-permuted strides
TEST_CASE("copy d2d nvmath flatten_common", "[copy][d2d][nvmath][flatten]")
{
  constexpr int alloc = 1 << 23;
  cuda::std::array<int, 23> shape{};
  for (auto& s : shape)
  {
    s = 2;
  }
  test_copy_stride_relaxed<data_t>(
    alloc, 0, shape, make_flatten_common_src_strides(), alloc, 0, make_flatten_common_dst_strides());
}

// src: (4,2,...,2):(5,2^4,...,2^22), alloc=2^23
// dst: (4,2,...,2):(2^19,2^18,...,1), alloc=2^21
TEST_CASE("copy d2d nvmath flatten_one", "[copy][d2d][nvmath][flatten]")
{
  constexpr int src_alloc = 1 << 23;
  constexpr int dst_alloc = 1 << 21;
  cuda::std::array<int, 20> shape{4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  cuda::std::array<int, 20> src_strides{
    5,       1 << 4,  1 << 5,  1 << 6,  1 << 7,  1 << 8,  1 << 9,  1 << 10, 1 << 11, 1 << 12,
    1 << 13, 1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22};
  cuda::std::array<int, 20> dst_strides{
    1 << 19, 1 << 18, 1 << 17, 1 << 16, 1 << 15, 1 << 14, 1 << 13, 1 << 12, 1 << 11, 1 << 10,
    1 << 9,  1 << 8,  1 << 7,  1 << 6,  1 << 5,  1 << 4,  1 << 3,  1 << 2,  1 << 1,  1 << 0};
  test_copy_stride_relaxed<data_t>(src_alloc, 0, shape, src_strides, dst_alloc, 0, dst_strides);
}

/***********************************************************************************************************************
 * nvmath vectorize test cases (device-to-device)
 **********************************************************************************************************************/

// src: (35,255,10,24):(61440,240,24,1), offset=240, alloc=35*256*10*24
// dst: (35,255,10,24):(61200,240,24,1), offset=0, alloc=35*255*10*24
TEST_CASE("copy d2d nvmath sliced_vec", "[copy][d2d][nvmath][vectorize]")
{
  constexpr int src_alloc  = 35 * 256 * 10 * 24;
  constexpr int dst_alloc  = 35 * 255 * 10 * 24;
  constexpr int src_offset = 240;
  cuda::std::array<int, 4> shape{35, 255, 10, 24};
  cuda::std::array<int, 4> src_strides{61440, 240, 24, 1};
  cuda::std::array<int, 4> dst_strides{61200, 240, 24, 1};
  test_copy_stride_relaxed<data_t>(src_alloc, src_offset, shape, src_strides, dst_alloc, 0, dst_strides);
}

// src: (355,255,4,3):(3072,12,3,1), offset=12, alloc=355*256*4*3
// dst: (355,255,4,3):(3060,12,3,1), offset=0, alloc=355*255*4*3
TEST_CASE("copy d2d nvmath sliced_vec_2", "[copy][d2d][nvmath][vectorize]")
{
  constexpr int src_alloc  = 355 * 256 * 4 * 3;
  constexpr int dst_alloc  = 355 * 255 * 4 * 3;
  constexpr int src_offset = 12;
  cuda::std::array<int, 4> shape{355, 255, 4, 3};
  cuda::std::array<int, 4> src_strides{3072, 12, 3, 1};
  cuda::std::array<int, 4> dst_strides{3060, 12, 3, 1};
  test_copy_stride_relaxed<data_t>(src_alloc, src_offset, shape, src_strides, dst_alloc, 0, dst_strides);
}

// src: (35,255,5,10):(153000,600,20,1), offset=10*20+5=205, alloc=35*255*30*20
// dst: (35,255,5,10):(12750,50,10,1), offset=0, alloc=35*255*5*10
TEST_CASE("copy d2d nvmath sliced_unaligned_ptr", "[copy][d2d][nvmath][vectorize]")
{
  constexpr int src_alloc  = 35 * 255 * 30 * 20;
  constexpr int dst_alloc  = 35 * 255 * 5 * 10;
  constexpr int src_offset = 10 * 20 + 5;
  cuda::std::array<int, 4> shape{35, 255, 5, 10};
  cuda::std::array<int, 4> src_strides{153000, 600, 20, 1};
  cuda::std::array<int, 4> dst_strides{12750, 50, 10, 1};
  test_copy_stride_relaxed<data_t>(src_alloc, src_offset, shape, src_strides, dst_alloc, 0, dst_strides);
}
