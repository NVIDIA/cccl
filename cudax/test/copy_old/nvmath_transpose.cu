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

#include "copy_bytes_common.cuh"
#include <cute/layout.hpp>

using data_t = int8_t;

/***********************************************************************************************************************
 * nvmath transpose test cases (device-to-device)
 **********************************************************************************************************************/

// transpose_merge_extents: 20-dim tensor (2,)*20, reversed stride orders
// dst F-order, src C-order. Tests merging extents for tiling.
//
// src (2,)^20:(2^19,2^18,...,1)
// stride_order (19,18,...,0), slice [:]
// dst (2,)^20:(1,2,...,2^19)
// stride_order (0,1,...,19), slice [:]
TEST_CASE("nvmath transpose_merge_extents", "[nvmath][transpose][d2d]")
{
  using namespace cute;
  constexpr int alloc = 1 << 20;
  // clang-format off
  auto shape = make_shape(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
  auto dst_layout = make_layout(shape,
    make_stride(1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7,
                1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15,
                1 << 16, 1 << 17, 1 << 18, 1 << 19));
  auto src_layout = make_layout(shape,
    make_stride(1 << 19, 1 << 18, 1 << 17, 1 << 16, 1 << 15, 1 << 14, 1 << 13, 1 << 12,
                1 << 11, 1 << 10, 1 << 9, 1 << 8, 1 << 7, 1 << 6, 1 << 5, 1 << 4,
                1 << 3, 1 << 2, 1 << 1, 1 << 0));
  // clang-format on
  test_impl<data_t>(alloc, 0, src_layout, alloc, 0, dst_layout);
}

// transpose: tiling efficiency test
//
// src (55,13,55,55):(13, 1, 39325, 715)
// stride_order (2,3,0,1), slice [:]
// dst (55,13,55,55):(39325, 3025, 55, 1)
// stride_order (0,1,2,3), slice [:]
TEST_CASE("nvmath transpose", "[nvmath][transpose][d2d]")
{
  using namespace cute;
  constexpr int alloc = 55 * 13 * 55 * 55;
  auto dst_layout     = make_layout(make_shape(55, 13, 55, 55), make_stride(39325, 3025, 55, 1));
  auto src_layout     = make_layout(make_shape(55, 13, 55, 55), make_stride(13, 1, 39325, 715));
  test_impl<data_t>(alloc, 0, src_layout, alloc, 0, dst_layout);
}

// transpose_2: transposed pairs (0,1) and (2,3)
//
// src (55,55,13,55):(715, 39325, 1, 13)            row-major order
// stride_order (1,0,3,2), slice [:]
// dst (55,55,13,55):(39325, 715, 55, 1)            strided order
// stride_order (0,1,2,3), slice [:]
TEST_CASE("nvmath transpose_2", "[nvmath][transpose][d2d]")
{
  using namespace cute;
  constexpr int alloc = 55 * 55 * 13 * 55;
  auto dst_layout     = make_layout(make_shape(55, 55, 13, 55), make_stride(39325, 715, 55, 1));
  auto src_layout     = make_layout(make_shape(55, 55, 13, 55), make_stride(715, 39325, 1, 13));
  test_impl<data_t>(alloc, 0, src_layout, alloc, 0, dst_layout);
}

// transpose_3: interleaved stride permutation
//
// src (101,101,101,101):(101, 1030301, 1, 10201)
// stride_order (1,3,0,2), slice [:]
// dst (101,101,101,101):(1030301, 10201, 101, 1)
// stride_order (0,1,2,3), slice [:]
TEST_CASE("nvmath transpose_3", "[nvmath][transpose][d2d]")
{
  using namespace cute;
  constexpr int alloc = 101 * 101 * 101 * 101;
  auto dst_layout     = make_layout(make_shape(101, 101, 101, 101), make_stride(1030301, 10201, 101, 1));
  auto src_layout     = make_layout(make_shape(101, 101, 101, 101), make_stride(101, 1030301, 1, 10201));
  test_impl<data_t>(alloc, 0, src_layout, alloc, 0, dst_layout);
}

// transpose_small_tile: fully reversed stride orders, small tile (7x7=49 elements)
//
// src (7,100019,7):(1, 7, 700133)
// stride_order (2,1,0), slice [:]
// dst (7,100019,7):(700133, 7, 1)
// stride_order (0,1,2), slice [:]
TEST_CASE("nvmath transpose_small_tile", "[nvmath][transpose][d2d]")
{
  using namespace cute;
  constexpr int alloc = 7 * 100019 * 7;
  auto dst_layout     = make_layout(make_shape(7, 100019, 7), make_stride(700133, 7, 1));
  auto src_layout     = make_layout(make_shape(7, 100019, 7), make_stride(1, 7, 700133));
  test_impl<data_t>(alloc, 0, src_layout, alloc, 0, dst_layout);
}

// transpose_small_tile_2: fully reversed stride orders
//
// src (33,100019,33):(1, 33, 3300627)
// stride_order (2,1,0), slice [:]
// dst (33,100019,33):(3300627, 33, 1)
// stride_order (0,1,2), slice [:]
TEST_CASE("nvmath transpose_small_tile_2", "[nvmath][transpose][d2d]")
{
  using namespace cute;
  constexpr int alloc = 33 * 100019 * 33;
  auto dst_layout     = make_layout(make_shape(33, 100019, 33), make_stride(3300627, 33, 1));
  auto src_layout     = make_layout(make_shape(33, 100019, 33), make_stride(1, 33, 3300627));
  test_impl<data_t>(alloc, 0, src_layout, alloc, 0, dst_layout);
}

// transpose_small_tile_3: only the 7x7 block transposed between src and dst
//
// src (7,100019,7):(1, 49, 7)
// stride_order (1,2,0), slice [:]
// dst (7,100019,7):(700133, 1, 100019)
// stride_order (0,2,1), slice [:]
TEST_CASE("nvmath transpose_small_tile_3", "[nvmath][transpose][d2d]")
{
  using namespace cute;
  constexpr int alloc = 7 * 100019 * 7;
  auto dst_layout     = make_layout(make_shape(7, 100019, 7), make_stride(700133, 1, 100019));
  auto src_layout     = make_layout(make_shape(7, 100019, 7), make_stride(1, 49, 7));
  test_impl<data_t>(alloc, 0, src_layout, alloc, 0, dst_layout);
}

// transpose_small_tile_4: only the 33x33 block transposed between src and dst
//
// src (33,100019,33):(1, 1089, 33)
// stride_order (1,2,0), slice [:]
// dst (33,100019,33):(3300627, 1, 100019)
// stride_order (0,2,1), slice [:]
TEST_CASE("nvmath transpose_small_tile_4", "[nvmath][transpose][d2d]")
{
  using namespace cute;
  constexpr int alloc = 33 * 100019 * 33;
  auto dst_layout     = make_layout(make_shape(33, 100019, 33), make_stride(3300627, 1, 100019));
  auto src_layout     = make_layout(make_shape(33, 100019, 33), make_stride(1, 1089, 33));
  test_impl<data_t>(alloc, 0, src_layout, alloc, 0, dst_layout);
}

// transpose_channels: shared least strided extent (3), complicates tiling
//
// src (10001,10001,3):(3, 30003, 1)
// stride_order (1,0,2), slice [:]
// dst (10001,10001,3):(30003, 3, 1)
// stride_order (0,1,2), slice [:]
TEST_CASE("nvmath transpose_channels", "[nvmath][transpose][d2d]")
{
  using namespace cute;
  constexpr int alloc = 10001 * 10001 * 3;
  auto dst_layout     = make_layout(make_shape(10001, 10001, 3), make_stride(30003, 3, 1));
  auto src_layout     = make_layout(make_shape(10001, 10001, 3), make_stride(3, 30003, 1));
  test_impl<data_t>(alloc, 0, src_layout, alloc, 0, dst_layout);
}

// transpose_channels_2: 5D with shared least strided extents
//
// src (1001,1001,3,3,3):(27, 27027, 9, 1, 3)
// stride_order (1,0,2,4,3), slice [:]
// dst (1001,1001,3,3,3):(27027, 27, 9, 3, 1)
// stride_order (0,1,2,3,4), slice [:]
TEST_CASE("nvmath transpose_channels_2", "[nvmath][transpose][d2d]")
{
  using namespace cute;
  constexpr int alloc = 1001 * 1001 * 3 * 3 * 3;
  auto dst_layout     = make_layout(make_shape(1001, 1001, 3, 3, 3), make_stride(27027, 27, 9, 3, 1));
  auto src_layout     = make_layout(make_shape(1001, 1001, 3, 3, 3), make_stride(27, 27027, 9, 1, 3));
  test_impl<data_t>(alloc, 0, src_layout, alloc, 0, dst_layout);
}

// transpose_inbalanced: 2D with highly imbalanced extents
// L2 cache pressure: large stride between rows in src exhausts cache
//
// src (3,1000033):(1, 3)
// stride_order (1,0), slice [:]
// dst (3,1000033):(1000033, 1)
// stride_order (0,1), slice [:]
TEST_CASE("nvmath transpose_inbalanced", "[nvmath][transpose][d2d]")
{
  using namespace cute;
  constexpr int alloc = 3 * 1000033;
  auto dst_layout     = make_layout(make_shape(3, 1000033), make_stride(1000033, 1));
  auto src_layout     = make_layout(make_shape(3, 1000033), make_stride(1, 3));
  test_impl<data_t>(alloc, 0, src_layout, alloc, 0, dst_layout);
}
