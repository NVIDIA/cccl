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
 * nvmath memcpy test cases (device-to-device)
 **********************************************************************************************************************/

// memcpy_layout_0: simple C-order memcopy
//
// shape (70,90,80,80):(90*80*80, 80*80, 80, 1)
// stride_order (0,1,2,3), slice [:]
TEST_CASE("nvmath memcpy_layout_0", "[nvmath][memcpy][d2d]")
{
  using namespace cute;
  constexpr int alloc = 70 * 90 * 80 * 80;
  auto layout         = make_layout(make_shape(70, 90, 80, 80), make_stride(576000, 6400, 80, 1));
  test_impl<data_t>(alloc, 0, layout);
}

// memcpy_layout_1: same stride order for src and dst, neither column- nor row-major order
//
// shape (70, 90, 80, 80):(90, 1, 6300, 504000)
// stride_order (3,2,0,1), slice [:]
TEST_CASE("nvmath memcpy_layout_1", "[nvmath][memcpy][d2d]")
{
  using namespace cute;
  constexpr int alloc = 70 * 90 * 80 * 80;
  auto layout         = make_layout(make_shape(70, 90, 80, 80), make_stride(90, 1, 6300, 504000));
  test_impl<data_t>(alloc, 0, layout);
}

// memcpy_layout_2: same stride order, sliced extent has largest stride so copy is contiguous
//
// src (1001, 1007, 3, 31):(31217, 1, 31248217, 1007)
// stride_order (2, 0, 3, 1), slice [:]
// dst (1001, 1007, 5, 31):(31217, 1, 31248217, 1007)
// stride_order (2, 0, 3, 1), slice [:,:,1:4]
TEST_CASE("nvmath memcpy_layout_2", "[nvmath][memcpy][d2d]")
{
  using namespace cute;
  constexpr int src_alloc  = 1001 * 1007 * 3 * 31;
  constexpr int dst_alloc  = 1001 * 1007 * 5 * 31;
  constexpr int dst_offset = 31248217;
  auto copy_layout         = make_layout(make_shape(1001, 1007, 3, 31), make_stride(31217, 1, 31248217, 1007));
  test_impl<data_t>(src_alloc, 0, copy_layout, dst_alloc, dst_offset, copy_layout);
}

// memcpy_layout_3: two most strided extents sliced to 1, different stride orders
//
// src (57,71,3,1007,3):(71497, 1, 12225987, 71, 4075329)
// stride_order (2,4,0,3,1), slice [:,:,1:2,:,1:2]
// dst (57,71,5,1007,1):(71497, 1, 4075329, 71, 20376645)
// stride_order (4,2,0,3,1), slice [:,:,3:4]
TEST_CASE("nvmath memcpy_layout_3", "[nvmath][memcpy][d2d]")
{
  using namespace cute;
  constexpr int src_alloc  = 57 * 71 * 3 * 1007 * 3;
  constexpr int src_offset = 12225987 + 4075329;
  auto src_layout          = make_layout(make_shape(57, 71, 1, 1007, 1), make_stride(71497, 1, 12225987, 71, 4075329));

  constexpr int dst_alloc  = 57 * 71 * 5 * 1007 * 1;
  constexpr int dst_offset = 3 * 4075329;
  auto dst_layout          = make_layout(make_shape(57, 71, 1, 1007, 1), make_stride(71497, 1, 4075329, 71, 20376645));

  test_impl<data_t>(src_alloc, src_offset, src_layout, dst_alloc, dst_offset, dst_layout);
}

// memcpy_neg: densely packed tensors with identical negative strides
//
// shape (63,70,1001):(1001, 63063, 1)
// stride_order (1,0,2), slice [:,:,::-1]
TEST_CASE("nvmath memcpy_neg", "[nvmath][memcpy][d2d]")
{
  using namespace cute;
  constexpr int alloc  = 63 * 70 * 1001;
  constexpr int offset = 1000;
  auto layout          = make_layout(make_shape(63, 70, 1001), make_stride(1001, 63063, -1));
  test_impl<data_t>(alloc, offset, layout);
}

/***********************************************************************************************************************
 * nvmath reorder strides test case (device-to-device)
 **********************************************************************************************************************/

// reorder_strides: src and dst have different stride orders, testing loop reordering
//
// src (8,100019,11):(1100209, 11, 1)
// stride_order (0,1,2), slice [:,:,::3]
// dst (8,100019,4):(1, 8, 800152)
// stride_order (2,1,0), slice [:]
TEST_CASE("nvmath reorder_strides", "[nvmath][reorder_strides][d2d]")
{
  using namespace cute;
  constexpr int dst_alloc = 8 * 100019 * 4;
  auto dst_layout         = make_layout(make_shape(8, 100019, 4), make_stride(1, 8, 800152));

  constexpr int src_alloc = 8 * 100019 * 11;
  auto src_layout         = make_layout(make_shape(8, 100019, 4), make_stride(1100209, 11, 3));

  test_impl<data_t>(src_alloc, 0, src_layout, dst_alloc, 0, dst_layout);
}

/***********************************************************************************************************************
 * nvmath negative strides test case (device-to-device)
 **********************************************************************************************************************/

// src_neg_stride: dst is C-order, src has all dimensions reversed
//
// src (63,70,1001):(1001, 63063, 1)
// stride_order (1,0,2), slice [::-1,::-1,::-1]
// dst (63,70,1001):(70070, 1001, 1)
// stride_order (0,1,2), slice [:]
TEST_CASE("nvmath src_neg_stride", "[nvmath][neg_stride][d2d]")
{
  using namespace cute;
  constexpr int alloc      = 63 * 70 * 1001;
  constexpr int src_offset = alloc - 1;
  auto src_layout          = make_layout(make_shape(63, 70, 1001), make_stride(-1001, -63063, -1));
  auto dst_layout          = make_layout(make_shape(63, 70, 1001), make_stride(70070, 1001, 1));

  test_impl<data_t>(alloc, src_offset, src_layout, alloc, 0, dst_layout);
}

/***********************************************************************************************************************
 * nvmath squeezing and flattening test cases (device-to-device)
 * High-dimensional layouts testing squeeze/flatten optimizations.
 **********************************************************************************************************************/

// flatten_common: 23-dim (2,)^23 tensor, stride orders differ only in dims 20-22
// Dims 0-19 share the same strides → common parts can be flattened
//
// shape (2,)^23, strides as bit-shift permutations (see code)
// src stride_order (7,6,5,4,3,2,1,0,8,9,10,11,12,13,14,15,16,17,18,19,20,22,21), slice [:]
// dst stride_order (7,6,5,4,3,2,1,0,8,9,10,11,12,13,14,15,16,17,18,19,21,20,22), slice [:]
TEST_CASE("nvmath flatten_common", "[nvmath][flatten][d2d]")
{
  using namespace cute;
  constexpr int alloc = 1 << 23;
  auto shape          = make_shape(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
  auto dst_layout     = make_layout(
    shape,
    make_stride(
      1 << 15,
      1 << 16,
      1 << 17,
      1 << 18,
      1 << 19,
      1 << 20,
      1 << 21,
      1 << 22,
      1 << 14,
      1 << 13,
      1 << 12,
      1 << 11,
      1 << 10,
      1 << 9,
      1 << 8,
      1 << 7,
      1 << 6,
      1 << 5,
      1 << 4,
      1 << 3,
      1 << 1,
      1 << 2,
      1 << 0));
  auto src_layout = make_layout(
    shape,
    make_stride(
      1 << 15,
      1 << 16,
      1 << 17,
      1 << 18,
      1 << 19,
      1 << 20,
      1 << 21,
      1 << 22,
      1 << 14,
      1 << 13,
      1 << 12,
      1 << 11,
      1 << 10,
      1 << 9,
      1 << 8,
      1 << 7,
      1 << 6,
      1 << 5,
      1 << 4,
      1 << 3,
      1 << 2,
      1 << 0,
      1 << 1));
  test_impl<data_t>(alloc, 0, src_layout, alloc, 0, dst_layout);
}

// flatten_one: 20-dim tensor, no common contiguous parts, one tensor flattenable to 1D
// No common contiguous parts to flatten, but one tensor can be flattened to 1D
// dst column-order, src row-order (fully reversed strides)
//
// src (16,2,...,2):(1,2^4,...,2^22)
// stride_order (19,18,...,0), slice [::5]
// dst (4,2,...,2):(2^19,2^18,...,1)
// stride_order (0,1,...,19), slice [:]
TEST_CASE("nvmath flatten_one", "[nvmath][flatten][d2d]")
{
  using namespace cute;
  constexpr int dst_alloc = 1 << 21;
  constexpr int src_alloc = 1 << 23;
  auto dst_layout         = make_layout(
    make_shape(4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
    make_stride(
      1 << 19,
      1 << 18,
      1 << 17,
      1 << 16,
      1 << 15,
      1 << 14,
      1 << 13,
      1 << 12,
      1 << 11,
      1 << 10,
      1 << 9,
      1 << 8,
      1 << 7,
      1 << 6,
      1 << 5,
      1 << 4,
      1 << 3,
      1 << 2,
      1 << 1,
      1 << 0));
  auto src_layout = make_layout(
    make_shape(4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
    make_stride(
      5,
      1 << 4,
      1 << 5,
      1 << 6,
      1 << 7,
      1 << 8,
      1 << 9,
      1 << 10,
      1 << 11,
      1 << 12,
      1 << 13,
      1 << 14,
      1 << 15,
      1 << 16,
      1 << 17,
      1 << 18,
      1 << 19,
      1 << 20,
      1 << 21,
      1 << 22));
  test_impl<data_t>(src_alloc, 0, src_layout, dst_alloc, 0, dst_layout);
}

/***********************************************************************************************************************
 * nvmath vectorize test cases (device-to-device)
 * Sliced tensors that benefit from vectorized loads/stores via layout manipulation.
 **********************************************************************************************************************/

// sliced_vec: src sliced along dim 1, vectorized accesses still apply
//
// src (35,256,10,24):(61440, 240, 24, 1)
// stride_order (0,1,2,3), slice [:,1:,:,:]
// dst (35,255,10,24):(61200, 240, 24, 1)
// stride_order (0,1,2,3), slice [:]
TEST_CASE("nvmath sliced_vec", "[nvmath][vectorize][d2d]")
{
  using namespace cute;
  constexpr int dst_alloc  = 35 * 255 * 10 * 24;
  constexpr int src_alloc  = 35 * 256 * 10 * 24;
  constexpr int src_offset = 240;
  auto dst_layout          = make_layout(make_shape(35, 255, 10, 24), make_stride(61200, 240, 24, 1));
  auto src_layout          = make_layout(make_shape(35, 255, 10, 24), make_stride(61440, 240, 24, 1));
  test_impl<data_t>(src_alloc, src_offset, src_layout, dst_alloc, 0, dst_layout);
}

// sliced_vec_2: least strided extent is odd (3), but flattened 4x3=12 is vectorizable
//
// src (355,256,4,3):(3072, 12, 3, 1)            row-major order
// stride_order (0,1,2,3), slice [:,1:,:,:]
// dst (355,255,4,3):(3060, 12, 3, 1)            row-major order
// stride_order (0,1,2,3), slice [:]
TEST_CASE("nvmath sliced_vec_2", "[nvmath][vectorize][d2d]")
{
  using namespace cute;
  constexpr int dst_alloc  = 355 * 255 * 4 * 3;
  constexpr int src_alloc  = 355 * 256 * 4 * 3;
  constexpr int src_offset = 12;
  auto dst_layout          = make_layout(make_shape(355, 255, 4, 3), make_stride(3060, 12, 3, 1));
  auto src_layout          = make_layout(make_shape(355, 255, 4, 3), make_stride(3072, 12, 3, 1));
  test_impl<data_t>(src_alloc, src_offset, src_layout, dst_alloc, 0, dst_layout);
}

// sliced_unaligned_ptr: misaligned pointer due to slicing (offset=205, 205%16!=0)
//
// src (35,255,30,20):(153000, 600, 20, 1)            row-major order
// stride_order (0,1,2,3), slice [:,:,10:-15,5:-5]
// dst (35,255,5,10):(12750, 50, 10, 1)            row-major order
// stride_order (0,1,2,3), slice [:]
TEST_CASE("nvmath sliced_unaligned_ptr", "[nvmath][vectorize][d2d]")
{
  using namespace cute;
  constexpr int dst_alloc  = 35 * 255 * 5 * 10;
  constexpr int src_alloc  = 35 * 255 * 30 * 20;
  constexpr int src_offset = 10 * 20 + 5;
  auto dst_layout          = make_layout(make_shape(35, 255, 5, 10), make_stride(12750, 50, 10, 1));
  auto src_layout          = make_layout(make_shape(35, 255, 5, 10), make_stride(153000, 600, 20, 1));
  test_impl<data_t>(src_alloc, src_offset, src_layout, dst_alloc, 0, dst_layout);
}
