//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/stream>

#include <cuda/experimental/__copy_bytes/copy_bytes_naive.cuh>
#include <cuda/experimental/__copy_bytes/copy_bytes_registers.cuh>

#include "testing.cuh"
#include <cute/layout.hpp>

using data_t = int8_t;

static const cuda::stream stream{cuda::device_ref{0}};

// D2D copy test: same layout for src and dst, with optional pointer offset (e.g. negative strides)
template <typename T, typename Layout>
void test_impl(int alloc_size, int offset, const Layout& layout)
{
  namespace cudax = cuda::experimental;
  thrust::host_vector<T> h_src(alloc_size);
  for (int i = 0; i < alloc_size; ++i)
  {
    h_src[i] = static_cast<T>(i);
  }

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(alloc_size, T{0});
  auto* src_ptr = thrust::raw_pointer_cast(d_src.data()) + offset;
  auto* dst_ptr = thrust::raw_pointer_cast(d_dst.data()) + offset;

  d_dst.assign(alloc_size, T{0});
  cudax::copy_bytes_naive(src_ptr, layout, dst_ptr, layout, stream);
  stream.sync();
  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == h_src);

  d_dst.assign(alloc_size, T{0});
  cudax::copy_bytes_registers(src_ptr, layout, dst_ptr, layout, stream);
  stream.sync();
  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == h_src);
}

// D2D copy test: different src/dst allocations, offsets, and layouts (for sliced cases)
template <typename T, typename SrcLayout, typename DstLayout>
void test_impl(
  int src_alloc, int src_offset, const SrcLayout& src_layout, int dst_alloc, int dst_offset, const DstLayout& dst_layout)
{
  namespace cudax = cuda::experimental;
  thrust::host_vector<T> h_src(src_alloc);
  for (int i = 0; i < src_alloc; ++i)
  {
    h_src[i] = static_cast<T>(i);
  }

  thrust::host_vector<T> expected(dst_alloc, T{0});
  const int num_items = static_cast<int>(cute::size(src_layout));
  for (int i = 0; i < num_items; ++i)
  {
    expected[dst_offset + dst_layout(i)] = h_src[src_offset + src_layout(i)];
  }

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(dst_alloc, T{0});
  auto* src_ptr = thrust::raw_pointer_cast(d_src.data()) + src_offset;
  auto* dst_ptr = thrust::raw_pointer_cast(d_dst.data()) + dst_offset;

  d_dst.assign(dst_alloc, T{0});
  cudax::copy_bytes_naive(src_ptr, src_layout, dst_ptr, dst_layout, stream);
  stream.sync();
  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == expected);

  d_dst.assign(dst_alloc, T{0});
  cudax::copy_bytes_registers(src_ptr, src_layout, dst_ptr, dst_layout, stream);
  stream.sync();
  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == expected);
}

/***********************************************************************************************************************
 * nvmath memcpy test cases (device-to-device)
 * Extracted from TensorCopy bench cases — first five memcpy cases
 **********************************************************************************************************************/

// memcpy_layout_0: simple C-order memcopy
// shape (70,90,80,80), stride_order (0,1,2,3), slice [:]
// strides: (90*80*80, 80*80, 80, 1)
TEST_CASE("nvmath memcpy_layout_0", "[nvmath][memcpy][d2d]")
{
  using namespace cute;
  constexpr int alloc = 70 * 90 * 80 * 80;
  auto layout         = make_layout(make_shape(70, 90, 80, 80), make_stride(576000, 6400, 80, 1));
  test_impl<data_t>(alloc, 0, layout);
}

// memcpy_layout_1: not C nor F, but the same stride order for src and dst
// shape (70,90,80,80), stride_order (3,2,0,1), slice [:]
// strides: (90, 1, 6300, 504000)
TEST_CASE("nvmath memcpy_layout_1", "[nvmath][memcpy][d2d]")
{
  using namespace cute;
  constexpr int alloc = 70 * 90 * 80 * 80;
  auto layout         = make_layout(make_shape(70, 90, 80, 80), make_stride(90, 1, 6300, 504000));
  test_impl<data_t>(alloc, 0, layout);
}

// memcpy_layout_2: the same stride order, extent=5 is sliced but it has the biggest stride
//                  so the copied slice is still contiguous
// dst_base (1001,1007,5,31) stride_order (2,0,3,1), dst_slice [:,:,1:4]
// src_base (1001,1007,3,31) stride_order (2,0,3,1), src_slice [:]
// Both strides: (31217, 1, 31248217, 1007)
// Copy shape: (1001,1007,3,31)
TEST_CASE("nvmath memcpy_layout_2", "[nvmath][memcpy][d2d]")
{
  using namespace cute;
  constexpr int src_alloc  = 1001 * 1007 * 3 * 31;
  constexpr int dst_alloc  = 1001 * 1007 * 5 * 31;
  constexpr int dst_offset = 31248217;
  auto copy_layout         = make_layout(make_shape(1001, 1007, 3, 31), make_stride(31217, 1, 31248217, 1007));
  test_impl<data_t>(src_alloc, 0, copy_layout, dst_alloc, dst_offset, copy_layout);
}

// memcpy_layout_3: the two most strided extents are sliced to 1, different stride orders ignored
// dst_base (57,71,5,1007,1) stride_order (4,2,0,3,1), dst_slice [:,:,3:4]
//   dst strides: (71497, 1, 4075329, 71, 20376645)
// src_base (57,71,3,1007,3) stride_order (2,4,0,3,1), src_slice [:,:,1:2,:,1:2]
//   src strides: (71497, 1, 12225987, 71, 4075329)
// Copy shape: (57,71,1,1007,1) — effective 3D layout (57,71,1007) with strides (71497,1,71)
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
// base shape (63,70,1001), stride_order (1,0,2)
// base strides: (1001, 63063, 1)
// slice [:,:,::-1] → strides (1001, 63063, -1), ptr offset = 1000
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
// dst_base (8,100019,4) stride_order (2,1,0), strides (1, 8, 800152)
// src_base (8,100019,11) stride_order (0,1,2), strides (1100209, 11, 1)
//   src_slice [:,:,::3] → stride[2] *= 3, shape[2] = 4
//   src view strides: (1100209, 11, 3), shape (8, 100019, 4)
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
// dst_base (63,70,1001) stride_order (0,1,2), strides (70070, 1001, 1)
// src_base (63,70,1001) stride_order (1,0,2), base strides (1001, 63063, 1)
//   src_slice [::-1,::-1,::-1] → all strides negated, ptr offset = 4414409
//   src view strides: (-1001, -63063, -1), shape (63, 70, 1001)
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

// flatten_common: 23-dim tensor with shape (2,)*23, stride orders differ only in dims 20-22
// Dims 0-19 share the same strides → common parts can be flattened
// dst stride_order: (7,6,5,4,3,2,1,0,8,9,10,11,12,13,14,15,16,17,18,19,21,20,22)
// src stride_order: (7,6,5,4,3,2,1,0,8,9,10,11,12,13,14,15,16,17,18,19,20,22,21)
TEST_CASE("nvmath flatten_common", "[nvmath][flatten][d2d]")
{
  using namespace cute;
  constexpr int alloc = 1 << 23;
  // clang-format off
  auto shape = make_shape(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
  auto dst_layout = make_layout(shape,
    make_stride(1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22,
                1 << 14, 1 << 13, 1 << 12, 1 << 11, 1 << 10, 1 <<  9, 1 <<  8, 1 <<  7,
                1 <<  6, 1 <<  5, 1 <<  4, 1 <<  3,
                1 <<  1, 1 <<  2, 1 <<  0));
  auto src_layout = make_layout(shape,
    make_stride(1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22,
                1 << 14, 1 << 13, 1 << 12, 1 << 11, 1 << 10, 1 <<  9, 1 <<  8, 1 <<  7,
                1 <<  6, 1 <<  5, 1 <<  4, 1 <<  3,
                1 <<  2, 1 <<  0, 1 <<  1));
  // clang-format on
  test_impl<data_t>(alloc, 0, src_layout, alloc, 0, dst_layout);
}

// flatten_one: 20-dim tensor, dst C-order, src F-order (fully reversed strides)
// No common contiguous parts to flatten, but one tensor can be flattened to 1D
// dst_base (4,2,...,2) stride_order (0,1,...,19) — C-order
// src_base (16,2,...,2) stride_order (19,18,...,0) — F-order
//   src_slice [::5] → stride[0] *= 5, shape[0] = 4
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

// sliced_vec: src is sliced along dim 1 (skip first row), gaps between elements prevent memcpy
// but vectorized accesses still apply
// dst_base (35,255,10,24) C-order, slice [:]
// src_base (35,256,10,24) C-order, slice [:, 1:, :, :] -> shape (35,255,10,24), offset = 240
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

// sliced_vec_2: least strided extent is odd (3), but flattened 4x3=12 is even, so vectorizable
// dst_base (355,255,4,3) C-order, slice [:]
// src_base (355,256,4,3) C-order, slice [:, 1:, :, :] -> shape (355,255,4,3), offset = 12
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

// sliced_unaligned_ptr: base pointer is misaligned due to slicing (offset=205, 205%16!=0),
// preventing simple vectorization
// dst_base (35,255,5,10) C-order, slice [:]
// src_base (35,255,30,20) C-order, slice [:, :, 10:-15, 5:-5] -> shape (35,255,5,10), offset = 205
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
