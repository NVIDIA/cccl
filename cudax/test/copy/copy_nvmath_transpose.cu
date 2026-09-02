//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/detail/__config>

// GCC -Warray-bounds false positive for high-rank (20+) __raw_tensor instantiations
#if _CCCL_COMPILER(GCC)
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Warray-bounds")
#endif // _CCCL_COMPILER(GCC)

#include <cuda/std/cstdint>

#include "copy_common.cuh"

using data_t = int8_t;

/***********************************************************************************************************************
 * nvmath transpose test cases (device-to-device)
 **********************************************************************************************************************/

// src: (2,)^20:(2^19,2^18,...,2^0), C-order
// dst: (2,)^20:(2^0,2^1,...,2^19), F-order
TEST_CASE("copy d2d nvmath transpose_merge_extents", "[copy][d2d][nvmath][transpose]")
{
  constexpr int alloc = 1 << 20;
  cuda::std::array<int, 20> shape{};
  for (auto& s : shape)
  {
    s = 2;
  }
  // clang-format off
  cuda::std::array<int, 20> src_strides{
    1 << 19, 1 << 18, 1 << 17, 1 << 16, 1 << 15, 1 << 14, 1 << 13, 1 << 12,
    1 << 11, 1 << 10, 1 << 9,  1 << 8,  1 << 7,  1 << 6,  1 << 5,  1 << 4,
    1 << 3,  1 << 2,  1 << 1,  1 << 0};
  cuda::std::array<int, 20> dst_strides{
    1 << 0,  1 << 1,  1 << 2,  1 << 3,  1 << 4,  1 << 5,  1 << 6,  1 << 7,
    1 << 8,  1 << 9,  1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15,
    1 << 16, 1 << 17, 1 << 18, 1 << 19};
  // clang-format on
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (55,13,55,55):(13,1,39325,715)
// dst: (55,13,55,55):(39325,3025,55,1)
TEST_CASE("copy d2d nvmath transpose", "[copy][d2d][nvmath][transpose]")
{
  constexpr int alloc = 55 * 13 * 55 * 55;
  cuda::std::array<int, 4> shape{55, 13, 55, 55};
  cuda::std::array<int, 4> src_strides{13, 1, 39325, 715};
  cuda::std::array<int, 4> dst_strides{39325, 3025, 55, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (55,55,13,55):(715,39325,1,13)
// dst: (55,55,13,55):(39325,715,55,1)
TEST_CASE("copy d2d nvmath transpose_2", "[copy][d2d][nvmath][transpose]")
{
  constexpr int alloc = 55 * 55 * 13 * 55;
  cuda::std::array<int, 4> shape{55, 55, 13, 55};
  cuda::std::array<int, 4> src_strides{715, 39325, 1, 13};
  cuda::std::array<int, 4> dst_strides{39325, 715, 55, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (101,101,101,101):(101,1030301,1,10201)
// dst: (101,101,101,101):(1030301,10201,101,1)
TEST_CASE("copy d2d nvmath transpose_3", "[copy][d2d][nvmath][transpose]")
{
  constexpr int alloc = 101 * 101 * 101 * 101;
  cuda::std::array<int, 4> shape{101, 101, 101, 101};
  cuda::std::array<int, 4> src_strides{101, 1030301, 1, 10201};
  cuda::std::array<int, 4> dst_strides{1030301, 10201, 101, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (7,100019,7):(1,7,700133)
// dst: (7,100019,7):(700133,7,1)
TEST_CASE("copy d2d nvmath transpose_small_tile", "[copy][d2d][nvmath][transpose]")
{
  constexpr int alloc = 7 * 100019 * 7;
  cuda::std::array<int, 3> shape{7, 100019, 7};
  cuda::std::array<int, 3> src_strides{1, 7, 700133};
  cuda::std::array<int, 3> dst_strides{700133, 7, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (33,100019,33):(1,33,3300627)
// dst: (33,100019,33):(3300627,33,1)
TEST_CASE("copy d2d nvmath transpose_small_tile_2", "[copy][d2d][nvmath][transpose]")
{
  constexpr int alloc = 33 * 100019 * 33;
  cuda::std::array<int, 3> shape{33, 100019, 33};
  cuda::std::array<int, 3> src_strides{1, 33, 3300627};
  cuda::std::array<int, 3> dst_strides{3300627, 33, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (7,100019,7):(1,49,7)
// dst: (7,100019,7):(700133,1,100019)
TEST_CASE("copy d2d nvmath transpose_small_tile_3", "[copy][d2d][nvmath][transpose]")
{
  constexpr int alloc = 7 * 100019 * 7;
  cuda::std::array<int, 3> shape{7, 100019, 7};
  cuda::std::array<int, 3> src_strides{1, 49, 7};
  cuda::std::array<int, 3> dst_strides{700133, 1, 100019};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (33,100019,33):(1,1089,33)
// dst: (33,100019,33):(3300627,1,100019)
TEST_CASE("copy d2d nvmath transpose_small_tile_4", "[copy][d2d][nvmath][transpose]")
{
  constexpr int alloc = 33 * 100019 * 33;
  cuda::std::array<int, 3> shape{33, 100019, 33};
  cuda::std::array<int, 3> src_strides{1, 1089, 33};
  cuda::std::array<int, 3> dst_strides{3300627, 1, 100019};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (10001,10001,3):(3,30003,1)
// dst: (10001,10001,3):(30003,3,1)
TEST_CASE("copy d2d nvmath transpose_channels", "[copy][d2d][nvmath][transpose]")
{
  constexpr int alloc = 10001 * 10001 * 3;
  cuda::std::array<int, 3> shape{10001, 10001, 3};
  cuda::std::array<int, 3> src_strides{3, 30003, 1};
  cuda::std::array<int, 3> dst_strides{30003, 3, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (1001,1001,3,3,3):(27,27027,9,1,3)
// dst: (1001,1001,3,3,3):(27027,27,9,3,1)
TEST_CASE("copy d2d nvmath transpose_channels_2", "[copy][d2d][nvmath][transpose]")
{
  constexpr int alloc = 1001 * 1001 * 3 * 3 * 3;
  cuda::std::array<int, 5> shape{1001, 1001, 3, 3, 3};
  cuda::std::array<int, 5> src_strides{27, 27027, 9, 1, 3};
  cuda::std::array<int, 5> dst_strides{27027, 27, 9, 3, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (3,1000033):(1,3)
// dst: (3,1000033):(1000033,1)
TEST_CASE("copy d2d nvmath transpose_inbalanced", "[copy][d2d][nvmath][transpose]")
{
  constexpr int alloc = 3 * 1000033;
  cuda::std::array<int, 2> shape{3, 1000033};
  cuda::std::array<int, 2> src_strides{1, 3};
  cuda::std::array<int, 2> dst_strides{1000033, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (4,4,4,4,4,4,16,8):(1,4,...,65536), column-major
// dst: (4,4,4,4,4,4,16,8):(131072,...,8,1), row-major
// Rank 8 == __max_shared_mem_kernel_rank, verifying the shared-memory kernel is still instantiated at the maximum
// allowed rank. The first 6 dimensions fit in one tile (4^6 = 4096 elements); the last 2 dimensions (16x8 = 128 tiles)
// provide sufficient grid utilization.
TEST_CASE("copy d2d nvmath transpose_max_shared_mem_rank", "[copy][d2d][nvmath][transpose]")
{
  constexpr int alloc = 4 * 4 * 4 * 4 * 4 * 4 * 16 * 8; // 524288
  cuda::std::array<int, 8> shape{4, 4, 4, 4, 4, 4, 16, 8};
  cuda::std::array<int, 8> src_strides{1, 4, 16, 64, 256, 1024, 4096, 65536};
  cuda::std::array<int, 8> dst_strides{131072, 32768, 8192, 2048, 512, 128, 8, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}
