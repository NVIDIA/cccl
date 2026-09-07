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

using data_t = int;

/***********************************************************************************************************************
 * Shared-memory tiled transpose test cases (device-to-device)
 **********************************************************************************************************************/

// src: (8192,32):(1,8192), column-major
// dst: (8192,32):(32,1), row-major
// Shape (8192,32) creates enough 32x32 tiles to satisfy the one-wave occupancy heuristic.
TEST_CASE("copy d2d shared_memory 2D transpose", "[copy][d2d][shared_memory][transpose]")
{
  constexpr int M     = 8192;
  constexpr int N     = 32;
  constexpr int alloc = M * N;
  cuda::std::array<int, 2> shape{M, N};
  cuda::std::array<int, 2> src_strides{1, M};
  cuda::std::array<int, 2> dst_strides{N, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (8193,37):(1,8193), column-major
// dst: (8193,37):(37,1), row-major
// Extents (8193,37) are not divisible by tile size 32.
// Boundary blocks handle the remainder.
TEST_CASE("copy d2d shared_memory 2D partial tiles", "[copy][d2d][shared_memory][transpose][partial]")
{
  constexpr int M     = 8193;
  constexpr int N     = 37;
  constexpr int alloc = M * N;
  cuda::std::array<int, 2> shape{M, N};
  cuda::std::array<int, 2> src_strides{1, M};
  cuda::std::array<int, 2> dst_strides{N, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (8192,16,16):(1,8192,131072), column-major
// dst: (8192,16,16):(256,16,1), row-major
// The simplified tensor rank remains 3, covering the generic shared-memory launch.
TEST_CASE("copy d2d shared_memory 3D transpose", "[copy][d2d][shared_memory][transpose][3d]")
{
  constexpr int D0    = 8192;
  constexpr int D1    = 16;
  constexpr int D2    = 16;
  constexpr int alloc = D0 * D1 * D2;
  cuda::std::array<int, 3> shape{D0, D1, D2};
  cuda::std::array<int, 3> src_strides{1, D0, D0 * D1};
  cuda::std::array<int, 3> dst_strides{D1 * D2, D2, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (8193,16,16):(1,8193,131088), column-major
// dst: (8193,16,16):(256,16,1), row-major
// The first dimension is not tile-aligned, so rank-3 boundary tiles use the direct-copy fallback.
TEST_CASE("copy d2d shared_memory 3D partial tiles", "[copy][d2d][shared_memory][transpose][3d][partial]")
{
  constexpr int D0    = 8193;
  constexpr int D1    = 16;
  constexpr int D2    = 16;
  constexpr int alloc = D0 * D1 * D2;
  cuda::std::array<int, 3> shape{D0, D1, D2};
  cuda::std::array<int, 3> src_strides{1, D0, D0 * D1};
  cuda::std::array<int, 3> dst_strides{D1 * D2, D2, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (16,8192,8):(1,128,16)
// dst: (16,8192,8):(131072,16,1), padded in the middle dimension.
// This mirrors the padded small-dimension benchmark shape at unit-test scale.
TEST_CASE("copy d2d shared_memory 3D padded small dimension", "[copy][d2d][shared_memory][transpose][3d][padded]")
{
  constexpr int D0        = 16;
  constexpr int D1        = 8192;
  constexpr int D2        = 8;
  constexpr int dst_pitch = 16;
  constexpr int src_alloc = D0 * D1 * D2;
  constexpr int dst_alloc = D0 * D1 * dst_pitch;
  cuda::std::array<int, 3> shape{D0, D1, D2};
  cuda::std::array<int, 3> src_strides{1, D0 * D2, D0};
  cuda::std::array<int, 3> dst_strides{D1 * dst_pitch, dst_pitch, 1};
  test_copy_stride_relaxed<data_t>(src_alloc, 0, shape, src_strides, dst_alloc, 0, dst_strides);
}
