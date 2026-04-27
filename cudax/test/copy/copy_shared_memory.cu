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

// src: (32,32):(1,32), column-major
// dst: (32,32):(32,1), row-major
// Shape (32,32) ensures tile sizes (32,32) evenly divide and the tile product (1024) exceeds the
// shared-memory kernel threshold (256). After sort_by_stride_paired, dst is stride-1 at mode 0
// while src is not, triggering the shared-memory transpose path.
TEST_CASE("copy d2d shared_memory 2D transpose", "[copy][d2d][shared_memory][transpose]")
{
  constexpr int N     = 32;
  constexpr int alloc = N * N;
  cuda::std::array<int, 2> shape{N, N};
  cuda::std::array<int, 2> src_strides{1, N};
  cuda::std::array<int, 2> dst_strides{N, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}

// src: (50,37):(1,50), column-major
// dst: (50,37):(37,1), row-major
// Extents (50,37) are not divisible by tile size 32.
// Full-tile blocks cover [0,32)x[0,32). Boundary blocks handle the remainder.
TEST_CASE("copy d2d shared_memory 2D partial tiles", "[copy][d2d][shared_memory][transpose][partial]")
{
  constexpr int M     = 50;
  constexpr int N     = 37;
  constexpr int alloc = M * N;
  cuda::std::array<int, 2> shape{M, N};
  cuda::std::array<int, 2> src_strides{1, M};
  cuda::std::array<int, 2> dst_strides{N, 1};
  test_copy_stride_relaxed<data_t>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
}
