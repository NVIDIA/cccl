//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "copy_bytes_common.cuh"
#include <cute/layout.hpp>

// 2D transpose: column-major source → row-major destination.
// Shape (32, 32) ensures tile sizes (32, 32) evenly divide and the
// tile product (1024) exceeds the shared-memory kernel threshold (256).
// After sort_by_stride_paired, dst is stride-1 at mode 0 while src is not,
// triggering the copy_bytes_shared_mem path.
TEST_CASE("shared_memory 2D transpose", "[shared_memory][2d]")
{
  using namespace cute;
  constexpr int N     = 32;
  constexpr int alloc = N * N;
  auto src_layout     = make_layout(make_shape(N, N), make_stride(1, N));
  auto dst_layout     = make_layout(make_shape(N, N), make_stride(N, 1));
  test_impl<int>(alloc, 0, src_layout, alloc, 0, dst_layout);
}

// 2D transpose with partial tiles: extents (50, 37) are not divisible by tile size 32.
// Full-tile blocks cover [0,32)x[0,32). Boundary blocks handle the remainder.
TEST_CASE("shared_memory 2D partial tiles", "[shared_memory][2d][partial]")
{
  using namespace cute;
  constexpr int M     = 50;
  constexpr int N     = 37;
  constexpr int alloc = M * N;
  auto src_layout     = make_layout(make_shape(M, N), make_stride(1, M));
  auto dst_layout     = make_layout(make_shape(M, N), make_stride(N, 1));
  test_impl<int>(alloc, 0, src_layout, alloc, 0, dst_layout);
}
