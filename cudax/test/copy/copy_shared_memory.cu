//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__copy/copy_shared_memory_utils.cuh>

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

TEMPLATE_TEST_CASE(
  "copy d2d shared_memory 2D small-element transpose", "[copy][d2d][shared_memory][transpose][small]", char, short)
{
  SECTION("column-major to row-major")
  {
    constexpr int M     = 8192;
    constexpr int N     = 32;
    constexpr int alloc = M * N;
    cuda::std::array<int, 2> shape{M, N};
    cuda::std::array<int, 2> src_strides{1, M};
    cuda::std::array<int, 2> dst_strides{N, 1};
    test_copy_stride_relaxed<TestType>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
  }

  SECTION("partial column-major to row-major")
  {
    constexpr int M     = 8193;
    constexpr int N     = 37;
    constexpr int alloc = M * N;
    cuda::std::array<int, 2> shape{M, N};
    cuda::std::array<int, 2> src_strides{1, M};
    cuda::std::array<int, 2> dst_strides{N, 1};
    test_copy_stride_relaxed<TestType>(alloc, 0, shape, src_strides, alloc, 0, dst_strides);
  }
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

// src: (33,257,33):(1,33,8481), column-major
// dst: (33,257,33):(8481,33,1), row-major
// Greedy 32x32x32 tiling requires 128 KiB, so devices with a smaller per-block limit use the logical 2D fallback.
TEST_CASE("copy d2d shared_memory 3D logical 2D fallback", "[copy][d2d][shared_memory][transpose][3d][fallback]")
{
  constexpr int D0    = 33;
  constexpr int D1    = 257;
  constexpr int D2    = 33;
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

//----------------------------------------------------------------------------------------------------------------------
// internal utilities

TEST_CASE("copy shared_memory tiling preserves greedy candidate", "[copy][shared_memory][tiling]")
{
  using raw_tensor_t = cuda::experimental::__raw_tensor<int, int, data_t, 3>;

  constexpr cuda::std::array<int, 3> shape{1024, 1024, 1024};
  const raw_tensor_t src{nullptr, 3, shape, {1, 1024, 1024 * 1024}};
  const raw_tensor_t dst{nullptr, 3, shape, {1024 * 1024, 1024, 1}};

  constexpr cuda::std::size_t max_shared_mem_bytes = 128 * 1024;
  constexpr cuda::std::size_t num_sms              = 1;
  const auto result =
    cuda::experimental::__find_shared_mem_tiling_with_limits<data_t>(src, dst, max_shared_mem_bytes, num_sms);

  constexpr cuda::std::array<unsigned, 3> expected_tile_sizes{32, 32, 32};
  REQUIRE(result.__is_valid);
  REQUIRE(result.__tile_sizes == expected_tile_sizes);
}

TEST_CASE("copy shared_memory tiling falls back to logical 2D", "[copy][shared_memory][tiling][fallback]")
{
  using raw_tensor_t = cuda::experimental::__raw_tensor<int, int, data_t, 3>;

  constexpr cuda::std::size_t max_shared_mem_bytes = 99 * 1024;
  constexpr cuda::std::size_t num_sms              = 1;
  constexpr cuda::std::array<unsigned, 3> expected_tile_sizes{32, 1, 32};

  SECTION("power-of-two extents")
  {
    constexpr cuda::std::array<int, 3> shape{1024, 1024, 1024};
    const raw_tensor_t src{nullptr, 3, shape, {1, 1024, 1024 * 1024}};
    const raw_tensor_t dst{nullptr, 3, shape, {1024 * 1024, 1024, 1}};
    const auto result =
      cuda::experimental::__find_shared_mem_tiling_with_limits<data_t>(src, dst, max_shared_mem_bytes, num_sms);

    REQUIRE(result.__is_valid);
    REQUIRE(result.__tile_sizes == expected_tile_sizes);
  }

  SECTION("odd extents")
  {
    constexpr cuda::std::array<int, 3> shape{1023, 1025, 1024};
    const raw_tensor_t src{nullptr, 3, shape, {1, 1023, 1023 * 1025}};
    const raw_tensor_t dst{nullptr, 3, shape, {1025 * 1024, 1024, 1}};
    const auto result =
      cuda::experimental::__find_shared_mem_tiling_with_limits<data_t>(src, dst, max_shared_mem_bytes, num_sms);

    REQUIRE(result.__is_valid);
    REQUIRE(result.__tile_sizes == expected_tile_sizes);
  }
}
