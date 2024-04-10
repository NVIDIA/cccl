//===----------------------------------------------------------------------===//
//
// Part of CUDA Next in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iostream>

#include "catch2_helpers/testing_common.cuh"
#include <cooperative_groups.h>

struct custom_level : public cuda_next::hierarchy_level
{
  using product_type  = unsigned int;
  using allowed_above = cuda_next::allowed_levels<cuda_next::grid_level>;
  using allowed_below = cuda_next::allowed_levels<cuda_next::block_level>;
};

template <typename Level, typename Dims>
struct custom_level_dims : public cuda_next::level_dimensions<Level, Dims>
{
  int dummy;
  constexpr custom_level_dims() : cuda_next::level_dimensions<Level, Dims>() {};
};

template <typename Dims>
__global__ void kernel_custom_level(Dims dims)
{
  auto cnt = dims.count(cuda_next::grid, custom_level());
}

TEST_CASE("Custom level", "[hierarchy]")
{
  // Check extending level_dimensions with custom info
  custom_level_dims<cuda_next::block_level, cuda_next::detail::dims<64>> custom_block;
  custom_block.dummy     = 2;
  auto custom_dims       = cuda_next::grid_dims<256>() & cuda_next::cluster_dims<8>() & custom_block;
  auto custom_block_back = custom_dims.level(cuda_next::block);
  REQUIRE(custom_block_back.dummy == 2);

  auto custom_dims_fragment = custom_dims.fragment(cuda_next::thread, cuda_next::block);
  auto custom_block_back2   = custom_dims_fragment.level(cuda_next::block);
  REQUIRE(custom_block_back2.dummy == 2);

  // Check creating a custom level type works
  auto custom_level_dims = cuda_next::dimensions<cuda_next::dimensions_index_type, 2, 2, 2>();
  auto custom_hierarchy  = cuda_next::make_hierarchy(
    cuda_next::grid_dims(42),
    cuda_next::level_dimensions<custom_level, decltype(custom_level_dims)>(custom_level_dims),
    cuda_next::block_dims<256>());

  static_assert(custom_hierarchy.flatten(cuda_next::thread, custom_level()) == dim3(512, 2, 2));
  static_assert(custom_hierarchy.count(cuda_next::thread, custom_level()) == 2048);

  test_host_dev(custom_hierarchy, [] __host__ __device__(const decltype(custom_hierarchy)& dims) {
    HOST_DEV_REQUIRE(dims.count() == 84 * 1024);
    HOST_DEV_REQUIRE(dims.count(custom_level(), cuda_next::grid) == 42);
    HOST_DEV_REQUIRE(dims.flatten() == dim3(42 * 512, 2, 2));
    HOST_DEV_REQUIRE(dims.flatten(custom_level(), cuda_next::grid) == dim3(42, 1, 1));
  });
}

template <typename Level, typename Dims>
struct level_disabled_copy : public cuda_next::level_dimensions<Level, Dims>
{
  constexpr __host__ __device__ level_disabled_copy(const Dims& d)
      : cuda_next::level_dimensions<Level, Dims>(d)
  {}

  constexpr level_disabled_copy(const level_disabled_copy<Level, Dims>& d) = delete;
  constexpr level_disabled_copy(level_disabled_copy<Level, Dims>&& d)      = default;
};

TEST_CASE("Disabled lvalue copy", "hierarchy")
{
  auto ext               = cuda::std::extents<size_t, cuda::std::dynamic_extent, 1, 1>(64, 1, 1);
  auto ext_static        = cuda::std::extents<size_t, 64, 1, 1>();
  auto block_dims        = level_disabled_copy<cuda_next::block_level, decltype(ext)>(ext);
  auto block_dims2       = level_disabled_copy<cuda_next::block_level, decltype(ext)>(ext);
  auto block_dims_static = level_disabled_copy<cuda_next::block_level, decltype(ext_static)>(ext_static);

  auto hierarchy     = cuda_next::make_hierarchy(cuda_next::grid_dims(256), std::move(block_dims));
  auto hierarchy_rev = cuda_next::make_hierarchy(std::move(block_dims2), cuda_next::grid_dims(256));
  static_assert(std::is_same_v<decltype(hierarchy), decltype(hierarchy_rev)>);

  REQUIRE(hierarchy.count() == 256 * 64);
  REQUIRE(hierarchy_rev.count() == 256 * 64);

  auto hierarchy_static = cuda_next::make_hierarchy(std::move(block_dims_static), cuda_next::grid_dims(256));

  static_assert(hierarchy_static.count(cuda_next::thread, cuda_next::block) == 64);
}
