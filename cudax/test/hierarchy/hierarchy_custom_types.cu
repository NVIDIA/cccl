//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include <cooperative_groups.h>
#include <host_device.cuh>

struct custom_level : public cudax::hierarchy_level
{
  using product_type  = unsigned int;
  using allowed_above = cudax::allowed_levels<cudax::grid_level>;
  using allowed_below = cudax::allowed_levels<cudax::block_level>;
};

template <typename Level, typename Dims>
struct custom_level_dims : public cudax::level_dimensions<Level, Dims>
{
  int dummy;
  constexpr custom_level_dims()
      : cudax::level_dimensions<Level, Dims>() {};
};

struct custom_level_test
{
  template <typename DynDims>
  __host__ __device__ void operator()(const DynDims& dims) const
  {
    CUDAX_REQUIRE(dims.count() == 84 * 1024);
    CUDAX_REQUIRE(dims.count(custom_level(), cudax::grid) == 42);
    CUDAX_REQUIRE(dims.extents() == dim3(42 * 512, 2, 2));
    CUDAX_REQUIRE(dims.extents(custom_level(), cudax::grid) == dim3(42, 1, 1));
  }

  void run()
  {
    // Check extending level_dimensions with custom info
    custom_level_dims<cudax::block_level, cudax::dimensions<int, 64, 1, 1>> custom_block;
    custom_block.dummy     = 2;
    auto custom_dims       = cudax::make_hierarchy(cudax::grid_dims<256>(), cudax::cluster_dims<8>(), custom_block);
    auto custom_block_back = custom_dims.level(cudax::block);
    CUDAX_REQUIRE(custom_block_back.dummy == 2);

    auto custom_dims_fragment = custom_dims.fragment(cudax::thread, cudax::block);
    auto custom_block_back2   = custom_dims_fragment.level(cudax::block);
    CUDAX_REQUIRE(custom_block_back2.dummy == 2);

    // Check creating a custom level type works
    auto custom_level_dims = cudax::dimensions<cudax::dimensions_index_type, 2, 2, 2>();
    auto custom_hierarchy  = cudax::make_hierarchy(
      cudax::grid_dims(42),
      cudax::level_dimensions<custom_level, decltype(custom_level_dims)>(custom_level_dims),
      cudax::block_dims<256>());

    static_assert(custom_hierarchy.extents(cudax::thread, custom_level()) == dim3(512, 2, 2));
    static_assert(custom_hierarchy.count(cudax::thread, custom_level()) == 2048);

    test_host_dev(custom_hierarchy, *this);
  }
};

C2H_TEST("Custom level", "[hierarchy]")
{
  custom_level_test().run();
}
