//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include <cooperative_groups.h>
#include <host_device.cuh>

struct custom_level : public cuda::hierarchy_level
{
  using product_type  = unsigned int;
  using allowed_above = cuda::allowed_levels<cuda::grid_level>;
  using allowed_below = cuda::allowed_levels<cuda::block_level>;
};

template <typename Level, typename Dims>
struct custom_level_dims : public cuda::level_dimensions<Level, Dims>
{
  int dummy;
  constexpr custom_level_dims()
      : cuda::level_dimensions<Level, Dims>() {};
};

struct custom_level_test
{
  template <typename DynDims>
  __host__ __device__ void operator()(const DynDims& dims) const
  {
    // device-side require doesn't work with clang-cuda for now
#if !_CCCL_CUDA_COMPILER(CLANG)
    CCCLRT_REQUIRE(dims.count() == 84 * 1024);
    CCCLRT_REQUIRE(dims.count(custom_level(), cuda::grid) == 42);
    CCCLRT_REQUIRE(dims.extents() == dim3(42 * 512, 2, 2));
    CCCLRT_REQUIRE(dims.extents(custom_level(), cuda::grid) == dim3(42, 1, 1));
#endif
  }

  void run()
  {
    // Check extending level_dimensions with custom info
    custom_level_dims<cuda::block_level, cuda::dimensions<int, 64, 1, 1>> custom_block;
    custom_block.dummy     = 2;
    auto custom_dims       = cuda::make_hierarchy(cuda::grid_dims<256>(), cuda::cluster_dims<8>(), custom_block);
    auto custom_block_back = custom_dims.level(cuda::block);
    CCCLRT_REQUIRE(custom_block_back.dummy == 2);

    auto custom_dims_fragment = custom_dims.fragment(cuda::thread, cuda::block);
    auto custom_block_back2   = custom_dims_fragment.level(cuda::block);
    CCCLRT_REQUIRE(custom_block_back2.dummy == 2);

    // Check creating a custom level type works
    auto custom_level_dims = cuda::dimensions<cuda::dimensions_index_type, 2, 2, 2>();
    auto custom_hierarchy  = cuda::make_hierarchy(
      cuda::grid_dims(42),
      cuda::level_dimensions<custom_level, decltype(custom_level_dims)>(custom_level_dims),
      cuda::block_dims<256>());

    static_assert(custom_hierarchy.extents(cuda::thread, custom_level()) == dim3(512, 2, 2));
    static_assert(custom_hierarchy.count(cuda::thread, custom_level()) == 2048);

    test_host_dev(custom_hierarchy, *this);
  }
};

C2H_TEST("Custom level", "[hierarchy]")
{
  custom_level_test().run();
}
