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

struct custom_level : public cuda::hierarchy_level_base<custom_level>
{
  using __product_type  = unsigned int;
  using __allowed_above = cuda::__allowed_levels<cuda::grid_level>;
  using __allowed_below = cuda::__allowed_levels<cuda::block_level>;
};

template <typename Level, typename Dims>
struct custom_level_dims : public cuda::hierarchy_level_desc<Level, Dims>
{
  int dummy;
  constexpr custom_level_dims()
      : cuda::hierarchy_level_desc<Level, Dims>() {};
};

struct custom_level_test
{
  template <typename DynDims>
  __host__ __device__ void operator()(const DynDims& dims) const
  {
    // todo: allow this after fixing CCCLRT_REQUIRE with clang-cuda
#if !_CCCL_CUDA_COMPILER(CLANG)
    CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, dims) == 84 * 1024);
    CCCLRT_REQUIRE(custom_level{}.count(cuda::grid, dims) == 42);
    CCCLRT_REQUIRE(cuda::gpu_thread.dims(cuda::grid, dims) == dim3(42 * 512, 2, 2));
    CCCLRT_REQUIRE(custom_level{}.dims(cuda::grid, dims) == dim3(42, 1, 1));
#endif // !_CCCL_CUDA_COMPILER(CLANG)
  }

  void run()
  {
    // Check extending hierarchy_level_desc with custom info
    custom_level_dims<cuda::block_level, cuda::std::extents<int, 64, 1, 1>> custom_block;
    custom_block.dummy     = 2;
    auto custom_dims       = cuda::make_hierarchy(cuda::grid_dims<256>(), cuda::cluster_dims<8>(), custom_block);
    auto custom_block_back = custom_dims.level(cuda::block);
    CCCLRT_REQUIRE(custom_block_back.dummy == 2);

    auto custom_dims_fragment = custom_dims.fragment(cuda::gpu_thread, cuda::block);
    auto custom_block_back2   = custom_dims_fragment.level(cuda::block);
    CCCLRT_REQUIRE(custom_block_back2.dummy == 2);

    // Check creating a custom level type works
    auto custom_level_dims = cuda::std::extents<cuda::dimensions_index_type, 2, 2, 2>();
    auto custom_hierarchy  = cuda::make_hierarchy(
      cuda::grid_dims(42),
      cuda::hierarchy_level_desc<custom_level, decltype(custom_level_dims)>(custom_level_dims),
      cuda::block_dims<256>());

    static_assert(cuda::gpu_thread.dims(custom_level(), custom_hierarchy) == dim3(512, 2, 2));
    static_assert(cuda::gpu_thread.count(custom_level(), custom_hierarchy) == 2048);

    test_host_dev(custom_hierarchy, *this);
  }
};

C2H_TEST("Custom level", "[hierarchy]")
{
  custom_level_test().run();
}
