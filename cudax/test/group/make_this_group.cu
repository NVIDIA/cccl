//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

namespace
{
template <template <class> class GroupTempl, class Level, class Config>
__device__ void test_make_this_group(const Level& level, const Config& config)
{
  // Test default construction.
  {
    static_assert(
      cuda::std::is_same_v<GroupTempl<cudax::__implicit_hierarchy_t>, decltype(cudax::make_this_group(level))>);
    static_assert(noexcept(cudax::make_this_group(level)));

    auto group = cudax::make_this_group(level);
    group.sync();
  }

  // Test construction from hierarchy-like.
  {
    using Hierarchy = typename Config::hierarchy_type;

    static_assert(cuda::std::is_same_v<GroupTempl<Hierarchy>, decltype(cudax::make_this_group(level, config))>);
    static_assert(noexcept(cudax::make_this_group(level, config)));

    auto group = cudax::make_this_group(level, config);
    group.sync();
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_make_this_group<cudax::this_thread>(cuda::gpu_thread, config);
    test_make_this_group<cudax::this_warp>(cuda::warp, config);
    test_make_this_group<cudax::this_block>(cuda::block, config);
    test_make_this_group<cudax::this_cluster>(cuda::cluster, config);
    test_make_this_group<cudax::this_grid>(cuda::grid, config);
  }
};
} // namespace

C2H_TEST("Make this group", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
  cuda::launch(stream, config, TestKernel{});

  if (cuda::device_attributes::compute_capability(device) >= cuda::compute_capability{90})
  {
    const auto config_cluster = cuda::make_config(
      cuda::grid_dims<2>(), cuda::cluster_dims<3>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
    cuda::launch(stream, config_cluster, TestKernel{});
  }

  stream.sync();
}
