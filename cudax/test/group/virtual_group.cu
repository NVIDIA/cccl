//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/atomic>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

namespace
{
template <class Config, class Level>
__device__ void test_virtual_group(Config config, Level level)
{
  const auto g = cudax::make_this_group(level, config);

  // Test virtual group with identity mapping.
  {
    cudax::virtual_group vg{cuda::gpu_thread, g, cudax::identity_mapping{}};

    REQUIRE(cuda::gpu_thread.is_part_of(vg));

    REQUIRE(cuda::gpu_thread.count(vg) == cuda::gpu_thread.count(g));
    REQUIRE(cuda::gpu_thread.rank(vg) == cuda::gpu_thread.rank(g));

    static_assert(vg.static_count(g) == 1);
    REQUIRE(vg.count(g) == 1);
    REQUIRE(vg.rank(g) == 0);

    vg.sync();
    vg.sync_aligned();
  }

  // Test virtual group with group_by mapping.
  if constexpr (!cuda::std::is_same_v<Level, cuda::thread_level>)
  {
    constexpr auto n = 4;

    cudax::virtual_group vg{cuda::gpu_thread, g, cudax::group_by<n>{}};

    REQUIRE(cuda::gpu_thread.is_part_of(vg));

    REQUIRE(cuda::gpu_thread.count(vg) == n);
    REQUIRE(cuda::gpu_thread.rank(vg) == cuda::gpu_thread.rank(g) % n);

    REQUIRE(vg.count(g) == cuda::gpu_thread.count(g) / n);
    REQUIRE(vg.rank(g) == cuda::gpu_thread.rank(g) / n);

    vg.sync();
    vg.sync_aligned();
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    test_virtual_group(config, cuda::gpu_thread);
    test_virtual_group(config, cuda::warp);
    test_virtual_group(config, cuda::block);
    test_virtual_group(config, cuda::cluster);
    test_virtual_group(config, cuda::grid);
  }
};

C2H_TEST("Virtual Group", "[virtual_group]")
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
} // namespace
