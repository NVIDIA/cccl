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

template <class Config, class Level>
__device__ void test_group_view(Config config, Level level)
{
  const auto g = cudax::make_this_group(level, config);

  // Test the view is not default constructible.
  {
    using GroupView = decltype(cudax::group_view{g});
    static_assert(!cuda::std::is_default_constructible_v<GroupView>);
  }

  // Test the view is copy constructible.
  {
    using GroupView = decltype(cudax::group_view{g});
    static_assert(cuda::std::is_copy_constructible_v<GroupView>);
  }

  // Test view over the group.
  {
    const cudax::group_view gv{g};

    REQUIRE(cuda::gpu_thread.is_part_of(gv));
    if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
    {
      REQUIRE(cuda::gpu_thread.count(gv) == 1);
      REQUIRE(cuda::gpu_thread.rank(gv) == 0);
    }
    else
    {
      REQUIRE(cuda::gpu_thread.count(gv) == cuda::gpu_thread.count(level));
      REQUIRE(cuda::gpu_thread.rank(gv) == cuda::gpu_thread.rank(level));
    }

    static_assert(gv.static_count(level) == 1);
    REQUIRE(gv.count(level) == 1);
    REQUIRE(gv.rank(level) == 0);

    gv.sync();

    // Test that the view is copyable.
    const cudax::group_view gv2{gv};

    REQUIRE(cuda::gpu_thread.is_part_of(gv2));
    if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == 1);
      REQUIRE(cuda::gpu_thread.rank(gv2) == 0);
    }
    else
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == cuda::gpu_thread.count(level));
      REQUIRE(cuda::gpu_thread.rank(gv2) == cuda::gpu_thread.rank(level));
    }

    static_assert(gv2.static_count(level) == 1);
    REQUIRE(gv2.count(level) == 1);
    REQUIRE(gv2.rank(level) == 0);

    gv2.sync();
  }

  // Test view over the group with the unit changed.
  {
    const cudax::group_view gv{cuda::gpu_thread, g};

    REQUIRE(cuda::gpu_thread.is_part_of(gv));
    if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
    {
      REQUIRE(cuda::gpu_thread.count(gv) == 1);
      REQUIRE(cuda::gpu_thread.rank(gv) == 0);
    }
    else
    {
      REQUIRE(cuda::gpu_thread.count(gv) == cuda::gpu_thread.count(level));
      REQUIRE(cuda::gpu_thread.rank(gv) == cuda::gpu_thread.rank(level));
    }

    static_assert(gv.static_count(level) == 1);
    REQUIRE(gv.count(level) == 1);
    REQUIRE(gv.rank(level) == 0);

    gv.sync();

    // Test that the view is copyable.
    const cudax::group_view gv2{gv};

    REQUIRE(cuda::gpu_thread.is_part_of(gv2));
    if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == 1);
      REQUIRE(cuda::gpu_thread.rank(gv2) == 0);
    }
    else
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == cuda::gpu_thread.count(level));
      REQUIRE(cuda::gpu_thread.rank(gv2) == cuda::gpu_thread.rank(level));
    }

    static_assert(gv2.static_count(level) == 1);
    REQUIRE(gv2.count(level) == 1);
    REQUIRE(gv2.rank(level) == 0);

    gv2.sync();
  }

  // Test view of the group view of the group with the unit changed.
  {
    const cudax::group_view gv{g};

    const cudax::group_view gv2{cuda::gpu_thread, gv};

    REQUIRE(cuda::gpu_thread.is_part_of(gv2));
    if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == 1);
      REQUIRE(cuda::gpu_thread.rank(gv2) == 0);
    }
    else
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == cuda::gpu_thread.count(level));
      REQUIRE(cuda::gpu_thread.rank(gv2) == cuda::gpu_thread.rank(level));
    }

    static_assert(gv2.static_count(level) == 1);
    REQUIRE(gv2.count(level) == 1);
    REQUIRE(gv2.rank(level) == 0);

    gv2.sync();
  }

  // Test view of a group view of a group with the unit changed twice.
  if constexpr (!cuda::std::is_same_v<Level, cuda::thread_level>)
  {
    const cudax::group_view gv{cuda::warp, g};

    const cudax::group_view gv2{cuda::gpu_thread, gv};

    REQUIRE(cuda::gpu_thread.is_part_of(gv2));
    if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == 1);
      REQUIRE(cuda::gpu_thread.rank(gv2) == 0);
    }
    else
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == cuda::gpu_thread.count(level));
      REQUIRE(cuda::gpu_thread.rank(gv2) == cuda::gpu_thread.rank(level));
    }

    static_assert(gv2.static_count(level) == 1);
    REQUIRE(gv2.count(level) == 1);
    REQUIRE(gv2.rank(level) == 0);

    gv2.sync();
  }

  // Test a view of a generic group.
  if constexpr (cuda::std::is_same_v<Level, cuda::block_level>)
  {
    constexpr auto n = 2;
    auto& barriers   = get_barriers<cuda::warp.static_count(level, config) / n>(cuda::warp);

    const cudax::group g2{cuda::warp, g, cudax::group_by<n>{}, cudax::barrier_synchronizer{barriers}};
    REQUIRE(cuda::warp.count(g2) == n);
    REQUIRE(g2.count(g) == cuda::warp.count(level) / n);

    const cudax::group_view g2_view{g2};
    REQUIRE(cuda::warp.is_part_of(g2_view));
    REQUIRE(cuda::warp.count(g2_view) == n);
    REQUIRE(g2_view.count(g2) == cuda::warp.count(level) / n);
    g2_view.sync();

    const cudax::group_view g2_view_threads{cuda::gpu_thread, g2_view};
    REQUIRE(cuda::gpu_thread.is_part_of(g2_view_threads));
    REQUIRE(cuda::gpu_thread.count(g2_view_threads) == n * cuda::gpu_thread.count(cuda::warp));
    REQUIRE(g2_view_threads.count(g2) == n);
    g2_view_threads.sync();
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    test_group_view(config, cuda::gpu_thread);
    test_group_view(config, cuda::warp);
    test_group_view(config, cuda::block);
    test_group_view(config, cuda::cluster);
    test_group_view(config, cuda::grid);
  }
};

C2H_TEST("Group View", "[group_view]")
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
