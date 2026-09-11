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

template <class Config>
__device__ void test_coalesced_group(Config config)
{
  const auto rank = cuda::gpu_thread.rank(cuda::warp);

  // Test coalesced group constructoed by all threads in a warp.
  {
    cudax::coalesced_group group{config};
    using Group = decltype(group);

    static_assert(cudax::is_group<Group>);

    static_assert(cuda::std::is_same_v<cuda::thread_level, typename Group::unit_type>);
    static_assert(cuda::std::is_same_v<cuda::warp_level, typename Group::level_type>);

    // Test that the group can be queried for it's hierarchy.
    {
      decltype(auto) hierarchy = cuda::std::as_const(group).hierarchy();
      static_assert(cuda::std::is_same_v<decltype(hierarchy), const typename Config::hierarchy_type&>);
    }

    group.sync();
    group.sync_aligned();

    static_assert(cuda::gpu_thread.static_count(group) == cuda::std::dynamic_extent);
    REQUIRE(cuda::gpu_thread.count(group) == 32);
    REQUIRE(cuda::gpu_thread.rank(group) == rank);

    static_assert(group.static_count(cuda::warp) == 1);
    // todo(dabayer): Refactor static_count query and uncomment these.
    // static_assert(group.static_count(cuda::block) == cuda::warp.static_count(cuda::block, config));
    // static_assert(group.static_count(cuda::cluster) == cuda::warp.static_count(cuda::cluster, config));
    // static_assert(group.static_count(cuda::grid) == cuda::warp.static_count(cuda::grid, config));

    REQUIRE(group.count(cuda::warp) == 1);
    REQUIRE(group.count(cuda::block) == cuda::warp.count(cuda::block));
    REQUIRE(group.count(cuda::cluster) == cuda::warp.count(cuda::cluster));
    REQUIRE(group.count(cuda::grid) == cuda::warp.count(cuda::grid));

    REQUIRE(group.rank(cuda::warp) == 0);
    REQUIRE(group.rank(cuda::block) == cuda::warp.rank(cuda::block));
    REQUIRE(group.rank(cuda::cluster) == cuda::warp.rank(cuda::cluster));
    REQUIRE(group.rank(cuda::grid) == cuda::warp.rank(cuda::grid));
  }

  // Test coalesced group constructoed by first half threads in a warp.
  if (rank < 16)
  {
    cudax::coalesced_group group{config};
    using Group = decltype(group);

    static_assert(cudax::is_group<Group>);

    static_assert(cuda::std::is_same_v<cuda::thread_level, typename Group::unit_type>);
    static_assert(cuda::std::is_same_v<cuda::warp_level, typename Group::level_type>);

    // Test that the group can be queried for it's hierarchy.
    {
      decltype(auto) hierarchy = cuda::std::as_const(group).hierarchy();
      static_assert(cuda::std::is_same_v<decltype(hierarchy), const typename Config::hierarchy_type&>);
    }

    group.sync();
    group.sync_aligned();

    static_assert(cuda::gpu_thread.static_count(group) == cuda::std::dynamic_extent);
    REQUIRE(cuda::gpu_thread.count(group) == 16);
    REQUIRE(cuda::gpu_thread.rank(group) == rank);

    static_assert(group.static_count(cuda::warp) == 1);
    // todo(dabayer): Refactor static_count query and uncomment these.
    // static_assert(group.static_count(cuda::block) == cuda::warp.static_count(cuda::block, config));
    // static_assert(group.static_count(cuda::cluster) == cuda::warp.static_count(cuda::cluster, config));
    // static_assert(group.static_count(cuda::grid) == cuda::warp.static_count(cuda::grid, config));

    REQUIRE(group.count(cuda::warp) == 1);
    REQUIRE(group.count(cuda::block) == cuda::warp.count(cuda::block));
    REQUIRE(group.count(cuda::cluster) == cuda::warp.count(cuda::cluster));
    REQUIRE(group.count(cuda::grid) == cuda::warp.count(cuda::grid));

    REQUIRE(group.rank(cuda::warp) == 0);
    REQUIRE(group.rank(cuda::block) == cuda::warp.rank(cuda::block));
    REQUIRE(group.rank(cuda::cluster) == cuda::warp.rank(cuda::cluster));
    REQUIRE(group.rank(cuda::grid) == cuda::warp.rank(cuda::grid));
  }

  // Test coalesced group constructoed by all even threads in a warp.
  if (rank % 2 == 0)
  {
    cudax::coalesced_group group{config};
    using Group = decltype(group);

    static_assert(cudax::is_group<Group>);

    static_assert(cuda::std::is_same_v<cuda::thread_level, typename Group::unit_type>);
    static_assert(cuda::std::is_same_v<cuda::warp_level, typename Group::level_type>);

    // Test that the group can be queried for it's hierarchy.
    {
      decltype(auto) hierarchy = cuda::std::as_const(group).hierarchy();
      static_assert(cuda::std::is_same_v<decltype(hierarchy), const typename Config::hierarchy_type&>);
    }

    group.sync();
    group.sync_aligned();

    static_assert(cuda::gpu_thread.static_count(group) == cuda::std::dynamic_extent);
    REQUIRE(cuda::gpu_thread.count(group) == 16);
    REQUIRE(cuda::gpu_thread.rank(group) == rank / 2);

    static_assert(group.static_count(cuda::warp) == 1);
    // todo(dabayer): Refactor static_count query and uncomment these.
    // static_assert(group.static_count(cuda::block) == cuda::warp.static_count(cuda::block, config));
    // static_assert(group.static_count(cuda::cluster) == cuda::warp.static_count(cuda::cluster, config));
    // static_assert(group.static_count(cuda::grid) == cuda::warp.static_count(cuda::grid, config));

    REQUIRE(group.count(cuda::warp) == 1);
    REQUIRE(group.count(cuda::block) == cuda::warp.count(cuda::block));
    REQUIRE(group.count(cuda::cluster) == cuda::warp.count(cuda::cluster));
    REQUIRE(group.count(cuda::grid) == cuda::warp.count(cuda::grid));

    REQUIRE(group.rank(cuda::warp) == 0);
    REQUIRE(group.rank(cuda::block) == cuda::warp.rank(cuda::block));
    REQUIRE(group.rank(cuda::cluster) == cuda::warp.rank(cuda::cluster));
    REQUIRE(group.rank(cuda::grid) == cuda::warp.rank(cuda::grid));
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    test_coalesced_group(config);
  }
};

C2H_TEST("Coalesced Group", "[coalesced_group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
  cuda::launch(stream, config, TestKernel{});

  stream.sync();
}
