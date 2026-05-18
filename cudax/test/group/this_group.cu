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

#include <cooperative_groups.h>

#include "group_testing.cuh"

namespace
{
__device__ unsigned global_var = 0;

template <class Level, class Hierarchy, class Group>
__device__ void test_common_properties(const Hierarchy&, Group& group)
{
  // Assert that Group satisfies the group concept.
  static_assert(cudax::is_group<Group>);

  // Test types
  static_assert(cuda::std::is_same_v<Level, typename Group::unit_type>);
  static_assert(cuda::std::is_same_v<Level, typename Group::level_type>);

  // Test that the group can be queried for it's hierarchy.
  {
    decltype(auto) hierarchy = cuda::std::as_const(group).hierarchy();
    static_assert(cuda::std::is_same_v<decltype(hierarchy), const Hierarchy&>);
  }

  // Test that the group can be synchronized using .sync() method.
  {
    static_assert(cuda::std::is_same_v<void, decltype(group.sync())>);
    static_assert(noexcept(group.sync()));

    // .sync() method must support calls from different branches. Add some dummy work to make sure the branches are not
    // collided.
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> atomic{global_var};
    if ((threadIdx.x + threadIdx.y + threadIdx.z) % 2 == 0)
    {
      atomic++;
      group.sync();
      atomic--;
    }
    else
    {
      atomic--;
      group.sync();
      atomic++;
    }
  }

  // Test that the group can be synchronized using .sync_aligned() method.
  {
    static_assert(cuda::std::is_same_v<void, decltype(group.sync_aligned())>);
    static_assert(noexcept(group.sync_aligned()));

    // .sync_aligned() method must be called by all threads in the group uniformly in one place.
    group.sync_aligned();
  }
}

template <class Hierarchy>
__device__ void test_this_queries(const cudax::this_thread<Hierarchy>& group)
{
  // todo(dabayer): These queries end up in `error: expression must have a constant value`, when group is taken by
  // reference. Can we find a solution that works without copying the group?
  // static_assert(cuda::gpu_thread.static_count(group) == 1);

  CUDAX_REQUIRE(cuda::gpu_thread.count(group) == 1);
  CUDAX_REQUIRE(group.count(cuda::warp) == cuda::gpu_thread.count(cuda::warp));
  CUDAX_REQUIRE(group.count(cuda::block) == cuda::gpu_thread.count(cuda::block));
  CUDAX_REQUIRE(group.count(cuda::cluster) == cuda::gpu_thread.count(cuda::cluster));
  CUDAX_REQUIRE(group.count(cuda::grid) == cuda::gpu_thread.count(cuda::grid));

  CUDAX_REQUIRE(cuda::gpu_thread.rank(group) == 0);
  CUDAX_REQUIRE(group.rank(cuda::warp) == cuda::gpu_thread.rank(cuda::warp));
  CUDAX_REQUIRE(group.rank(cuda::block) == cuda::gpu_thread.rank(cuda::block));
  CUDAX_REQUIRE(group.rank(cuda::cluster) == cuda::gpu_thread.rank(cuda::cluster));
  CUDAX_REQUIRE(group.rank(cuda::grid) == cuda::gpu_thread.rank(cuda::grid));

  CUDAX_REQUIRE(cuda::gpu_thread.is_root_rank(group));

  CUDAX_REQUIRE(cuda::gpu_thread.is_part_of(group));
}

template <class Hierarchy>
__device__ void test_this_queries(const cudax::this_warp<Hierarchy>& group)
{
  // todo(dabayer): These queries end up in `error: expression must have a constant value`, when group is taken by
  // reference. Can we find a solution that works without copying the group?
  // static_assert(cuda::gpu_thread.static_count(group) == cuda::gpu_thread.static_count(cuda::warp,
  // group.hierarchy())); static_assert(cuda::warp.static_count(group) == 1);

  CUDAX_REQUIRE(cuda::gpu_thread.count(group) == cuda::gpu_thread.count(cuda::warp));
  CUDAX_REQUIRE(cuda::warp.count(group) == 1);
  CUDAX_REQUIRE(group.count(cuda::block) == cuda::warp.count(cuda::block));
  CUDAX_REQUIRE(group.count(cuda::cluster) == cuda::warp.count(cuda::cluster));
  CUDAX_REQUIRE(group.count(cuda::grid) == cuda::warp.count(cuda::grid));

  CUDAX_REQUIRE(cuda::gpu_thread.rank(group) == cuda::gpu_thread.rank(cuda::warp));
  CUDAX_REQUIRE(cuda::warp.rank(group) == 0);
  CUDAX_REQUIRE(group.rank(cuda::block) == cuda::warp.rank(cuda::block));
  CUDAX_REQUIRE(group.rank(cuda::cluster) == cuda::warp.rank(cuda::cluster));
  CUDAX_REQUIRE(group.rank(cuda::grid) == cuda::warp.rank(cuda::grid));

  CUDAX_REQUIRE(cuda::gpu_thread.is_root_rank(group) == (cuda::gpu_thread.rank(cuda::warp) == 0));
  CUDAX_REQUIRE(cuda::warp.is_root_rank(group));

  CUDAX_REQUIRE(cuda::gpu_thread.is_part_of(group));
  CUDAX_REQUIRE(cuda::warp.is_part_of(group));
}

template <class Hierarchy>
__device__ void test_this_queries(const cudax::this_block<Hierarchy>& group)
{
  // todo(dabayer): These queries end up in `error: expression must have a constant value`, when group is taken by
  // reference. Can we find a solution that works without copying the group?
  // static_assert(cuda::gpu_thread.static_count(group) == cuda::gpu_thread.static_count(cuda::block,
  // group.hierarchy())); static_assert(cuda::warp.static_count(group) == cuda::warp.static_count(cuda::block,
  // group.hierarchy())); static_assert(cuda::block.static_count(group) == 1);

  CUDAX_REQUIRE(cuda::gpu_thread.count(group) == cuda::gpu_thread.count(cuda::block));
  CUDAX_REQUIRE(cuda::warp.count(group) == cuda::warp.count(cuda::block));
  CUDAX_REQUIRE(cuda::block.count(group) == 1);
  CUDAX_REQUIRE(group.count(cuda::cluster) == cuda::block.count(cuda::cluster));
  CUDAX_REQUIRE(group.count(cuda::grid) == cuda::block.count(cuda::grid));

  CUDAX_REQUIRE(cuda::gpu_thread.rank(group) == cuda::gpu_thread.rank(cuda::block));
  CUDAX_REQUIRE(cuda::warp.rank(group) == cuda::warp.rank(cuda::block));
  CUDAX_REQUIRE(cuda::block.rank(group) == 0);
  CUDAX_REQUIRE(group.rank(cuda::cluster) == cuda::block.rank(cuda::cluster));
  CUDAX_REQUIRE(group.rank(cuda::grid) == cuda::block.rank(cuda::grid));

  CUDAX_REQUIRE(cuda::gpu_thread.is_root_rank(group) == (cuda::gpu_thread.rank(cuda::block) == 0));
  CUDAX_REQUIRE(cuda::warp.is_root_rank(group) == (cuda::warp.rank(cuda::block) == 0));
  CUDAX_REQUIRE(cuda::block.is_root_rank(group));

  CUDAX_REQUIRE(cuda::gpu_thread.is_part_of(group));
  CUDAX_REQUIRE(cuda::warp.is_part_of(group));
  CUDAX_REQUIRE(cuda::block.is_part_of(group));
}

template <class Hierarchy>
__device__ void test_this_queries(const cudax::this_cluster<Hierarchy>& group)
{
  // todo(dabayer): These queries end up in `error: expression must have a constant value`, when group is taken by
  // reference. Can we find a solution that works without copying the group?
  // static_assert(cuda::gpu_thread.static_count(group) == cuda::gpu_thread.static_count(cuda::cluster,
  // group.hierarchy())); static_assert(cuda::warp.static_count(group) == cuda::warp.static_count(cuda::cluster,
  // group.hierarchy())); static_assert(cuda::block.static_count(group) == cuda::block.static_count(cuda::cluster,
  // group.hierarchy())); static_assert(cuda::cluster.static_count(group) == 1);

  CUDAX_REQUIRE(cuda::gpu_thread.count(group) == cuda::gpu_thread.count(cuda::cluster));
  CUDAX_REQUIRE(cuda::warp.count(group) == cuda::warp.count(cuda::cluster));
  CUDAX_REQUIRE(cuda::block.count(group) == cuda::block.count(cuda::cluster));
  CUDAX_REQUIRE(cuda::cluster.count(group) == 1);
  CUDAX_REQUIRE(group.count(cuda::grid) == cuda::cluster.count(cuda::grid));

  CUDAX_REQUIRE(cuda::gpu_thread.rank(group) == cuda::gpu_thread.rank(cuda::cluster));
  CUDAX_REQUIRE(cuda::warp.rank(group) == cuda::warp.rank(cuda::cluster));
  CUDAX_REQUIRE(cuda::block.rank(group) == cuda::block.rank(cuda::cluster));
  CUDAX_REQUIRE(cuda::cluster.rank(group) == 0);
  CUDAX_REQUIRE(group.rank(cuda::grid) == cuda::cluster.rank(cuda::grid));

  CUDAX_REQUIRE(cuda::gpu_thread.is_root_rank(group) == (cuda::gpu_thread.rank(cuda::cluster) == 0));
  CUDAX_REQUIRE(cuda::warp.is_root_rank(group) == (cuda::warp.rank(cuda::cluster) == 0));
  CUDAX_REQUIRE(cuda::block.is_root_rank(group) == (cuda::block.rank(cuda::cluster) == 0));
  CUDAX_REQUIRE(cuda::cluster.is_root_rank(group));

  CUDAX_REQUIRE(cuda::gpu_thread.is_part_of(group));
  CUDAX_REQUIRE(cuda::warp.is_part_of(group));
  CUDAX_REQUIRE(cuda::block.is_part_of(group));
  CUDAX_REQUIRE(cuda::cluster.is_part_of(group));
}

template <class Hierarchy>
__device__ void test_this_queries(const cudax::this_grid<Hierarchy>& group)
{
  // todo(dabayer): These queries end up in `error: expression must have a constant value`, when group is taken by
  // reference. Can we find a solution that works without copying the group?
  // static_assert(cuda::gpu_thread.static_count(group) == cuda::gpu_thread.static_count(cuda::grid,
  // group.hierarchy())); static_assert(cuda::warp.static_count(group) == cuda::warp.static_count(cuda::grid,
  // group.hierarchy())); static_assert(cuda::block.static_count(group) == cuda::block.static_count(cuda::grid,
  // group.hierarchy())); static_assert(cuda::cluster.static_count(group) == cuda::cluster.static_count(cuda::grid,
  // group.hierarchy())); static_assert(cuda::grid.static_count(group) == 1);

  CUDAX_REQUIRE(cuda::gpu_thread.count(group) == cuda::gpu_thread.count(cuda::grid));
  CUDAX_REQUIRE(cuda::warp.count(group) == cuda::warp.count(cuda::grid));
  CUDAX_REQUIRE(cuda::block.count(group) == cuda::block.count(cuda::grid));
  CUDAX_REQUIRE(cuda::cluster.count(group) == cuda::cluster.count(cuda::grid));
  CUDAX_REQUIRE(cuda::grid.count(group) == 1);

  CUDAX_REQUIRE(cuda::gpu_thread.rank(group) == cuda::gpu_thread.rank(cuda::grid));
  CUDAX_REQUIRE(cuda::warp.rank(group) == cuda::warp.rank(cuda::grid));
  CUDAX_REQUIRE(cuda::block.rank(group) == cuda::block.rank(cuda::grid));
  CUDAX_REQUIRE(cuda::cluster.rank(group) == cuda::cluster.rank(cuda::grid));
  CUDAX_REQUIRE(cuda::grid.rank(group) == 0);

  CUDAX_REQUIRE(cuda::gpu_thread.is_root_rank(group) == (cuda::gpu_thread.rank(cuda::grid) == 0));
  CUDAX_REQUIRE(cuda::warp.is_root_rank(group) == (cuda::warp.rank(cuda::grid) == 0));
  CUDAX_REQUIRE(cuda::block.is_root_rank(group) == (cuda::block.rank(cuda::grid) == 0));
  CUDAX_REQUIRE(cuda::cluster.is_root_rank(group) == (cuda::cluster.rank(cuda::grid) == 0));
  CUDAX_REQUIRE(cuda::grid.is_root_rank(group));

  CUDAX_REQUIRE(cuda::gpu_thread.is_part_of(group));
  CUDAX_REQUIRE(cuda::warp.is_part_of(group));
  CUDAX_REQUIRE(cuda::block.is_part_of(group));
  CUDAX_REQUIRE(cuda::cluster.is_part_of(group));
  CUDAX_REQUIRE(cuda::grid.is_part_of(group));
}

template <class Level, class Hierarchy>
__device__ void test_cg_interop(const Hierarchy& hierarchy)
{
  if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
  {
    cudax::this_thread group{cooperative_groups::this_thread()};
    test_common_properties<Level>(hierarchy, group);
  }
  else if constexpr (cuda::std::is_same_v<Level, cuda::warp_level>)
  {
    cudax::this_warp group{cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block())};
    test_common_properties<Level>(hierarchy, group);
  }
  else if constexpr (cuda::std::is_same_v<Level, cuda::block_level>)
  {
    cudax::this_block group{cooperative_groups::this_thread_block()};
    test_common_properties<Level>(hierarchy, group);
  }
  else if constexpr (cuda::std::is_same_v<Level, cuda::cluster_level>)
  {
#if defined(_CG_HAS_CLUSTER_GROUP)
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   cudax::this_cluster group{cooperative_groups::this_cluster()};
                   test_common_properties<Level>(hierarchy, group);
                 }))
#endif // _CG_HAS_CLUSTER_GROUP
  }
  else if constexpr (cuda::std::is_same_v<Level, cuda::grid_level>)
  {
    cudax::this_grid group{cooperative_groups::this_grid()};
    test_common_properties<Level>(hierarchy, group);
  }
}

template <template <class> class GroupTempl, class Level, class Config>
__device__ void test_this_group(const Level& level, const Config& config)
{
  const auto implicit_hierarchy = cudax::__implicit_hierarchy();

  // Test implicit construction.
  {
    GroupTempl group;
    static_assert(cuda::std::is_same_v<GroupTempl<cudax::__implicit_hierarchy_t>, decltype(group)>);
    static_assert(cuda::std::is_nothrow_default_constructible_v<decltype(group)>);

    test_common_properties<Level>(implicit_hierarchy, group);
    test_this_queries(group);
  }

  // Test construction from kernel_config.
  {
    GroupTempl group{config};
    // nvcc 12.0 doesn't evaluate these static asserts correctly
#if !_CCCL_CUDA_COMPILER(NVCC, ==, 12, 0)
    static_assert(cuda::std::is_same_v<GroupTempl<typename Config::hierarchy_type>, decltype(group)>);
    static_assert(cuda::std::is_nothrow_constructible_v<decltype(group), const typename Config::hierarchy_type&>);
#endif // !_CCCL_CUDA_COMPILER(NVCC, ==, 12, 0)

    test_common_properties<Level>(config.hierarchy(), group);
    test_this_queries(group);
  }

  // Test construction from CG equivalents.
  test_cg_interop<Level>(implicit_hierarchy);
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_this_group<cudax::this_thread>(cuda::gpu_thread, config);
    test_this_group<cudax::this_warp>(cuda::warp, config);
    test_this_group<cudax::this_block>(cuda::block, config);
    test_this_group<cudax::this_cluster>(cuda::cluster, config);
    test_this_group<cudax::this_grid>(cuda::grid, config);
  }
};
} // namespace

C2H_TEST("This group", "[group]")
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
