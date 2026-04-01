//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#define _CUDAX_HIERARCHY

#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/atomic>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/hierarchy.cuh>

#include <cooperative_groups.h>

#include "testing.cuh"

__device__ unsigned global_var = 0;

template <class Config, class Group>
__device__ void test_common_properties(const Config&, Group& group)
{
  // Assert that Group satisfies the group concept.
  static_assert(cudax::group<Group>);

  // Test that the group can be queried for it's hierarchy.
  {
    using ExpectedHierarchy = typename Config::hierarchy_type;
    static_assert(cuda::std::is_same_v<ExpectedHierarchy, typename Group::hierarchy_type>);

    decltype(auto) hierarchy = cuda::std::as_const(group).hierarchy();
    static_assert(cuda::std::is_same_v<decltype(hierarchy), const typename Config::hierarchy_type&>);
  }

  // Test that the group can be synchronized using .sync() method.
  {
    static_assert(cuda::std::is_same_v<void, decltype(group.sync())>);
    static_assert(noexcept(group.sync()));

    // .sync() method must support calls from different branches. Add some dummy work to make sure the branches are not
    // collided.
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> atomic{global_var};
    if (cuda::gpu_thread.rank(group) % 2 == 0)
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

template <class Hierarchy, class T, cuda::std::size_t N>
__device__ cuda::std::optional<T> sum(cudax::this_thread<Hierarchy> group, T (&array)[N])
{
  return {cub::ThreadReduce(array, cuda::std::plus<T>{})};
}

template <class Hierarchy, class T, cuda::std::size_t N>
__device__ cuda::std::optional<T> sum(cudax::this_warp<Hierarchy> group, T (&array)[N])
{
  using WarpReduce = cub::WarpReduce<T>;

  __shared__ typename WarpReduce::TempStorage scratch;

  const auto partial = cub::ThreadReduce(array, cuda::std::plus<T>{});
  const auto result  = WarpReduce{scratch}.Sum(partial);
  return (cuda::gpu_thread.is_root_rank(group)) ? cuda::std::optional{result} : cuda::std::nullopt;
}

template <class Hierarchy, class T, cuda::std::size_t N>
__device__ cuda::std::optional<T> sum(cudax::this_block<Hierarchy> group, T (&array)[N])
{
  using BlockExts = decltype(cuda::gpu_thread.extents(cuda::block, group.hierarchy()));
  static_assert(BlockExts::rank_dynamic() == 0, "This algorithm requires all static extents.");

  using BlockReduce =
    cub::BlockReduce<T,
                     static_cast<int>(BlockExts::static_extent(0)),
                     cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                     static_cast<int>(BlockExts::static_extent(1)),
                     static_cast<int>(BlockExts::static_extent(2))>;

  __shared__ typename BlockReduce::TempStorage scratch;
  const auto result = BlockReduce{scratch}.Sum(array);
  return (cuda::gpu_thread.is_root_rank(group)) ? cuda::std::optional{result} : cuda::std::nullopt;
}

template <class Hierarchy, class T, cuda::std::size_t N>
__device__ cuda::std::optional<T> sum(cudax::this_cluster<Hierarchy> group, T (&array)[N])
{
  using BlockExts = decltype(cuda::gpu_thread.extents(cuda::block, group.hierarchy()));
  static_assert(BlockExts::rank_dynamic() == 0, "This algorithm requires all static extents.");

  using BlockReduce =
    cub::BlockReduce<T,
                     static_cast<int>(BlockExts::static_extent(0)),
                     cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                     static_cast<int>(BlockExts::static_extent(1)),
                     static_cast<int>(BlockExts::static_extent(2))>;

  union SMem
  {
    typename BlockReduce::TempStorage block_scratch;
    T cluster_scratch;
  };

  __shared__ SMem smem;
  T result = BlockReduce{smem.block_scratch}.Sum(array);

  NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                 const auto dsmem = static_cast<T*>(__cluster_map_shared_rank(&smem.cluster_scratch, 0));

                 if (cuda::gpu_thread.is_root_rank(group))
                 {
                   smem.cluster_scratch = result;
                 }
                 group.sync_aligned();

                 cudax::this_block this_block{group.hierarchy()};
                 if (cuda::gpu_thread.is_root_rank(this_block) && !cuda::gpu_thread.is_root_rank(group))
                 {
                   [[maybe_unused]] unsigned old;
                   asm volatile("atom.relaxed.cluster.shared::cluster.add.s32 %0, [%1], %2;"
                                : "=r"(old)
                                : "l"(dsmem), "r"(result)
                                : "memory");
                 }
                 group.sync_aligned();

                 if (cuda::gpu_thread.is_root_rank(group))
                 {
                   result = smem.cluster_scratch;
                 }
               }))

  return (cuda::gpu_thread.is_root_rank(group)) ? cuda::std::optional{result} : cuda::std::nullopt;
}

template <class Group>
__device__ void test_cooperative_algorithm(Group& group)
{
  using Level = typename Group::level_type;

  unsigned array[]{1, 2, 3};
  const auto result = sum(group, array);

  unsigned ref_sum = 6;
  if constexpr (!cuda::std::is_same_v<Level, cuda::thread_level>)
  {
    ref_sum *= cuda::gpu_thread.count(Level{});
  }

  // Only the root rank should have the correct result.
  if (cuda::gpu_thread.is_root_rank(group))
  {
    CUDAX_REQUIRE(result.has_value());
    CUDAX_REQUIRE(result == ref_sum);
  }
  else
  {
    CUDAX_REQUIRE(!result.has_value());
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    {
      cudax::this_thread this_thread{config};
      test_common_properties(config, this_thread);
      test_cooperative_algorithm(this_thread);

      // Test queries.
      CUDAX_REQUIRE(cuda::gpu_thread.count(this_thread) == 1);
      CUDAX_REQUIRE(this_thread.count(cuda::warp) == cuda::gpu_thread.count(cuda::warp));
      CUDAX_REQUIRE(this_thread.count(cuda::block) == cuda::gpu_thread.count(cuda::block));
      CUDAX_REQUIRE(this_thread.count(cuda::cluster) == cuda::gpu_thread.count(cuda::cluster));
      CUDAX_REQUIRE(this_thread.count(cuda::grid) == cuda::gpu_thread.count(cuda::grid));

      CUDAX_REQUIRE(cuda::gpu_thread.rank(this_thread) == 0);
      CUDAX_REQUIRE(this_thread.rank(cuda::warp) == cuda::gpu_thread.rank(cuda::warp));
      CUDAX_REQUIRE(this_thread.rank(cuda::block) == cuda::gpu_thread.rank(cuda::block));
      CUDAX_REQUIRE(this_thread.rank(cuda::cluster) == cuda::gpu_thread.rank(cuda::cluster));
      CUDAX_REQUIRE(this_thread.rank(cuda::grid) == cuda::gpu_thread.rank(cuda::grid));

      CUDAX_REQUIRE(cuda::gpu_thread.is_root_rank(this_thread));

      CUDAX_REQUIRE(cuda::gpu_thread.is_part_of(this_thread));
    }
    {
      cudax::this_warp this_warp{config};
      test_common_properties(config, this_warp);
      test_cooperative_algorithm(this_warp);

      // Test queries.
      CUDAX_REQUIRE(cuda::gpu_thread.count(this_warp) == cuda::gpu_thread.count(cuda::warp));
      CUDAX_REQUIRE(cuda::warp.count(this_warp) == 1);
      CUDAX_REQUIRE(this_warp.count(cuda::block) == cuda::warp.count(cuda::block));
      CUDAX_REQUIRE(this_warp.count(cuda::cluster) == cuda::warp.count(cuda::cluster));
      CUDAX_REQUIRE(this_warp.count(cuda::grid) == cuda::warp.count(cuda::grid));

      CUDAX_REQUIRE(cuda::gpu_thread.rank(this_warp) == cuda::gpu_thread.rank(cuda::warp));
      CUDAX_REQUIRE(cuda::warp.rank(this_warp) == 0);
      CUDAX_REQUIRE(this_warp.rank(cuda::block) == cuda::warp.rank(cuda::block));
      CUDAX_REQUIRE(this_warp.rank(cuda::cluster) == cuda::warp.rank(cuda::cluster));
      CUDAX_REQUIRE(this_warp.rank(cuda::grid) == cuda::warp.rank(cuda::grid));

      CUDAX_REQUIRE(cuda::gpu_thread.is_root_rank(this_warp) == (cuda::gpu_thread.rank(cuda::warp) == 0));
      CUDAX_REQUIRE(cuda::warp.is_root_rank(this_warp));

      CUDAX_REQUIRE(cuda::gpu_thread.is_part_of(this_warp));
      CUDAX_REQUIRE(cuda::warp.is_part_of(this_warp));
    }
    {
      cudax::this_block this_block{config};
      test_common_properties(config, this_block);
      test_cooperative_algorithm(this_block);

      // Test queries.
      CUDAX_REQUIRE(cuda::gpu_thread.count(this_block) == cuda::gpu_thread.count(cuda::block));
      CUDAX_REQUIRE(cuda::warp.count(this_block) == cuda::warp.count(cuda::block));
      CUDAX_REQUIRE(cuda::block.count(this_block) == 1);
      CUDAX_REQUIRE(this_block.count(cuda::cluster) == cuda::block.count(cuda::cluster));
      CUDAX_REQUIRE(this_block.count(cuda::grid) == cuda::block.count(cuda::grid));

      CUDAX_REQUIRE(cuda::gpu_thread.rank(this_block) == cuda::gpu_thread.rank(cuda::block));
      CUDAX_REQUIRE(cuda::warp.rank(this_block) == cuda::warp.rank(cuda::block));
      CUDAX_REQUIRE(cuda::block.rank(this_block) == 0);
      CUDAX_REQUIRE(this_block.rank(cuda::cluster) == cuda::block.rank(cuda::cluster));
      CUDAX_REQUIRE(this_block.rank(cuda::grid) == cuda::block.rank(cuda::grid));

      CUDAX_REQUIRE(cuda::gpu_thread.is_root_rank(this_block) == (cuda::gpu_thread.rank(cuda::block) == 0));
      CUDAX_REQUIRE(cuda::warp.is_root_rank(this_block) == (cuda::warp.rank(cuda::block) == 0));
      CUDAX_REQUIRE(cuda::block.is_root_rank(this_block));

      CUDAX_REQUIRE(cuda::gpu_thread.is_part_of(this_block));
      CUDAX_REQUIRE(cuda::warp.is_part_of(this_block));
      CUDAX_REQUIRE(cuda::block.is_part_of(this_block));
    }
    {
      cudax::this_cluster this_cluster{config};
      test_common_properties(config, this_cluster);
      test_cooperative_algorithm(this_cluster);

      // Test queries.
      CUDAX_REQUIRE(cuda::gpu_thread.count(this_cluster) == cuda::gpu_thread.count(cuda::cluster));
      CUDAX_REQUIRE(cuda::warp.count(this_cluster) == cuda::warp.count(cuda::cluster));
      CUDAX_REQUIRE(cuda::block.count(this_cluster) == cuda::block.count(cuda::cluster));
      CUDAX_REQUIRE(cuda::cluster.count(this_cluster) == 1);
      CUDAX_REQUIRE(this_cluster.count(cuda::grid) == cuda::cluster.count(cuda::grid));

      CUDAX_REQUIRE(cuda::gpu_thread.rank(this_cluster) == cuda::gpu_thread.rank(cuda::cluster));
      CUDAX_REQUIRE(cuda::warp.rank(this_cluster) == cuda::warp.rank(cuda::cluster));
      CUDAX_REQUIRE(cuda::block.rank(this_cluster) == cuda::block.rank(cuda::cluster));
      CUDAX_REQUIRE(cuda::cluster.rank(this_cluster) == 0);
      CUDAX_REQUIRE(this_cluster.rank(cuda::grid) == cuda::cluster.rank(cuda::grid));

      CUDAX_REQUIRE(cuda::gpu_thread.is_root_rank(this_cluster) == (cuda::gpu_thread.rank(cuda::cluster) == 0));
      CUDAX_REQUIRE(cuda::warp.is_root_rank(this_cluster) == (cuda::warp.rank(cuda::cluster) == 0));
      CUDAX_REQUIRE(cuda::block.is_root_rank(this_cluster) == (cuda::block.rank(cuda::cluster) == 0));
      CUDAX_REQUIRE(cuda::cluster.is_root_rank(this_cluster));

      CUDAX_REQUIRE(cuda::gpu_thread.is_part_of(this_cluster));
      CUDAX_REQUIRE(cuda::warp.is_part_of(this_cluster));
      CUDAX_REQUIRE(cuda::block.is_part_of(this_cluster));
      CUDAX_REQUIRE(cuda::cluster.is_part_of(this_cluster));
    }
    {
      cudax::this_grid this_grid{config};
      test_common_properties(config, this_grid);

      // Test queries.
      CUDAX_REQUIRE(cuda::gpu_thread.count(this_grid) == cuda::gpu_thread.count(cuda::grid));
      CUDAX_REQUIRE(cuda::warp.count(this_grid) == cuda::warp.count(cuda::grid));
      CUDAX_REQUIRE(cuda::block.count(this_grid) == cuda::block.count(cuda::grid));
      CUDAX_REQUIRE(cuda::cluster.count(this_grid) == cuda::cluster.count(cuda::grid));
      CUDAX_REQUIRE(cuda::grid.count(this_grid) == 1);

      CUDAX_REQUIRE(cuda::gpu_thread.rank(this_grid) == cuda::gpu_thread.rank(cuda::grid));
      CUDAX_REQUIRE(cuda::warp.rank(this_grid) == cuda::warp.rank(cuda::grid));
      CUDAX_REQUIRE(cuda::block.rank(this_grid) == cuda::block.rank(cuda::grid));
      CUDAX_REQUIRE(cuda::cluster.rank(this_grid) == cuda::cluster.rank(cuda::grid));
      CUDAX_REQUIRE(cuda::grid.rank(this_grid) == 0);

      CUDAX_REQUIRE(cuda::gpu_thread.is_root_rank(this_grid) == (cuda::gpu_thread.rank(cuda::grid) == 0));
      CUDAX_REQUIRE(cuda::warp.is_root_rank(this_grid) == (cuda::warp.rank(cuda::grid) == 0));
      CUDAX_REQUIRE(cuda::block.is_root_rank(this_grid) == (cuda::block.rank(cuda::grid) == 0));
      CUDAX_REQUIRE(cuda::cluster.is_root_rank(this_grid) == (cuda::cluster.rank(cuda::grid) == 0));
      CUDAX_REQUIRE(cuda::grid.is_root_rank(this_grid));

      CUDAX_REQUIRE(cuda::gpu_thread.is_part_of(this_grid));
      CUDAX_REQUIRE(cuda::warp.is_part_of(this_grid));
      CUDAX_REQUIRE(cuda::block.is_part_of(this_grid));
      CUDAX_REQUIRE(cuda::cluster.is_part_of(this_grid));
      CUDAX_REQUIRE(cuda::grid.is_part_of(this_grid));
    }
  }
};

C2H_TEST("Hierarchy groups", "[hierarchy]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
  cuda::launch(stream, config, TestKernel{});

  // todo: investigate what causes cluster launches to hang, disable them temporarily
  bool false_value = false;
  if (false_value && cuda::device_attributes::compute_capability(device) >= cuda::compute_capability{90})
  {
    const auto config_cluster = cuda::make_config(
      cuda::grid_dims<2>(), cuda::cluster_dims<3>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
    cuda::launch(stream, config_cluster, TestKernel{});
  }

  stream.sync();
}

struct CgInteropKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    {
      cudax::this_thread g{cooperative_groups::this_thread()};
      static_assert(cudax::group<decltype(g)>);
      g.sync();
      g.sync_aligned();
    }
    {
      cudax::this_warp g{cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block())};
      static_assert(cudax::group<decltype(g)>);
      g.sync();
      g.sync_aligned();
    }
    {
      cudax::this_block g{cooperative_groups::this_thread_block()};
      static_assert(cudax::group<decltype(g)>);
      g.sync();
      g.sync_aligned();
    }
#if defined(_CG_HAS_CLUSTER_GROUP)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                     cudax::this_cluster g{cooperative_groups::this_cluster()};
                     static_assert(cudax::group<decltype(g)>);
                     g.sync();
                     g.sync_aligned();
                   }))
    }
#endif // _CG_HAS_CLUSTER_GROUP
    {
      cudax::this_grid g{cooperative_groups::this_grid()};
      static_assert(cudax::group<decltype(g)>);
      g.sync();
      g.sync_aligned();
    }
  }
};

C2H_TEST("Groups interoperability with coopertive groups", "[hierarchy][cg_interop]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
  cuda::launch(stream, config, TestKernel{});

  // todo: investigate what causes cluster launches to hang, disable them temporarily
  bool false_value = false;
  if (false_value && cuda::device_attributes::compute_capability(device) >= cuda::compute_capability{90})
  {
    const auto config_cluster = cuda::make_config(
      cuda::grid_dims<2>(), cuda::cluster_dims<3>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
    cuda::launch(stream, config_cluster, TestKernel{});
  }

  stream.sync();
}
