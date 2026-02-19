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

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/stream>

#include <cuda/experimental/hierarchy.cuh>

#include "testing.cuh"

template <class Group,
          class T,
          cuda::std::size_t N,
          cuda::std::enable_if_t<cudax::hierarchy_group<Group>, int>                                        = 0,
          cuda::std::enable_if_t<cuda::std::is_same_v<typename Group::level_type, cuda::thread_level>, int> = 0>
[[nodiscard]] __device__ T sum(Group group, T (&array)[N])
{
  return cub::ThreadReduce(array, cuda::std::plus<T>{});
}

template <class Hierarchy, class T, cuda::std::size_t N>
[[nodiscard]] __device__ T sum(cudax::this_warp<Hierarchy> group, T (&array)[N])
{
  using WarpReduce = cub::WarpReduce<T>;

  __shared__ typename WarpReduce::TempStorage temp_storage;

  const auto partial = cub::ThreadReduce(array, cuda::std::plus<T>{});
  return WarpReduce{temp_storage}.Sum(partial);
}

template <class Hierarchy, class T, cuda::std::size_t N>
[[nodiscard]] __device__ T sum(cudax::this_block<Hierarchy> group, T (&array)[N])
{
  // todo: Replace 32 with value from group.
  using BlockReduce = cub::BlockReduce<T, 32>;

  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce{temp_storage}.Sum(array);
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    {
      unsigned array[]{1, 2, 3};

      cudax::this_thread thread{config};
      static_assert(cudax::hierarchy_group<decltype(thread)>);

      thread.sync();

      const auto result = sum(thread, array);
      CUDAX_REQUIRE(result == 6);

      CUDAX_REQUIRE(cuda::gpu_thread.count(thread) == 1);
      CUDAX_REQUIRE(cuda::gpu_thread.rank(thread) == 0);
      CUDAX_REQUIRE(thread.count(cuda::warp) == cuda::gpu_thread.count(cuda::warp));
      CUDAX_REQUIRE(thread.rank(cuda::warp) == cuda::gpu_thread.rank(cuda::warp));
      CUDAX_REQUIRE(thread.count(cuda::block) == cuda::gpu_thread.count(cuda::block));
      CUDAX_REQUIRE(thread.rank(cuda::block) == cuda::gpu_thread.rank(cuda::block));
      CUDAX_REQUIRE(thread.count(cuda::cluster) == cuda::gpu_thread.count(cuda::cluster));
      CUDAX_REQUIRE(thread.rank(cuda::cluster) == cuda::gpu_thread.rank(cuda::cluster));
      CUDAX_REQUIRE(thread.count(cuda::grid) == cuda::gpu_thread.count(cuda::grid));
      CUDAX_REQUIRE(thread.rank(cuda::grid) == cuda::gpu_thread.rank(cuda::grid));
    }
    {
      unsigned array[]{1, 2, 3};

      cudax::this_warp warp{config};
      static_assert(cudax::hierarchy_group<decltype(warp)>);

      warp.sync();

      const auto result = sum(warp, array);
      if (cuda::gpu_thread.rank(cuda::warp) == 0)
      {
        CUDAX_REQUIRE(result == 6 * cuda::gpu_thread.count(cuda::warp));
      }

      CUDAX_REQUIRE(cuda::gpu_thread.count(warp) == cuda::gpu_thread.count(cuda::warp));
      CUDAX_REQUIRE(cuda::gpu_thread.rank(warp) == cuda::gpu_thread.rank(cuda::warp));
      CUDAX_REQUIRE(cuda::warp.count(warp) == 1);
      CUDAX_REQUIRE(cuda::warp.rank(warp) == 0);
      CUDAX_REQUIRE(warp.count(cuda::block) == cuda::warp.count(cuda::block));
      CUDAX_REQUIRE(warp.rank(cuda::block) == cuda::warp.rank(cuda::block));
      CUDAX_REQUIRE(warp.count(cuda::cluster) == cuda::warp.count(cuda::cluster));
      CUDAX_REQUIRE(warp.rank(cuda::cluster) == cuda::warp.rank(cuda::cluster));
      CUDAX_REQUIRE(warp.count(cuda::grid) == cuda::warp.count(cuda::grid));
      CUDAX_REQUIRE(warp.rank(cuda::grid) == cuda::warp.rank(cuda::grid));
    }
    {
      unsigned array[]{1, 2, 3};

      cudax::this_block block{config};
      static_assert(cudax::hierarchy_group<decltype(block)>);

      block.sync();

      const auto result = sum(block, array);
      if (cuda::gpu_thread.rank(cuda::block) == 0)
      {
        CUDAX_REQUIRE(result == 6 * cuda::gpu_thread.count(cuda::block));
      }

      CUDAX_REQUIRE(cuda::gpu_thread.count(block) == cuda::gpu_thread.count(cuda::block));
      CUDAX_REQUIRE(cuda::gpu_thread.rank(block) == cuda::gpu_thread.rank(cuda::block));
      CUDAX_REQUIRE(cuda::warp.count(block) == cuda::warp.count(cuda::block));
      CUDAX_REQUIRE(cuda::warp.rank(block) == cuda::warp.rank(cuda::block));
      CUDAX_REQUIRE(cuda::block.count(block) == 1);
      CUDAX_REQUIRE(cuda::block.rank(block) == 0);
      CUDAX_REQUIRE(block.count(cuda::cluster) == cuda::block.count(cuda::cluster));
      CUDAX_REQUIRE(block.rank(cuda::cluster) == cuda::block.rank(cuda::cluster));
      CUDAX_REQUIRE(block.count(cuda::grid) == cuda::block.count(cuda::grid));
      CUDAX_REQUIRE(block.rank(cuda::grid) == cuda::block.rank(cuda::grid));
    }
    {
      cudax::this_cluster cluster{config};
      static_assert(cudax::hierarchy_group<decltype(cluster)>);

      cluster.sync();

      CUDAX_REQUIRE(cuda::gpu_thread.count(cluster) == cuda::gpu_thread.count(cuda::cluster));
      CUDAX_REQUIRE(cuda::gpu_thread.rank(cluster) == cuda::gpu_thread.rank(cuda::cluster));
      CUDAX_REQUIRE(cuda::warp.count(cluster) == cuda::warp.count(cuda::cluster));
      CUDAX_REQUIRE(cuda::warp.rank(cluster) == cuda::warp.rank(cuda::cluster));
      CUDAX_REQUIRE(cuda::block.count(cluster) == cuda::block.count(cuda::cluster));
      CUDAX_REQUIRE(cuda::block.rank(cluster) == cuda::block.rank(cuda::cluster));
      CUDAX_REQUIRE(cuda::cluster.count(cluster) == 1);
      CUDAX_REQUIRE(cuda::cluster.rank(cluster) == 0);
      CUDAX_REQUIRE(cluster.count(cuda::grid) == cuda::cluster.count(cuda::grid));
      CUDAX_REQUIRE(cluster.rank(cuda::grid) == cuda::cluster.rank(cuda::grid));
    }
    {
      cudax::this_grid grid{config};
      static_assert(cudax::hierarchy_group<decltype(grid)>);

      grid.sync();

      CUDAX_REQUIRE(cuda::gpu_thread.count(grid) == cuda::gpu_thread.count(cuda::grid));
      CUDAX_REQUIRE(cuda::gpu_thread.rank(grid) == cuda::gpu_thread.rank(cuda::grid));
      CUDAX_REQUIRE(cuda::warp.count(grid) == cuda::warp.count(cuda::grid));
      CUDAX_REQUIRE(cuda::warp.rank(grid) == cuda::warp.rank(cuda::grid));
      CUDAX_REQUIRE(cuda::block.count(grid) == cuda::block.count(cuda::grid));
      CUDAX_REQUIRE(cuda::block.rank(grid) == cuda::block.rank(cuda::grid));
      CUDAX_REQUIRE(cuda::cluster.count(grid) == cuda::cluster.count(cuda::grid));
      CUDAX_REQUIRE(cuda::cluster.rank(grid) == cuda::cluster.rank(cuda::grid));
      CUDAX_REQUIRE(cuda::grid.count(grid) == 1);
      CUDAX_REQUIRE(cuda::grid.rank(grid) == 0);
    }
  }
};

C2H_TEST("This hierarchy groups", "[hierarchy][this_group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<32>(), cuda::cooperative_launch{});

  cuda::launch(stream, config, TestKernel{});

  stream.sync();
}

// missing level_type and sync()
struct InvalidGroup1
{};

// missing sync()
struct InvalidGroup2
{
  using level_type = cuda::thread_level;
};

// missing level_type
struct InvalidGroup3
{
  __device__ void sync();
};

struct ValidGroup
{
  using level_type = cuda::thread_level;
  __device__ void sync();
};

static_assert(!cudax::hierarchy_group<InvalidGroup1>);
static_assert(!cudax::hierarchy_group<InvalidGroup2>);
static_assert(!cudax::hierarchy_group<InvalidGroup3>);
static_assert(cudax::hierarchy_group<ValidGroup>);

struct TestKernel2
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    cudax::thread_group even_or_odd_threads_in_warp{
      cuda::warp,
      [](auto rank) {
        return rank % 2 == 0;
      },
      config};
    even_or_odd_threads_in_warp.sync();

    CUDAX_REQUIRE(even_or_odd_threads_in_warp.is_part_of(cuda::gpu_thread));
    CUDAX_REQUIRE(even_or_odd_threads_in_warp.count(cuda::warp) == 2);
    CUDAX_REQUIRE(even_or_odd_threads_in_warp.rank(cuda::warp) == (cuda::gpu_thread.rank(cuda::warp) % 2 == 0));

    cudax::thread_group even_threads_in_warp{
      cuda::warp,
      [](auto rank) {
        return (rank % 2 == 0) ? cuda::std::optional{0} : cuda::std::nullopt;
      },
      config};
    even_threads_in_warp.sync();

    CUDAX_REQUIRE(even_threads_in_warp.is_part_of(cuda::gpu_thread) == (cuda::gpu_thread.rank(cuda::warp) % 2 == 0));
    CUDAX_REQUIRE(even_threads_in_warp.count(cuda::warp) == 1);
    if (even_threads_in_warp.is_part_of(cuda::gpu_thread))
    {
      CUDAX_REQUIRE(even_threads_in_warp.rank(cuda::warp) == 0);
    }

    cudax::thread_group grouped_threads_in_warp{cuda::warp, cudax::group_by<4>, config};
    grouped_threads_in_warp.sync();

    CUDAX_REQUIRE(grouped_threads_in_warp.is_part_of(cuda::gpu_thread));
    CUDAX_REQUIRE(grouped_threads_in_warp.count(cuda::warp) == 8);
    CUDAX_REQUIRE(grouped_threads_in_warp.rank(cuda::warp) == cuda::gpu_thread.rank(cuda::warp) / 4);
  }
};

C2H_TEST("Generic hierarchy groups", "[hierarchy][generic_group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<32>(), cuda::cooperative_launch{});

  cuda::launch(stream, config, TestKernel2{});

  stream.sync();
}
