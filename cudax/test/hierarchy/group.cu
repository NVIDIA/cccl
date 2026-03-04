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

#if _CCCL_HAS_COOPERATIVE_GROUPS()
#  include <cooperative_groups.h>
#endif // _CCCL_HAS_COOPERATIVE_GROUPS()

#include "testing.cuh"

template <class... GArgs, class T, cuda::std::size_t N>
[[nodiscard]] __device__ T sum(cudax::thread_group<GArgs...> group, T (&array)[N])
{
  return cub::ThreadReduce(array, cuda::std::plus<T>{});
}

template <class... GArgs, class T, cuda::std::size_t N>
[[nodiscard]] __device__ T sum(cudax::warp_group<GArgs...> group, T (&array)[N])
{
  using WarpReduce = cub::WarpReduce<T>;

  __shared__ typename WarpReduce::TempStorage temp_storage;

  const auto partial = cub::ThreadReduce(array, cuda::std::plus<T>{});
  return WarpReduce{temp_storage}.Sum(partial);
}

template <class... GArgs, class T, cuda::std::size_t N>
[[nodiscard]] __device__ T sum(cudax::block_group<GArgs...> group, T (&array)[N])
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

      auto this_thread = cudax::this_thread(config);

      this_thread.sync();

      const auto result = sum(this_thread, array);
      CUDAX_REQUIRE(result == 6);

      CUDAX_REQUIRE(cuda::gpu_thread.count(this_thread) == 1);
      CUDAX_REQUIRE(cuda::gpu_thread.rank(this_thread) == 0);
      CUDAX_REQUIRE(this_thread.count(cuda::warp) == cuda::gpu_thread.count(cuda::warp));
      CUDAX_REQUIRE(this_thread.rank(cuda::warp) == cuda::gpu_thread.rank(cuda::warp));
      CUDAX_REQUIRE(this_thread.count(cuda::block) == cuda::gpu_thread.count(cuda::block));
      CUDAX_REQUIRE(this_thread.rank(cuda::block) == cuda::gpu_thread.rank(cuda::block));
      CUDAX_REQUIRE(this_thread.count(cuda::cluster) == cuda::gpu_thread.count(cuda::cluster));
      CUDAX_REQUIRE(this_thread.rank(cuda::cluster) == cuda::gpu_thread.rank(cuda::cluster));
      CUDAX_REQUIRE(this_thread.count(cuda::grid) == cuda::gpu_thread.count(cuda::grid));
      CUDAX_REQUIRE(this_thread.rank(cuda::grid) == cuda::gpu_thread.rank(cuda::grid));
    }
    {
      unsigned array[]{1, 2, 3};

      auto this_warp = cudax::this_warp(config);
      this_warp.sync();

      const auto result = sum(this_warp, array);
      if (cuda::gpu_thread.rank(cuda::warp) == 0)
      {
        CUDAX_REQUIRE(result == 6 * cuda::gpu_thread.count(cuda::warp));
      }

      CUDAX_REQUIRE(cuda::gpu_thread.count(this_warp) == cuda::gpu_thread.count(cuda::warp));
      CUDAX_REQUIRE(cuda::gpu_thread.rank(this_warp) == cuda::gpu_thread.rank(cuda::warp));
      CUDAX_REQUIRE(cuda::warp.count(this_warp) == 1);
      CUDAX_REQUIRE(cuda::warp.rank(this_warp) == 0);
      CUDAX_REQUIRE(this_warp.count(cuda::block) == cuda::warp.count(cuda::block));
      CUDAX_REQUIRE(this_warp.rank(cuda::block) == cuda::warp.rank(cuda::block));
      CUDAX_REQUIRE(this_warp.count(cuda::cluster) == cuda::warp.count(cuda::cluster));
      CUDAX_REQUIRE(this_warp.rank(cuda::cluster) == cuda::warp.rank(cuda::cluster));
      CUDAX_REQUIRE(this_warp.count(cuda::grid) == cuda::warp.count(cuda::grid));
      CUDAX_REQUIRE(this_warp.rank(cuda::grid) == cuda::warp.rank(cuda::grid));
    }
    {
      unsigned array[]{1, 2, 3};

      auto this_block = cudax::this_block(config);
      this_block.sync();

      const auto result = sum(this_block, array);
      if (cuda::gpu_thread.rank(cuda::block) == 0)
      {
        CUDAX_REQUIRE(result == 6 * cuda::gpu_thread.count(cuda::block));
      }

      CUDAX_REQUIRE(cuda::gpu_thread.count(this_block) == cuda::gpu_thread.count(cuda::block));
      CUDAX_REQUIRE(cuda::gpu_thread.rank(this_block) == cuda::gpu_thread.rank(cuda::block));
      CUDAX_REQUIRE(cuda::warp.count(this_block) == cuda::warp.count(cuda::block));
      CUDAX_REQUIRE(cuda::warp.rank(this_block) == cuda::warp.rank(cuda::block));
      CUDAX_REQUIRE(cuda::block.count(this_block) == 1);
      CUDAX_REQUIRE(cuda::block.rank(this_block) == 0);
      CUDAX_REQUIRE(this_block.count(cuda::cluster) == cuda::block.count(cuda::cluster));
      CUDAX_REQUIRE(this_block.rank(cuda::cluster) == cuda::block.rank(cuda::cluster));
      CUDAX_REQUIRE(this_block.count(cuda::grid) == cuda::block.count(cuda::grid));
      CUDAX_REQUIRE(this_block.rank(cuda::grid) == cuda::block.rank(cuda::grid));
    }
    {
      auto this_cluster = cudax::this_cluster(config);
      CUDAX_REQUIRE(this_cluster.count(cuda::grid) == cuda::cluster.count(cuda::grid));
      CUDAX_REQUIRE(this_cluster.rank(cuda::grid) == cuda::cluster.rank(cuda::grid));
      this_cluster.sync();

      CUDAX_REQUIRE(cuda::gpu_thread.count(this_cluster) == cuda::gpu_thread.count(cuda::cluster));
      CUDAX_REQUIRE(cuda::gpu_thread.rank(this_cluster) == cuda::gpu_thread.rank(cuda::cluster));
      CUDAX_REQUIRE(cuda::warp.count(this_cluster) == cuda::warp.count(cuda::cluster));
      CUDAX_REQUIRE(cuda::warp.rank(this_cluster) == cuda::warp.rank(cuda::cluster));
      CUDAX_REQUIRE(cuda::block.count(this_cluster) == cuda::block.count(cuda::cluster));
      CUDAX_REQUIRE(cuda::block.rank(this_cluster) == cuda::block.rank(cuda::cluster));
      CUDAX_REQUIRE(cuda::cluster.count(this_cluster) == 1);
      CUDAX_REQUIRE(cuda::cluster.rank(this_cluster) == 0);
      CUDAX_REQUIRE(this_cluster.count(cuda::grid) == cuda::cluster.count(cuda::grid));
      CUDAX_REQUIRE(this_cluster.rank(cuda::grid) == cuda::cluster.rank(cuda::grid));
    }
    {
      auto this_grid = cudax::this_grid(config);
      this_grid.sync();

      CUDAX_REQUIRE(cuda::gpu_thread.count(this_grid) == cuda::gpu_thread.count(cuda::grid));
      CUDAX_REQUIRE(cuda::gpu_thread.rank(this_grid) == cuda::gpu_thread.rank(cuda::grid));
      CUDAX_REQUIRE(cuda::warp.count(this_grid) == cuda::warp.count(cuda::grid));
      CUDAX_REQUIRE(cuda::warp.rank(this_grid) == cuda::warp.rank(cuda::grid));
      CUDAX_REQUIRE(cuda::block.count(this_grid) == cuda::block.count(cuda::grid));
      CUDAX_REQUIRE(cuda::block.rank(this_grid) == cuda::block.rank(cuda::grid));
      CUDAX_REQUIRE(cuda::cluster.count(this_grid) == cuda::cluster.count(cuda::grid));
      CUDAX_REQUIRE(cuda::cluster.rank(this_grid) == cuda::cluster.rank(cuda::grid));
      CUDAX_REQUIRE(cuda::grid.count(this_grid) == 1);
      CUDAX_REQUIRE(cuda::grid.rank(this_grid) == 0);
    }
  }
};

C2H_TEST("Hierarchy groups", "[hierarchy]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<32>(), cuda::cooperative_launch{});

  cuda::launch(stream, config, TestKernel{});

  stream.sync();
}

struct ThreadGroupKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    // 1. This thread
    {
      auto group = cudax::this_thread(config);
      group.sync();
      CUDAX_REQUIRE(group.is_part_of(cuda::gpu_thread));
    }

    // 2. Thread groups in warp
    {
      // a. Group by
      {
        cudax::thread_group group{cuda::warp, cudax::group_by<4>, config};
        group.sync();
        CUDAX_REQUIRE(group.is_part_of(cuda::gpu_thread));
      }

      // b. Group as
      {
        constexpr unsigned mapping[]{1, 4, 4, 1};
        cudax::thread_group group{cuda::warp, cudax::group_as{mapping}, config};
        group.sync();
        CUDAX_REQUIRE(group.is_part_of(cuda::gpu_thread) == (cuda::gpu_thread.rank(cuda::warp) < 10));
      }

      // c. Generic group with default rank
      {
        cudax::thread_group group{
          cuda::warp,
          8,
          [](unsigned thread_rank) {
            return (thread_rank % 8 + 1) % 8;
          },
          config};
        group.sync();
        CUDAX_REQUIRE(group.is_part_of(cuda::gpu_thread));
      }

      // d. Generic group with manual rank
      {
        cudax::thread_group group{
          cuda::warp,
          8,
          [](unsigned thread_rank) {
            return cuda::std::tuple{(thread_rank % 8 + 1) % 8, (thread_rank / 8 + 1) % 4};
          },
          config};
        group.sync();
        CUDAX_REQUIRE(group.is_part_of(cuda::gpu_thread));
      }

      // e. Generic group with optionals and default rank
      {
        cudax::thread_group group{
          cuda::warp,
          7,
          [](unsigned thread_rank) {
            const auto grank = (thread_rank % 8 + 1) % 8;
            return (grank != 0) ? cuda::std::optional{grank} : cuda::std::nullopt;
          },
          config};
        group.sync();

        const auto is_part_of = ((cuda::gpu_thread.rank(cuda::warp) % 8 + 1) % 8 != 0);
        CUDAX_REQUIRE(group.is_part_of(cuda::gpu_thread) == is_part_of);
      }

      // f. Generic group with optionals and manual rank
      {
        cudax::thread_group group{
          cuda::warp,
          7,
          [](unsigned thread_rank) {
            const auto grank = (thread_rank % 8 + 1) % 8;
            const auto rank  = (thread_rank / 8 + 1) % 4;
            return (grank != 0) ? cuda::std::optional{cuda::std::tuple{grank, rank}} : cuda::std::nullopt;
          },
          config};
        group.sync();

        const auto is_part_of = ((cuda::gpu_thread.rank(cuda::warp) % 8 + 1) % 8 != 0);
        CUDAX_REQUIRE(group.is_part_of(cuda::gpu_thread) == is_part_of);
      }
    }

    // 3. Thread groups in block
    {
      // a. Group by
      {
        using Barriers = cuda::barrier<cuda::thread_scope_block>[2];
        __shared__ alignas(Barriers) unsigned char barriers_storage[sizeof(Barriers)];

        Barriers& barriers = reinterpret_cast<Barriers&>(barriers_storage);

        cudax::thread_group group{cuda::block, cudax::group_by<32>, barriers, config};
        group.sync();
        CUDAX_REQUIRE(group.is_part_of(cuda::gpu_thread));
      }

      // b. Group as
      {
        using Barriers = cuda::barrier<cuda::thread_scope_block>[4];
        __shared__ alignas(Barriers) unsigned char barriers_storage[sizeof(Barriers)];

        Barriers& barriers = reinterpret_cast<Barriers&>(barriers_storage);

        constexpr unsigned mapping[]{10, 10, 10, 10};

        cudax::thread_group group{cuda::block, cudax::group_as{mapping}, barriers, config};
        group.sync();
        CUDAX_REQUIRE(group.is_part_of(cuda::gpu_thread) == (cuda::gpu_thread.rank(cuda::block) < 40));
      }

      // c. Generic group with manual rank
      {
        using Barriers = cuda::barrier<cuda::thread_scope_block>[8];
        __shared__ alignas(Barriers) unsigned char barriers_storage[sizeof(Barriers)];

        Barriers& barriers = reinterpret_cast<Barriers&>(barriers_storage);

        cudax::thread_group group{
          cuda::block,
          8,
          [](unsigned thread_rank) {
            return cuda::std::tuple{thread_rank % 8, 8, thread_rank / 8};
          },
          barriers,
          config};
        group.sync();
        CUDAX_REQUIRE(group.is_part_of(cuda::gpu_thread));
      }

      // d. Generic group with optionals and manual rank
      {
        using Barriers = cuda::barrier<cuda::thread_scope_block>[8];
        __shared__ alignas(Barriers) unsigned char barriers_storage[sizeof(Barriers)];

        Barriers& barriers = reinterpret_cast<Barriers&>(barriers_storage);

        cudax::thread_group group{
          cuda::block,
          7,
          [](unsigned thread_rank) {
            const auto grank = thread_rank % 8;
            const auto rank  = thread_rank / 8;
            return (grank != 0) ? cuda::std::optional{cuda::std::tuple{grank, 8, rank}} : cuda::std::nullopt;
          },
          barriers,
          config};
        group.sync();

        const auto is_part_of = (cuda::gpu_thread.rank(cuda::block) % 8 != 0);
        CUDAX_REQUIRE(group.is_part_of(cuda::gpu_thread) == is_part_of);
      }
    }
  }
};

C2H_TEST("Thread groups", "[hierarchy][thread_group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<64>(), cuda::cooperative_launch{});

  cuda::launch(stream, config, ThreadGroupKernel{});

  stream.sync();
}

struct InteropKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    {
      cudax::thread_group g{cooperative_groups::this_thread()};
      g.sync();
    }
    {
      cudax::block_group g{cooperative_groups::this_thread_block()};
      g.sync();
    }
#if _CCCL_HAS_COOPERATIVE_GROUPS() && defined(_CG_HAS_CLUSTER_GROUP)
    {
      cudax::cluster_group g{cooperative_groups::this_cluster()};
      g.sync();
    }
#endif // _CCCL_HAS_COOPERATIVE_GROUPS() && _CG_HAS_CLUSTER_GROUP
    {
      cudax::grid_group g{cooperative_groups::this_grid()};
      g.sync();
    }
  }
};

C2H_TEST("Groups interoperability with coopertive groups", "[hierarchy][cg_interop]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<64>(), cuda::cooperative_launch{});

  cuda::launch(stream, config, InteropKernel{});

  stream.sync();
}