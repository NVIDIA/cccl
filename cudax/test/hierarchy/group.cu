//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/stream>

#include <cuda/experimental/hierarchy.cuh>

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
      CUDAX_REQUIRE(this_thread.count(cuda::grid) == cuda::gpu_thread.count(cuda::grid));
      CUDAX_REQUIRE(this_thread.rank(cuda::grid) == cuda::gpu_thread.rank(cuda::grid));
      this_thread.sync();

      const auto result = sum(this_thread, array);
      CUDAX_REQUIRE(result == 6);
    }
    {
      unsigned array[]{1, 2, 3};

      auto this_warp = cudax::this_warp(config);
      CUDAX_REQUIRE(this_warp.count(cuda::grid) == cuda::warp.count(cuda::grid));
      CUDAX_REQUIRE(this_warp.rank(cuda::grid) == cuda::warp.rank(cuda::grid));
      this_warp.sync();

      const auto result = sum(this_warp, array);
      if (cuda::gpu_thread.rank(cuda::warp) == 0)
      {
        CUDAX_REQUIRE(result == 6 * cuda::gpu_thread.count(cuda::warp));
      }
    }
    {
      unsigned array[]{1, 2, 3};

      auto this_block = cudax::this_block(config);
      CUDAX_REQUIRE(this_block.count(cuda::grid) == cuda::block.count(cuda::grid));
      CUDAX_REQUIRE(this_block.rank(cuda::grid) == cuda::block.rank(cuda::grid));
      this_block.sync();

      const auto result = sum(this_block, array);
      if (cuda::gpu_thread.rank(cuda::block) == 0)
      {
        CUDAX_REQUIRE(result == 6 * cuda::gpu_thread.count(cuda::block));
      }
    }
    {
      auto this_cluster = cudax::this_cluster(config);
      CUDAX_REQUIRE(this_cluster.count(cuda::grid) == cuda::cluster.count(cuda::grid));
      CUDAX_REQUIRE(this_cluster.rank(cuda::grid) == cuda::cluster.rank(cuda::grid));
      this_cluster.sync();
    }
    {
      auto this_grid = cudax::this_grid(config);
      this_grid.sync();
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
