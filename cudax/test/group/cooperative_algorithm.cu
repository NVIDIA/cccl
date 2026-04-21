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

#include <cuda/atomic>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include <cooperative_groups.h>

#include "testing.cuh"

namespace
{
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
__device__ void test_cooperative_algorithm(Group&& group)
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
    test_cooperative_algorithm(cudax::this_thread{config});
    test_cooperative_algorithm(cudax::this_warp{config});
    test_cooperative_algorithm(cudax::this_block{config});
    test_cooperative_algorithm(cudax::this_cluster{config});
  }
};
} // namespace

C2H_TEST("Collective algorithm", "[group]")
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
