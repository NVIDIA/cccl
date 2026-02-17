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

// Disclaimer
//   This not an attempt to design a cuda::coop replacement for cub algorithms. This example should help us detect and
//   understand the limitations of the current hierarchy groups design.
namespace coop
{
template <class T>
struct shared_memory_scratch
{
  T& ref;
};

template <class T>
__host__ __device__ shared_memory_scratch(T&) -> shared_memory_scratch<T>;

struct sum_t
{
  template <class T>
  using _warp_reduce_type = cub::WarpReduce<T>;

  template <class Exts, class T>
  using _block_reduce_type =
    cub::BlockReduce<T,
                     Exts::static_extent(0),
                     cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                     Exts::static_extent(1),
                     Exts::static_extent(2)>;

  explicit sum_t() = default;

  // todo: would it be nice if this_thread_group would be a concept?
  // We could do operator()(cudax::this_thread_group auto group, ...) in C++20
  template <class Hier, class T, cuda::std::size_t N, class ScratchT>
  __device__ T operator()(cudax::this_thread_group<Hier> group, T (&array)[N], shared_memory_scratch<ScratchT>) const
  {
    return cub::ThreadReduce(array, cuda::std::plus<T>{});
  }

  template <class Hier, class T, cuda::std::size_t N, class ScratchT>
  [[nodiscard]] __device__ T
  operator()(cudax::this_warp_group<Hier> group, T (&array)[N], shared_memory_scratch<ScratchT> scratch) const
  {
    using WarpReduce   = _warp_reduce_type<T>;
    const auto partial = cub::ThreadReduce(array, cuda::std::plus<T>{});
    return WarpReduce{reinterpret_cast<typename WarpReduce::TempStorage&>(scratch.ref)}.Sum(partial);
  }

  template <class Hier, class T, cuda::std::size_t N, class ScratchT>
  [[nodiscard]] __device__ T
  operator()(cudax::this_block_group<Hier> group, T (&array)[N], shared_memory_scratch<ScratchT> scratch) const
  {
    using Hierarchy = typename cudax::this_block_group<Hier>::hierarchy_type;
    using BlockDesc = typename Hierarchy::template level_desc_type<cuda::block_level>;
    using BlockExts = typename BlockDesc::extents_type;

    static_assert(BlockExts::rank_dynamic() == 0, "blocks of dynamically known size are unsupported");

    using BlockReduce = _block_reduce_type<BlockExts, T>;
    return BlockReduce{reinterpret_cast<typename BlockReduce::TempStorage&>(scratch.ref)}.Sum(array);
  }
};

_CCCL_GLOBAL_CONSTANT sum_t sum;

template <class T>
struct value_type_t
{
  using type              = T;
  explicit value_type_t() = default;
};

template <class T>
_CCCL_GLOBAL_CONSTANT value_type_t<T> value_type;

struct required_shared_memory_result
{
  cuda::std::size_t size;
  cuda::std::size_t alignment;
};

template <class Group, class T>
[[nodiscard]] __host__ __device__ constexpr required_shared_memory_result
required_shared_memory(const sum_t&, const value_type_t<T>&)
{
  using GroupLevel = typename Group::level_type;
  if constexpr (cuda::std::is_same_v<GroupLevel, cuda::thread_level>)
  {
    return {0, 0};
  }
  else if constexpr (cuda::std::is_same_v<GroupLevel, cuda::warp_level>)
  {
    using TempStorage = typename sum_t::_warp_reduce_type<T>::TempStorage;
    return {sizeof(TempStorage), alignof(TempStorage)};
  }
  else
  {
    using Hierarchy = typename Group::hierarchy_type;
    using BlockDesc = typename Hierarchy::template level_desc_type<cuda::block_level>;
    using BlockExts = typename BlockDesc::extents_type;

    static_assert(BlockExts::rank_dynamic() == 0, "blocks of dynamically known size are unsupported");

    using TempStorage = typename sum_t::_block_reduce_type<BlockExts, T>::TempStorage;
    return {sizeof(TempStorage), alignof(TempStorage)};
  }
}
} // namespace coop

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    // todo: I want to write:
    //         coop::requires_shared_memory(coop::sum, cudax::this_thread(config), coop::value_type<unsigned>);
    //
    //       The problem is that the group holds a reference to the hierarchy, but that's unavailable in constexpr
    //       context.
    constexpr auto thread_required_smem =
      coop::required_shared_memory<decltype(cudax::this_thread(config))>(coop::sum, coop::value_type<unsigned>);
    constexpr auto warp_required_smem =
      coop::required_shared_memory<decltype(cudax::this_warp(config))>(coop::sum, coop::value_type<unsigned>);
    constexpr auto block_required_smem =
      coop::required_shared_memory<decltype(cudax::this_block(config))>(coop::sum, coop::value_type<unsigned>);
    constexpr auto tmp_storage_size =
      cuda::std::max({thread_required_smem.size, warp_required_smem.size, block_required_smem.size});
    constexpr auto tmp_storage_alignment =
      cuda::std::max({thread_required_smem.alignment, warp_required_smem.alignment, block_required_smem.alignment});

    __shared__ alignas(tmp_storage_alignment) unsigned char tmp_storage[tmp_storage_size];

    {
      unsigned array[]{1, 2, 3};

      auto this_thread = cudax::this_thread(config);

      this_thread.sync();

      const auto result = coop::sum(this_thread, array, coop::shared_memory_scratch{tmp_storage});
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

      const auto result = coop::sum(this_warp, array, coop::shared_memory_scratch{tmp_storage});
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

      const auto result = coop::sum(this_block, array, coop::shared_memory_scratch{tmp_storage});
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

  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<64>(), cuda::cooperative_launch{});

  cuda::launch(stream, config, TestKernel{});

  stream.sync();
}
