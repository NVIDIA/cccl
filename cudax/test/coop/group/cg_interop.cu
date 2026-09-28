//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <cuda/utility>

#include <cuda/experimental/coop/group>

#include <cooperative_groups.h>

#include "group_testing.cuh"

namespace cg = cooperative_groups;

template <class Config>
__device__ void test_cg_interop(Config config)
{
  using Hierarchy = typename Config::hierarchy_type;

  // Test cg::this_grid().
  {
    auto cg_group = cg::this_grid();

    static_assert(noexcept(cudax::coop::make_cg_equivalent_group(cg_group, config)));
    auto cccl_group = cudax::coop::make_cg_equivalent_group(cg_group, config);

    static_assert(cuda::std::is_same_v<cudax::coop::this_grid<Hierarchy>, decltype(cccl_group)>);
    REQUIRE(cuda::gpu_thread.count(cccl_group) == cg_group.size());
    REQUIRE(cuda::gpu_thread.rank(cccl_group) == cg_group.thread_rank());
    cccl_group.sync();
  }

  // Test cg::this_cluster().
#if defined(_CG_HAS_CLUSTER_GROUP)
  {
    auto cg_group = cg::this_cluster();

    static_assert(noexcept(cudax::coop::make_cg_equivalent_group(cg_group, config)));
    auto cccl_group = cudax::coop::make_cg_equivalent_group(cg_group, config);

    static_assert(cuda::std::is_same_v<cudax::coop::this_cluster<Hierarchy>, decltype(cccl_group)>);
    REQUIRE(cuda::gpu_thread.count(cccl_group) == cg_group.size());
    REQUIRE(cuda::gpu_thread.rank(cccl_group) == cg_group.thread_rank());
    cccl_group.sync();
  }
#endif // _CG_HAS_CLUSTER_GROUP

  // Test cg::this_thread_block().
  {
    auto cg_group = cg::this_thread_block();

    static_assert(noexcept(cudax::coop::make_cg_equivalent_group(cg_group, config)));
    auto cccl_group = cudax::coop::make_cg_equivalent_group(cg_group, config);

    static_assert(cuda::std::is_same_v<cudax::coop::this_block<Hierarchy>, decltype(cccl_group)>);
    REQUIRE(cuda::gpu_thread.count(cccl_group) == cg_group.size());
    REQUIRE(cuda::gpu_thread.rank(cccl_group) == cg_group.thread_rank());
    cccl_group.sync();
  }

  // Test cg::this_thread().
  {
    auto cg_group = cg::this_thread();

    static_assert(noexcept(cudax::coop::make_cg_equivalent_group(cg_group, config)));
    auto cccl_group = cudax::coop::make_cg_equivalent_group(cg_group, config);

    static_assert(cuda::std::is_same_v<cudax::coop::this_thread<Hierarchy>, decltype(cccl_group)>);
    REQUIRE(cuda::gpu_thread.count(cccl_group) == cg_group.size());
    REQUIRE(cuda::gpu_thread.rank(cccl_group) == cg_group.thread_rank());
    cccl_group.sync();
  }

  // Test cg::tiled_partition<N>(cg::this_thread_block()) with N < 32.
  cuda::static_for<cuda::ilog2(16)>([&](auto i) {
    constexpr auto n = cuda::ipow(2, i());
    const cudax::coop::this_warp this_warp{config};

    auto cg_group = cg::tiled_partition<n>(cg::this_thread_block());

    static_assert(noexcept(cudax::coop::make_cg_equivalent_group(cg_group, config)));
    auto cccl_group = cudax::coop::make_cg_equivalent_group(cg_group, config);

    static_assert(
      cuda::std::is_same_v<decltype(cudax::coop::generic_group{
                             cuda::gpu_thread, this_warp, cudax::coop::group_by<n>{}, cudax::coop::lane_synchronizer{}}),
                           decltype(cccl_group)>);
    REQUIRE(cuda::gpu_thread.count(cccl_group) == cg_group.size());
    REQUIRE(cuda::gpu_thread.rank(cccl_group) == cg_group.thread_rank());
    cccl_group.sync();
  });

  // Test cg::tiled_partition<N>(cg::this_thread_block()) with N == 32.
  {
    auto cg_group = cg::tiled_partition<32>(cg::this_thread_block());

    static_assert(noexcept(cudax::coop::make_cg_equivalent_group(cg_group, config)));
    auto cccl_group = cudax::coop::make_cg_equivalent_group(cg_group, config);

    static_assert(cuda::std::is_same_v<cudax::coop::this_warp<Hierarchy>, decltype(cccl_group)>);
    REQUIRE(cuda::gpu_thread.count(cccl_group) == cg_group.size());
    REQUIRE(cuda::gpu_thread.rank(cccl_group) == cg_group.thread_rank());
    cccl_group.sync();
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    test_cg_interop(config);
  }
};

C2H_TEST("CG interop", "[cg_interop]")
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
