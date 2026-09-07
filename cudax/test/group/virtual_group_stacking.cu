//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/barrier>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

template <class Group>
__device__ void test_group_membership(const Group& group, cuda::std::uint32_t count, cuda::std::uint32_t rank)
{
  static_assert(cudax::is_group<Group>);
  static_assert(cuda::std::is_same_v<cuda::thread_level, typename Group::unit_type>);

  REQUIRE(cuda::gpu_thread.is_part_of(group));
  REQUIRE(cuda::gpu_thread.count(group) == count);
  REQUIRE(cuda::gpu_thread.rank(group) == rank);
  REQUIRE(group.__mapping_result().unit_count() == count);
  REQUIRE(group.__mapping_result().unit_rank() == rank);
}

struct TestKernel1
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    auto parent_group = cudax::make_this_group(cuda::warp, config);
    const auto rank   = cuda::gpu_thread.rank_as<cuda::std::uint32_t>(parent_group);

    cudax::virtual_group group1{cuda::gpu_thread, parent_group, cudax::group_by<16>{}};
    test_group_membership(group1, 16, rank % 16);
    REQUIRE(group1.count(parent_group) == 2);
    REQUIRE(group1.rank(parent_group) == rank / 16);

    cudax::virtual_group group2{
      cuda::gpu_thread, group1, cudax::group_as{cuda::std::integer_sequence<cuda::std::size_t, 8, 8>{}}};
    test_group_membership(group2, 8, rank % 8);
    REQUIRE(group2.count(group1) == 2);
    REQUIRE(group2.rank(group1) == (rank % 16) / 8);

    cudax::virtual_group group3{cuda::gpu_thread, group2, cudax::group_by{4}};
    test_group_membership(group3, 4, rank % 4);
    REQUIRE(group3.count(group2) == 2);
    REQUIRE(group3.rank(group2) == (rank % 8) / 4);

    group1.sync();
    group2.sync_aligned();
    group3.sync();
  }
};

struct TestKernel2
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    auto parent_group = cudax::make_this_group(cuda::block, config);
    const auto rank   = cuda::gpu_thread.rank_as<cuda::std::uint32_t>(parent_group);

    cudax::virtual_group group1{cuda::gpu_thread, parent_group, cudax::group_by<32>{}};
    test_group_membership(group1, 32, rank % 32);
    REQUIRE(group1.count(parent_group) == 4);
    REQUIRE(group1.rank(parent_group) == rank / 32);

    cudax::virtual_group group2{
      cuda::gpu_thread, group1, cudax::group_as{cuda::std::integer_sequence<cuda::std::size_t, 16, 16>{}}};
    test_group_membership(group2, 16, rank % 16);
    REQUIRE(group2.count(group1) == 2);
    REQUIRE(group2.rank(group1) == (rank % 32) / 16);

    cudax::virtual_group group3{cuda::gpu_thread, group2, cudax::group_by<8>{}};
    test_group_membership(group3, 8, rank % 8);
    REQUIRE(group3.count(group2) == 2);
    REQUIRE(group3.rank(group2) == (rank % 16) / 8);

    cudax::virtual_group group4{cuda::gpu_thread, group3, cudax::identity_mapping{}};
    test_group_membership(group4, 8, rank % 8);
    REQUIRE(group4.count(group3) == 1);
    REQUIRE(group4.rank(group3) == 0);

    group1.sync();
    group2.sync();
    group3.sync_aligned();
    group4.sync();
  }
};

struct TestKernel3
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    auto parent_group = cudax::make_this_group(cuda::block, config);
    const auto rank   = cuda::warp.rank_as<cuda::std::uint32_t>(parent_group);

    cudax::virtual_group group1{cuda::warp, parent_group, cudax::group_by<16>{}};
    REQUIRE(cuda::warp.is_part_of(group1));
    REQUIRE(cuda::warp.count(group1) == 16);
    REQUIRE(cuda::warp.rank(group1) == rank % 16);
    REQUIRE(group1.count(parent_group) == 2);
    REQUIRE(group1.rank(parent_group) == rank / 16);

    cudax::virtual_group group2{cuda::warp, group1, cudax::group_by<8>{}};
    REQUIRE(cuda::warp.is_part_of(group2));
    REQUIRE(cuda::warp.count(group2) == 8);
    REQUIRE(cuda::warp.rank(group2) == rank % 8);
    REQUIRE(group2.count(group1) == 2);
    REQUIRE(group2.rank(group1) == (rank / 8) % 2);

    group1.sync();
    group2.sync();
    group1.sync_aligned();
    group2.sync_aligned();
  }
};

C2H_TEST("Virtual Group Stacking", "[virtual_group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<32>());
    cuda::launch(stream, config, TestKernel1{});
  }
  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<128>());
    cuda::launch(stream, config, TestKernel2{});
  }
  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<1024>());
    cuda::launch(stream, config, TestKernel3{});
  }

  stream.sync();
}
