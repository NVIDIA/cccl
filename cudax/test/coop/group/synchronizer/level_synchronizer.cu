//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/utility>
#include <cuda/stream>
#include <cuda/type_traits>

#include <cuda/experimental/coop/group>

#include "group_testing.cuh"

namespace
{
template <class Level, class Config>
__device__ void test_level_synchronizer(const Level& level, Config config)
{
  const auto& hierarchy = config.hierarchy();

  using Synchronizer = cudax::coop::level_synchronizer;
  static_assert(cuda::std::is_empty_v<Synchronizer>);

  // Test default constructor.
  static_assert(cuda::std::is_trivially_default_constructible_v<Synchronizer>);

  // Test __synchronizer_instance<Level>
  {
    using MappingResult        = cudax::coop::__this_mapping_result<Level>;
    using SynchronizerInstance = typename Synchronizer::template __synchronizer_instance<Level>;

    const MappingResult mapping_result{};

    // Test default constructor.
    static_assert(cuda::std::is_default_constructible_v<SynchronizerInstance>);
    SynchronizerInstance synchronizer_instance{};

    static_assert(cuda::is_trivially_copyable_v<SynchronizerInstance>);
    static_assert(cuda::std::is_trivially_destructible_v<SynchronizerInstance>);

    using ArrivalToken = typename SynchronizerInstance::arrival_token;

    // Test arrival_token type.
    static_assert(cuda::std::is_move_constructible_v<ArrivalToken>);
    static_assert(cuda::std::is_move_assignable_v<ArrivalToken>);
    static_assert(cuda::std::is_destructible_v<ArrivalToken>);

    // Test do_sync(...).
    // NOLINTNEXTLINE(cccl-simplify-type-traits)
    static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync(hierarchy))>);
    synchronizer_instance.do_sync(hierarchy);

    // Test do_arrive(...).
    static_assert(cuda::std::is_same_v<ArrivalToken, decltype(synchronizer_instance.do_arrive(hierarchy))>);
    auto arrival_token = synchronizer_instance.do_arrive(hierarchy);

    // Test do_wait(...).
    static_assert(
      // NOLINTNEXTLINE(cccl-simplify-type-traits)
      cuda::std::is_same_v<void, decltype(synchronizer_instance.do_wait(hierarchy, cuda::std::move(arrival_token)))>);
    synchronizer_instance.do_wait(hierarchy, cuda::std::move(arrival_token));

    // Test do_sync_aligned(...).
    // NOLINTNEXTLINE(cccl-simplify-type-traits)
    static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync_aligned(hierarchy))>);
    synchronizer_instance.do_sync_aligned(hierarchy);

    // Test do_arrive_aligned(...).
    static_assert(cuda::std::is_same_v<ArrivalToken, decltype(synchronizer_instance.do_arrive_aligned(hierarchy))>);
    auto arrival_token_a = synchronizer_instance.do_arrive_aligned(hierarchy);

    // Test do_wait_aligned(...).
    static_assert(
      cuda::std:: // NOLINT(cccl-simplify-type-traits)
      is_same_v<void, decltype(synchronizer_instance.do_wait_aligned(hierarchy, cuda::std::move(arrival_token_a)))>);
    synchronizer_instance.do_wait_aligned(hierarchy, cuda::std::move(arrival_token_a));

    // Test deinit(...);
    // NOLINTNEXTLINE(cccl-simplify-type-traits)
    static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.deinit(mapping_result, hierarchy))>);
    synchronizer_instance.deinit(mapping_result, hierarchy);
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_level_synchronizer(cuda::gpu_thread, config);
    test_level_synchronizer(cuda::warp, config);
    test_level_synchronizer(cuda::block, config);
    test_level_synchronizer(cuda::cluster, config);
    test_level_synchronizer(cuda::grid, config);
  }
};
} // namespace

C2H_TEST("Level synchronizer", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  {
    const auto config = cuda::make_config(cuda::grid_dims<8>(), cuda::block_dims<8, 16>(), cuda::cooperative_launch{});
    cuda::launch(stream, config, TestKernel{});
  }

  {
    const auto config =
      cuda::make_config(cuda::grid_dims<8>(), cuda::block_dims(dim3{8, 16}), cuda::cooperative_launch{});
    cuda::launch(stream, config, TestKernel{});
  }

  if (cuda::device_attributes::compute_capability_major(device) >= 9)
  {
    const auto config = cuda::make_config(
      cuda::grid_dims<8>(), cuda::cluster_dims<4>(), cuda::block_dims(dim3{8, 16}), cuda::cooperative_launch{});
    cuda::launch(stream, config, TestKernel{});
  }

  stream.sync();
}
