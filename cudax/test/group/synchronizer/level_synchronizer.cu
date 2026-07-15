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
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

namespace
{
template <class Level, class Config>
__device__ void test_level_synchronizer(const Level& level, Config config)
{
  const auto& hierarchy = config.hierarchy();

  using Synchronizer = cudax::level_synchronizer;
  static_assert(cuda::std::is_empty_v<Synchronizer>);

  // Test default constructor.
  static_assert(cuda::std::is_trivially_default_constructible_v<Synchronizer>);

  // Test __synchronizer_instance<Level>
  {
    using MappingResult        = cudax::__this_mapping_result<Level>;
    using SynchronizerInstance = typename Synchronizer::template __synchronizer_instance<Level>;

    MappingResult mapping_result{};
    Synchronizer synchronizer{};

    // Test default constructor.
    static_assert(cuda::std::is_nothrow_default_constructible_v<SynchronizerInstance>);
    SynchronizerInstance synchronizer_instance{};

    // Test do_sync(...).
    static_assert(
      cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync(mapping_result, synchronizer, hierarchy))>);
    static_assert(noexcept(synchronizer_instance.do_sync(mapping_result, synchronizer, hierarchy)));
    synchronizer_instance.do_sync(mapping_result, synchronizer, hierarchy);

    // Test do_sync_aligned(...).
    static_assert(
      cuda::std::is_same_v<void,
                           decltype(synchronizer_instance.do_sync_aligned(mapping_result, synchronizer, hierarchy))>);
    static_assert(noexcept(synchronizer_instance.do_sync_aligned(mapping_result, synchronizer, hierarchy)));
    synchronizer_instance.do_sync_aligned(mapping_result, synchronizer, hierarchy);
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
