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
#include <cuda/std/bit>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

namespace
{
template <class Level, class Config>
__device__ void test_lane_synchronizer(const Level& level, Config config)
{
  using Synchronizer = cudax::lane_synchronizer;
  static_assert(cuda::std::is_empty_v<Synchronizer>);

  // Test default constructor.
  static_assert(cuda::std::is_trivially_default_constructible_v<Synchronizer>);

  // Test make_instance(...).
  {
    const auto parent_group = cudax::make_this_group(level, config);
    const ThreadsInWarpMappingResult prev_mapping_result;

    const cudax::group_by mapping{2};
    const Synchronizer synchronizer{};

    const auto mapping_result = mapping.map(parent_group, prev_mapping_result);
    const auto synchronizer_instance =
      synchronizer.make_instance(cuda::gpu_thread, parent_group, mapping, mapping_result);

    const auto this_lane_mask = cuda::ptx::get_sreg_lanemask_eq();
    const auto another_lane_mask =
      cuda::std::rotl(this_lane_mask, (cuda::std::countr_zero(this_lane_mask) % 2 == 0) ? 1 : -1);
    CUDAX_CHECK(synchronizer_instance.__lane_mask_ == (this_lane_mask | another_lane_mask));

    // Test do_sync(...).
    static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync(mapping_result, synchronizer))>);
    static_assert(noexcept(synchronizer_instance.do_sync(mapping_result, synchronizer)));
    synchronizer_instance.do_sync(mapping_result, synchronizer);

    // Test do_sync_aligned(...).
    static_assert(
      cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync_aligned(mapping_result, synchronizer))>);
    static_assert(noexcept(synchronizer_instance.do_sync_aligned(mapping_result, synchronizer)));
    synchronizer_instance.do_sync_aligned(mapping_result, synchronizer);
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_lane_synchronizer(cuda::warp, config);
    test_lane_synchronizer(cuda::block, config);
    test_lane_synchronizer(cuda::cluster, config);
    test_lane_synchronizer(cuda::grid, config);
  }
};
} // namespace

C2H_TEST("Lane synchronizer", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<8, 4>());
    cuda::launch(stream, config, TestKernel{});
  }
  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims(dim3{8, 4}));
    cuda::launch(stream, config, TestKernel{});
  }

  stream.sync();
}
