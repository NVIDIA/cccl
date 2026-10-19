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
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/coop/group>

#include "group_testing.cuh"

template <template <class...> class GroupTempl, class... Args>
__device__ void test_make_group(Args&&... args)
{
  auto group = cudax::coop::make_group<GroupTempl>(cuda::std::forward<Args>(args)...);

  static_assert(cuda::std::is_same_v<decltype(GroupTempl{cuda::std::forward<Args>(args)...}), decltype(group)>);
  static_assert(noexcept(GroupTempl{cuda::std::forward<Args>(args)...})
                == noexcept(cudax::coop::make_group<GroupTempl>(cuda::std::forward<Args>(args)...)));

  group.sync_aligned();
}

template <template <class...> class GroupTempl, class... Args>
__device__ void test_make_group_non_exhaustive(bool is_valid, Args&&... args)
{
  auto group = cudax::coop::make_group<GroupTempl>(cudax::coop::non_exhaustive, cuda::std::forward<Args>(args)...);

  static_assert(
    cuda::std::is_same_v<cuda::std::optional<decltype(GroupTempl{cuda::std::forward<Args>(args)...})>, decltype(group)>);
  static_assert(
    noexcept(GroupTempl{cuda::std::forward<Args>(args)...})
    == noexcept(cudax::coop::make_group<GroupTempl>(cudax::coop::non_exhaustive, cuda::std::forward<Args>(args)...)));

  REQUIRE(group.has_value() == is_valid);
  if (group)
  {
    group->sync_aligned();
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    // Test this group construction.
    test_make_group<cudax::coop::this_thread>(config);
    test_make_group<cudax::coop::this_warp>(config);
    test_make_group<cudax::coop::this_block>(config);
    test_make_group<cudax::coop::this_cluster>(config);
    test_make_group<cudax::coop::this_grid>(config);

    // Test coalesced group.
    test_make_group<cudax::coop::coalesced_group>(config);

    // Test generic group.
    test_make_group<cudax::coop::generic_group>(
      cuda::gpu_thread,
      cudax::coop::this_warp{config},
      cudax::coop::group_by{cuda::std::cw<4>},
      cudax::coop::lane_synchronizer{});

    // Test virtual group.
    test_make_group<cudax::coop::virtual_group>(
      cuda::gpu_thread, cudax::coop::this_warp{config}, cudax::coop::group_by{cuda::std::cw<4>});

    // Test group view.
    test_make_group<cudax::coop::group_view>(cuda::gpu_thread, cudax::coop::this_warp{config});

    // Test non-exhaustive generic group.
    test_make_group_non_exhaustive<cudax::coop::generic_group>(
      cuda::gpu_thread.rank(cuda::warp) < 28,
      cuda::gpu_thread,
      cudax::coop::this_warp{config},
      cudax::coop::group_by{cudax::coop::non_exhaustive, 7},
      cudax::coop::lane_synchronizer{});
  }
};

C2H_TEST("Make group", "[group]")
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
