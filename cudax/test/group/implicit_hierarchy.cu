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
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

template <bool V>
struct Is1D_t : cuda::std::bool_constant<V>
{};

struct TestKernel
{
  template <bool Is1D>
  __device__ void operator()(Is1D_t<Is1D>)
  {
    // Test implicit_hierarchy()
    {
      using Extents3D = cuda::std::dims<3, unsigned>;

      static_assert(
        cuda::std::is_same_v<decltype(cudax::implicit_hierarchy()),
                             cuda::hierarchy<cuda::thread_level,
                                             cuda::hierarchy_level_desc<cuda::grid_level, Extents3D>,
                                             cuda::hierarchy_level_desc<cuda::cluster_level, Extents3D>,
                                             cuda::hierarchy_level_desc<cuda::block_level, Extents3D>>>);
      static_assert(noexcept(cudax::implicit_hierarchy()));

      const auto hier = cudax::implicit_hierarchy();
      REQUIRE(cuda::gpu_thread.extents(cuda::block) == cuda::gpu_thread.extents(cuda::block, hier));
      REQUIRE(cuda::block.extents(cuda::cluster) == cuda::block.extents(cuda::cluster, hier));
      REQUIRE(cuda::cluster.extents(cuda::grid) == cuda::cluster.extents(cuda::grid, hier));
    }

    // Test implicit_hierarchy_1d()
    if constexpr (Is1D)
    {
      using Extents1D = cuda::std::extents<unsigned, cuda::std::dynamic_extent, 1, 1>;

      static_assert(
        cuda::std::is_same_v<decltype(cudax::implicit_hierarchy_1d()),
                             cuda::hierarchy<cuda::thread_level,
                                             cuda::hierarchy_level_desc<cuda::grid_level, Extents1D>,
                                             cuda::hierarchy_level_desc<cuda::cluster_level, Extents1D>,
                                             cuda::hierarchy_level_desc<cuda::block_level, Extents1D>>>);
      static_assert(noexcept(cudax::implicit_hierarchy_1d()));

      const auto hier = cudax::implicit_hierarchy_1d();
      REQUIRE(cuda::gpu_thread.extents(cuda::block) == cuda::gpu_thread.extents(cuda::block, hier));
      REQUIRE(cuda::block.extents(cuda::cluster) == cuda::block.extents(cuda::cluster, hier));
      REQUIRE(cuda::cluster.extents(cuda::grid) == cuda::cluster.extents(cuda::grid, hier));
    }
  }
};

C2H_TEST("Implicit Hierarchy", "[group][implicit_hierarchy]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  {
    const auto config =
      cuda::make_config(cuda::grid_dims<2, 3>(), cuda::block_dims<4, 6>(), cuda::cooperative_launch{});
    cuda::launch(stream, config, TestKernel{}, Is1D_t<false>{});
  }
  {
    const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
    cuda::launch(stream, config, TestKernel{}, Is1D_t<true>{});
  }

  if (cuda::device_attributes::compute_capability(device) >= cuda::compute_capability{90})
  {
    {
      const auto config_cluster = cuda::make_config(
        cuda::grid_dims<2, 2>(), cuda::cluster_dims<2, 2>(), cuda::block_dims<2, 2>(), cuda::cooperative_launch{});
      cuda::launch(stream, config_cluster, TestKernel{}, Is1D_t<false>{});
    }
    {
      const auto config_cluster = cuda::make_config(
        cuda::grid_dims<2>(), cuda::cluster_dims<2>(), cuda::block_dims<2>(), cuda::cooperative_launch{});
      cuda::launch(stream, config_cluster, TestKernel{}, Is1D_t<true>{});
    }
  }

  stream.sync();
}
