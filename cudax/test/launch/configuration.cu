//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda.h>

void test_launch_kernel_replacement(CUlaunchConfig& config, CUfunction kernel, void* args[]);

// This is a replacement for the launch kernel function that is used to test
// the configuration of the launch kernel. It checks if the configuration
// matches the expected configuration and calls the original launch kernel
// function if it does. If the configuration does not match, it will fail the
// test.
#define _CUDAX_LAUNCH_CONFIG_TEST
#include <cuda/experimental/launch.cuh>

#include <host_device.cuh>

static CUlaunchConfig expectedConfig;
static bool replacementCalled = false;

void test_launch_kernel_replacement(CUlaunchConfig& config, CUfunction kernel, void* args[])
{
  replacementCalled = true;
  bool has_cluster  = false;

  CUDAX_CHECK(expectedConfig.gridDimX == config.gridDimX);
  CUDAX_CHECK(expectedConfig.gridDimY == config.gridDimY);
  CUDAX_CHECK(expectedConfig.gridDimZ == config.gridDimZ);
  CUDAX_CHECK(expectedConfig.blockDimX == config.blockDimX);
  CUDAX_CHECK(expectedConfig.blockDimY == config.blockDimY);
  CUDAX_CHECK(expectedConfig.blockDimZ == config.blockDimZ);
  CUDAX_CHECK(expectedConfig.sharedMemBytes == config.sharedMemBytes);
  CUDAX_CHECK(expectedConfig.hStream == config.hStream);
  CUDAX_CHECK(expectedConfig.numAttrs == config.numAttrs);

  for (unsigned int i = 0; i < expectedConfig.numAttrs; ++i)
  {
    auto& expectedAttr = expectedConfig.attrs[i];
    unsigned int j;
    for (j = 0; j < expectedConfig.numAttrs; ++j)
    {
      auto& actualAttr = config.attrs[j];
      if (expectedAttr.id == actualAttr.id)
      {
        switch (expectedAttr.id)
        {
          case CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION:
            CUDAX_CHECK(expectedAttr.value.clusterDim.x == actualAttr.value.clusterDim.x);
            CUDAX_CHECK(expectedAttr.value.clusterDim.y == actualAttr.value.clusterDim.y);
            CUDAX_CHECK(expectedAttr.value.clusterDim.z == actualAttr.value.clusterDim.z);
            has_cluster = true;
            break;
          case CU_LAUNCH_ATTRIBUTE_COOPERATIVE:
            CUDAX_CHECK(expectedAttr.value.cooperative == actualAttr.value.cooperative);
            break;
          case CU_LAUNCH_ATTRIBUTE_PRIORITY:
            CUDAX_CHECK(expectedAttr.value.priority == actualAttr.value.priority);
            break;
          default:
            CUDAX_CHECK(false);
            break;
        }
        break;
      }
    }
    INFO("Searched attribute is " << expectedAttr.id);
    CUDAX_CHECK(j != expectedConfig.numAttrs);
  }

  if (!has_cluster || !skip_device_exec(arch_filter<std::less<int>, 90>))
  {
    return ::cuda::__driver::__launchKernel(config, kernel, args);
  }
}

__global__ void empty_kernel(int i) {}

template <bool HasCluster>
auto make_test_dims(const dim3& grid_dims, const dim3& block_dims, const dim3& cluster_dims = dim3())
{
  if constexpr (HasCluster)
  {
    return cuda::make_hierarchy(
      cuda::grid_dims(grid_dims), cuda::cluster_dims(cluster_dims), cuda::block_dims(block_dims));
  }
  else
  {
    return cuda::make_hierarchy(cuda::grid_dims(grid_dims), cuda::block_dims(block_dims));
  }
}

auto add_cluster(const dim3& cluster_dims, CUlaunchAttribute& attr)
{
  attr.id               = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attr.value.clusterDim = {cluster_dims.x, cluster_dims.y, cluster_dims.z};
}

template <bool HasCluster, typename... Dims>
auto configuration_test(
  ::cuda::stream_ref stream, const dim3& grid_dims, const dim3& block_dims, const dim3& cluster_dims = dim3())
{
  auto dims              = make_test_dims<HasCluster>(grid_dims, block_dims, cluster_dims);
  expectedConfig         = {};
  expectedConfig.hStream = stream.get();
  if constexpr (HasCluster)
  {
    expectedConfig.gridDimX = grid_dims.x * cluster_dims.x;
    expectedConfig.gridDimY = grid_dims.y * cluster_dims.y;
    expectedConfig.gridDimZ = grid_dims.z * cluster_dims.z;
  }
  else
  {
    expectedConfig.gridDimX = grid_dims.x;
    expectedConfig.gridDimY = grid_dims.y;
    expectedConfig.gridDimZ = grid_dims.z;
  }
  expectedConfig.blockDimX = block_dims.x;
  expectedConfig.blockDimY = block_dims.y;
  expectedConfig.blockDimZ = block_dims.z;

  SECTION("Simple cooperative launch")
  {
    CUlaunchAttribute attrs[2];
    auto config                               = cudax::make_config(dims, cudax::cooperative_launch());
    expectedConfig.numAttrs                   = 1 + HasCluster;
    expectedConfig.attrs                      = &attrs[0];
    expectedConfig.attrs[0].id                = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    expectedConfig.attrs[0].value.cooperative = 1;
    if constexpr (HasCluster)
    {
      add_cluster(cluster_dims, expectedConfig.attrs[1]);
    }
    cudax::launch(stream, config, empty_kernel, 0);
  }

  SECTION("Priority and dynamic smem")
  {
    CUlaunchAttribute attrs[2];
    constexpr int priority = 42;
    constexpr int num_ints = 128;
    auto config =
      cudax::make_config(dims, cudax::launch_priority(priority), cudax::dynamic_shared_memory<int[num_ints]>());
    expectedConfig.sharedMemBytes          = num_ints * sizeof(int);
    expectedConfig.numAttrs                = 1 + HasCluster;
    expectedConfig.attrs                   = &attrs[0];
    expectedConfig.attrs[0].id             = CU_LAUNCH_ATTRIBUTE_PRIORITY;
    expectedConfig.attrs[0].value.priority = priority;
    if constexpr (HasCluster)
    {
      add_cluster(cluster_dims, expectedConfig.attrs[1]);
    }
    cudax::launch(stream, config, empty_kernel, 0);
  }

  SECTION("Large dynamic smem")
  {
    // Exceed the default 48kB of shared to check if its properly handled
    // TODO move to launch option (available since CUDA 12.4)
    struct S
    {
      int arr[13 * 1024];
    };
    CUlaunchAttribute attrs[1];
    auto config                   = cudax::make_config(dims, cudax::dynamic_shared_memory<S>(cudax::non_portable));
    expectedConfig.sharedMemBytes = sizeof(S);
    expectedConfig.numAttrs       = HasCluster;
    expectedConfig.attrs          = &attrs[0];
    if constexpr (HasCluster)
    {
      add_cluster(cluster_dims, expectedConfig.attrs[0]);
    }
    cudax::launch(stream, config, empty_kernel, 0);
  }
  stream.sync();
}

C2H_TEST("Launch configuration", "[launch]")
{
  cudaStream_t stream;
  CUDART(cudaStreamCreate(&stream));
  SECTION("No cluster")
  {
    configuration_test<false>(stream, 8, 64);
  }
  SECTION("With cluster")
  {
    configuration_test<true>(stream, 8, 32, 2);
  }

  CUDART(cudaStreamDestroy(stream));
  CUDAX_CHECK(replacementCalled);
}

C2H_TEST("Hierarchy construction in config", "[launch]")
{
  auto config = cudax::make_config(cuda::grid_dims<2>(), cudax::cooperative_launch());
  static_assert(config.dims.count(cuda::block) == 2);

  auto config_larger = cudax::make_config(cuda::grid_dims<2>(), cuda::block_dims(256), cudax::cooperative_launch());
  CUDAX_REQUIRE(config_larger.dims.count(cuda::thread) == 512);

  auto config_no_options = cudax::make_config(cuda::grid_dims(2), cuda::block_dims<128>());
  CUDAX_REQUIRE(config_no_options.dims.count(cuda::thread) == 256);

  [[maybe_unused]] auto config_no_dims = cudax::make_config(cudax::cooperative_launch());
  static_assert(cuda::std::is_same_v<decltype(config_no_dims.dims), cuda::__empty_hierarchy>);
}

C2H_TEST("Configuration combine", "[launch]")
{
  auto grid    = cuda::grid_dims<2>;
  auto cluster = cuda::cluster_dims<2, 2>;
  auto block   = cuda::block_dims(256);
  SECTION("Combine with no overlap")
  {
    auto config_part1                         = cudax::make_config(grid);
    auto config_part2                         = cudax::make_config(block, cudax::launch_priority(2));
    auto combined                             = config_part1.combine(config_part2);
    [[maybe_unused]] auto combined_other_way  = config_part2.combine(config_part1);
    [[maybe_unused]] auto combined_with_empty = combined.combine(cudax::make_config());
    [[maybe_unused]] auto empty_with_combined = cudax::make_config().combine(combined);
    static_assert(
      cuda::std::is_same_v<decltype(combined), decltype(cudax::make_config(grid, block, cudax::launch_priority(2)))>);
    static_assert(cuda::std::is_same_v<decltype(combined), decltype(combined_other_way)>);
    static_assert(cuda::std::is_same_v<decltype(combined), decltype(combined_with_empty)>);
    static_assert(cuda::std::is_same_v<decltype(combined), decltype(empty_with_combined)>);
    CUDAX_REQUIRE(combined.dims.count(cuda::thread) == 512);
  }
  SECTION("Combine with overlap")
  {
    auto config_part1 = make_config(grid, cluster, cudax::launch_priority(2));
    auto config_part2 = make_config(cuda::cluster_dims<256>, block, cudax::launch_priority(42));
    auto combined     = config_part1.combine(config_part2);
    CUDAX_REQUIRE(combined.dims.count(cuda::thread) == 2048);
    CUDAX_REQUIRE(cuda::std::get<0>(combined.options).priority == 2);

    auto replaced_one_option = cudax::make_config(cudax::launch_priority(3)).combine(combined);
    CUDAX_REQUIRE(replaced_one_option.dims.count(cuda::thread) == 2048);
    CUDAX_REQUIRE(cuda::std::get<0>(replaced_one_option.options).priority == 3);

    [[maybe_unused]] auto combined_with_extra_option =
      combined.combine(cudax::make_config(cudax::cooperative_launch()));
    static_assert(cuda::std::is_same_v<decltype(combined.dims), decltype(combined_with_extra_option.dims)>);
    static_assert(cuda::std::tuple_size_v<decltype(combined_with_extra_option.options)> == 2);
  }
}
