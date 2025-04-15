//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Test translation of launch function arguments to cudaLaunchConfig_t sent to cudaLaunchKernelEx internally
// We replace cudaLaunchKernelEx with a test function here through a macro to intercept the cudaLaunchConfig_t
#define cudaLaunchKernelEx cudaLaunchKernelExTestReplacement
#include <cuda/experimental/launch.cuh>
#undef cudaLaunchKernelEx

#include <host_device.cuh>

static cudaLaunchConfig_t expectedConfig;
static bool replacementCalled = false;

template <typename... ExpTypes, typename... ActTypes>
cudaError_t
cudaLaunchKernelExTestReplacement(const cudaLaunchConfig_t* config, void (*kernel)(ExpTypes...), ActTypes&&... args)
{
  replacementCalled = true;
  bool has_cluster  = false;

  CUDAX_CHECK(expectedConfig.numAttrs == config->numAttrs);
  CUDAX_CHECK(expectedConfig.blockDim == config->blockDim);
  CUDAX_CHECK(expectedConfig.gridDim == config->gridDim);
  CUDAX_CHECK(expectedConfig.stream == config->stream);
  CUDAX_CHECK(expectedConfig.dynamicSmemBytes == config->dynamicSmemBytes);

  for (unsigned int i = 0; i < expectedConfig.numAttrs; ++i)
  {
    auto& expectedAttr = expectedConfig.attrs[i];
    unsigned int j;
    for (j = 0; j < expectedConfig.numAttrs; ++j)
    {
      auto& actualAttr = config->attrs[j];
      if (expectedAttr.id == actualAttr.id)
      {
        switch (expectedAttr.id)
        {
          case cudaLaunchAttributeClusterDimension:
            CUDAX_CHECK(expectedAttr.val.clusterDim.x == actualAttr.val.clusterDim.x);
            CUDAX_CHECK(expectedAttr.val.clusterDim.y == actualAttr.val.clusterDim.y);
            CUDAX_CHECK(expectedAttr.val.clusterDim.z == actualAttr.val.clusterDim.z);
            has_cluster = true;
            break;
          case cudaLaunchAttributeCooperative:
            CUDAX_CHECK(expectedAttr.val.cooperative == actualAttr.val.cooperative);
            break;
          case cudaLaunchAttributePriority:
            CUDAX_CHECK(expectedAttr.val.priority == actualAttr.val.priority);
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
    return cudaLaunchKernelEx(config, kernel, cuda::std::forward<ActTypes>(args)...);
  }
  else
  {
    return cudaSuccess;
  }
}

__global__ void empty_kernel(int i) {}

template <bool HasCluster>
auto make_test_dims(const dim3& grid_dims, const dim3& block_dims, const dim3& cluster_dims = dim3())
{
  if constexpr (HasCluster)
  {
    return cudax::make_hierarchy(
      cudax::grid_dims(grid_dims), cudax::cluster_dims(cluster_dims), cudax::block_dims(block_dims));
  }
  else
  {
    return cudax::make_hierarchy(cudax::grid_dims(grid_dims), cudax::block_dims(block_dims));
  }
}

auto add_cluster(const dim3& cluster_dims, cudaLaunchAttribute& attr)
{
  attr.id             = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {cluster_dims.x, cluster_dims.y, cluster_dims.z};
}

template <bool HasCluster, typename... Dims>
auto configuration_test(
  ::cuda::stream_ref stream, const dim3& grid_dims, const dim3& block_dims, const dim3& cluster_dims = dim3())
{
  auto dims             = make_test_dims<HasCluster>(grid_dims, block_dims, cluster_dims);
  expectedConfig        = {};
  expectedConfig.stream = stream.get();
  if constexpr (HasCluster)
  {
    expectedConfig.gridDim =
      dim3(grid_dims.x * cluster_dims.x, grid_dims.y * cluster_dims.y, grid_dims.z * cluster_dims.z);
  }
  else
  {
    expectedConfig.gridDim = grid_dims;
  }
  expectedConfig.blockDim = block_dims;

  SECTION("Simple cooperative launch")
  {
    cudaLaunchAttribute attrs[2];
    auto config                             = cudax::make_config(dims, cudax::cooperative_launch());
    expectedConfig.numAttrs                 = 1 + HasCluster;
    expectedConfig.attrs                    = &attrs[0];
    expectedConfig.attrs[0].id              = cudaLaunchAttributeCooperative;
    expectedConfig.attrs[0].val.cooperative = 1;
    if constexpr (HasCluster)
    {
      add_cluster(cluster_dims, expectedConfig.attrs[1]);
    }
    cudax::launch(stream, config, empty_kernel, 0);
  }

  SECTION("Priority and dynamic smem")
  {
    cudaLaunchAttribute attrs[2];
    const int priority = 42;
    const int num_ints = 128;
    auto config =
      cudax::make_config(dims, cudax::launch_priority(priority), cudax::dynamic_shared_memory<int>(num_ints));
    expectedConfig.dynamicSmemBytes      = num_ints * sizeof(int);
    expectedConfig.numAttrs              = 1 + HasCluster;
    expectedConfig.attrs                 = &attrs[0];
    expectedConfig.attrs[0].id           = cudaLaunchAttributePriority;
    expectedConfig.attrs[0].val.priority = priority;
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
    cudaLaunchAttribute attrs[1];
    auto config                     = cudax::make_config(dims, cudax::dynamic_shared_memory<S, 1, true>());
    expectedConfig.dynamicSmemBytes = sizeof(S);
    expectedConfig.numAttrs         = HasCluster;
    expectedConfig.attrs            = &attrs[0];
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
  auto config = cudax::make_config(cudax::grid_dims<2>(), cudax::cooperative_launch());
  static_assert(config.dims.count(cudax::block) == 2);

  auto config_larger = cudax::make_config(cudax::grid_dims<2>(), cudax::block_dims(256), cudax::cooperative_launch());
  CUDAX_REQUIRE(config_larger.dims.count(cudax::thread) == 512);

  auto config_no_options = cudax::make_config(cudax::grid_dims(2), cudax::block_dims<128>());
  CUDAX_REQUIRE(config_no_options.dims.count(cudax::thread) == 256);

  [[maybe_unused]] auto config_no_dims = cudax::make_config(cudax::cooperative_launch());
  static_assert(cuda::std::is_same_v<decltype(config_no_dims.dims), cudax::__empty_hierarchy>);
}

C2H_TEST("Configuration combine", "[launch]")
{
  auto grid    = cudax::grid_dims<2>;
  auto cluster = cudax::cluster_dims<2, 2>;
  auto block   = cudax::block_dims(256);
  SECTION("Combine with no overlap")
  {
    auto config_part1                         = make_config(grid);
    auto config_part2                         = make_config(block, cudax::launch_priority(2));
    auto combined                             = config_part1.combine(config_part2);
    [[maybe_unused]] auto combined_other_way  = config_part2.combine(config_part1);
    [[maybe_unused]] auto combined_with_empty = combined.combine(cudax::make_config());
    [[maybe_unused]] auto empty_with_combined = cudax::make_config().combine(combined);
    static_assert(
      cuda::std::is_same_v<decltype(combined), decltype(make_config(grid, block, cudax::launch_priority(2)))>);
    static_assert(cuda::std::is_same_v<decltype(combined), decltype(combined_other_way)>);
    static_assert(cuda::std::is_same_v<decltype(combined), decltype(combined_with_empty)>);
    static_assert(cuda::std::is_same_v<decltype(combined), decltype(empty_with_combined)>);
    CUDAX_REQUIRE(combined.dims.count(cudax::thread) == 512);
  }
  SECTION("Combine with overlap")
  {
    auto config_part1 = make_config(grid, cluster, cudax::launch_priority(2));
    auto config_part2 = make_config(cudax::cluster_dims<256>, block, cudax::launch_priority(42));
    auto combined     = config_part1.combine(config_part2);
    CUDAX_REQUIRE(combined.dims.count(cudax::thread) == 2048);
    CUDAX_REQUIRE(cuda::std::get<0>(combined.options).priority == 2);

    auto replaced_one_option = cudax::make_config(cudax::launch_priority(3)).combine(combined);
    CUDAX_REQUIRE(replaced_one_option.dims.count(cudax::thread) == 2048);
    CUDAX_REQUIRE(cuda::std::get<0>(replaced_one_option.options).priority == 3);

    [[maybe_unused]] auto combined_with_extra_option =
      combined.combine(cudax::make_config(cudax::cooperative_launch()));
    static_assert(cuda::std::is_same_v<decltype(combined.dims), decltype(combined_with_extra_option.dims)>);
    static_assert(cuda::std::tuple_size_v<decltype(combined_with_extra_option.options)> == 2);
  }
}
