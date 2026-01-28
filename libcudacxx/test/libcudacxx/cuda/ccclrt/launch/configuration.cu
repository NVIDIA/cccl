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
#define _CCCLRT_LAUNCH_CONFIG_TEST
#include <cuda/launch>

#include <host_device.cuh>

static CUlaunchConfig expectedConfig;
static bool replacementCalled = false;

void test_launch_kernel_replacement(CUlaunchConfig& config, CUfunction kernel, void* args[])
{
  replacementCalled = true;
  bool has_cluster  = false;

  CCCLRT_CHECK(expectedConfig.gridDimX == config.gridDimX);
  CCCLRT_CHECK(expectedConfig.gridDimY == config.gridDimY);
  CCCLRT_CHECK(expectedConfig.gridDimZ == config.gridDimZ);
  CCCLRT_CHECK(expectedConfig.blockDimX == config.blockDimX);
  CCCLRT_CHECK(expectedConfig.blockDimY == config.blockDimY);
  CCCLRT_CHECK(expectedConfig.blockDimZ == config.blockDimZ);
  CCCLRT_CHECK(expectedConfig.sharedMemBytes == config.sharedMemBytes);
  CCCLRT_CHECK(expectedConfig.hStream == config.hStream);
  CCCLRT_CHECK(expectedConfig.numAttrs == config.numAttrs);

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
            CCCLRT_CHECK(expectedAttr.value.clusterDim.x == actualAttr.value.clusterDim.x);
            CCCLRT_CHECK(expectedAttr.value.clusterDim.y == actualAttr.value.clusterDim.y);
            CCCLRT_CHECK(expectedAttr.value.clusterDim.z == actualAttr.value.clusterDim.z);
            has_cluster = true;
            break;
          case CU_LAUNCH_ATTRIBUTE_COOPERATIVE:
            CCCLRT_CHECK(expectedAttr.value.cooperative == actualAttr.value.cooperative);
            break;
          case CU_LAUNCH_ATTRIBUTE_PRIORITY:
            CCCLRT_CHECK(expectedAttr.value.priority == actualAttr.value.priority);
            break;
          default:
            CCCLRT_CHECK(false);
            break;
        }
        break;
      }
    }
    INFO("Searched attribute is " << expectedAttr.id);
    CCCLRT_CHECK(j != expectedConfig.numAttrs);
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
    auto config                               = cuda::make_config(dims, cuda::cooperative_launch());
    expectedConfig.numAttrs                   = 1 + HasCluster;
    expectedConfig.attrs                      = &attrs[0];
    expectedConfig.attrs[0].id                = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    expectedConfig.attrs[0].value.cooperative = 1;
    if constexpr (HasCluster)
    {
      add_cluster(cluster_dims, expectedConfig.attrs[1]);
    }
    cuda::launch(stream, config, empty_kernel, 0);
  }

  SECTION("Priority and dynamic smem")
  {
    CUlaunchAttribute attrs[2];
    constexpr int priority = 42;
    constexpr int num_ints = 128;
    auto config =
      cuda::make_config(dims, cuda::launch_priority(priority), cuda::dynamic_shared_memory<int[num_ints]>());
    expectedConfig.sharedMemBytes          = num_ints * sizeof(int);
    expectedConfig.numAttrs                = 1 + HasCluster;
    expectedConfig.attrs                   = &attrs[0];
    expectedConfig.attrs[0].id             = CU_LAUNCH_ATTRIBUTE_PRIORITY;
    expectedConfig.attrs[0].value.priority = priority;
    if constexpr (HasCluster)
    {
      add_cluster(cluster_dims, expectedConfig.attrs[1]);
    }
    cuda::launch(stream, config, empty_kernel, 0);
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
    auto config                   = cuda::make_config(dims, cuda::dynamic_shared_memory<S>(cuda::non_portable));
    expectedConfig.sharedMemBytes = sizeof(S);
    expectedConfig.numAttrs       = HasCluster;
    expectedConfig.attrs          = &attrs[0];
    if constexpr (HasCluster)
    {
      add_cluster(cluster_dims, expectedConfig.attrs[0]);
    }
    cuda::launch(stream, config, empty_kernel, 0);
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
  CCCLRT_CHECK(replacementCalled);
}

C2H_TEST("Hierarchy construction in config", "[launch]")
{
  auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::cooperative_launch());
  static_assert(cuda::block.count(cuda::grid, config) == 2);

  auto config_larger = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims(256), cuda::cooperative_launch());
  CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, config_larger) == 512);

  auto config_no_options = cuda::make_config(cuda::grid_dims(2), cuda::block_dims<128>());
  CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, config_no_options) == 256);

  [[maybe_unused]] auto config_no_dims = cuda::make_config(cuda::cooperative_launch());
  static_assert(
    cuda::std::is_same_v<::cuda::std::remove_cvref_t<decltype(config_no_dims.hierarchy())>, cuda::__empty_hierarchy>);
}

C2H_TEST("Configuration combine", "[launch]")
{
  auto grid    = cuda::grid_dims<2>;
  auto cluster = cuda::cluster_dims<2, 2>;
  auto block   = cuda::block_dims(256);
  SECTION("Combine with no overlap")
  {
    auto config_part1                         = cuda::make_config(grid);
    auto config_part2                         = cuda::make_config(block, cuda::launch_priority(2));
    auto combined                             = config_part1.combine(config_part2);
    [[maybe_unused]] auto combined_other_way  = config_part2.combine(config_part1);
    [[maybe_unused]] auto combined_with_empty = combined.combine(cuda::make_config());
    [[maybe_unused]] auto empty_with_combined = cuda::make_config().combine(combined);
    static_assert(
      cuda::std::is_same_v<decltype(combined), decltype(cuda::make_config(grid, block, cuda::launch_priority(2)))>);
    static_assert(cuda::std::is_same_v<decltype(combined), decltype(combined_other_way)>);
    static_assert(cuda::std::is_same_v<decltype(combined), decltype(combined_with_empty)>);
    static_assert(cuda::std::is_same_v<decltype(combined), decltype(empty_with_combined)>);
    CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, combined) == 512);
  }
  SECTION("Combine with overlap")
  {
    auto config_part1 = make_config(grid, cluster, cuda::launch_priority(2));
    auto config_part2 = make_config(cuda::cluster_dims<256>(), block, cuda::launch_priority(42));
    auto combined     = config_part1.combine(config_part2);
    CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, combined) == 2048);
    CCCLRT_REQUIRE(cuda::std::get<0>(combined.options()).priority == 2);

    auto replaced_one_option = cuda::make_config(cuda::launch_priority(3)).combine(combined);
    CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, replaced_one_option) == 2048);
    CCCLRT_REQUIRE(cuda::std::get<0>(replaced_one_option.options()).priority == 3);

    [[maybe_unused]] auto combined_with_extra_option = combined.combine(cuda::make_config(cuda::cooperative_launch()));
    static_assert(
      cuda::std::is_same_v<decltype(combined.hierarchy()), decltype(combined_with_extra_option.hierarchy())>);
    static_assert(
      cuda::std::tuple_size_v<::cuda::std::remove_cvref_t<decltype(combined_with_extra_option.options())>> == 2);
  }
}

#if !_CCCL_CUDA_COMPILER(CLANG)
template <typename Config>
__host__ __device__ void test_queries_on_config(const Config& config)
{
  CCCLRT_REQUIRE(cuda::gpu_thread.dims(cuda::grid, config) == dim3(1024));
  {
    auto dims = cuda::gpu_thread.dims_as<int>(cuda::grid, config);
    CCCLRT_REQUIRE(dims.x == 1024);
    CCCLRT_REQUIRE(dims.y == 1);
    CCCLRT_REQUIRE(dims.z == 1);
  }
  CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::block, config) == 256);
  CCCLRT_REQUIRE(cuda::gpu_thread.count_as<int>(cuda::block, config) == 256);
  CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, config) == 1024);
  CCCLRT_REQUIRE(cuda::gpu_thread.count_as<int>(cuda::grid, config) == 1024);
  CCCLRT_REQUIRE(cuda::block.extents(cuda::grid, config).extent(0) == 4);
  CCCLRT_REQUIRE(cuda::block.extents_as<int>(cuda::grid, config).extent(0) == 4);
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (CCCLRT_REQUIRE(cuda::block.rank(cuda::grid, config) == blockIdx.x);
     CCCLRT_REQUIRE(cuda::block.rank_as<int>(cuda::grid, config) == blockIdx.x);
     CCCLRT_REQUIRE(cuda::gpu_thread.index(cuda::block, config) == threadIdx);
     {
       auto idx = cuda::gpu_thread.index_as<int>(cuda::block, config);
       CCCLRT_REQUIRE(idx.x == static_cast<int>(threadIdx.x));
       CCCLRT_REQUIRE(idx.y == static_cast<int>(threadIdx.y));
       CCCLRT_REQUIRE(idx.z == static_cast<int>(threadIdx.z));
     }));
}

template <typename Config>
__global__ void test_kernel(Config config)
{
  test_queries_on_config(config);
}

C2H_TEST("Queries on config", "[launch]")
{
  auto config = cuda::make_config(cuda::grid_dims(4), cuda::block_dims<256>(), cuda::cooperative_launch());
  test_queries_on_config(config);
  test_kernel<<<4, 256>>>(config);
  CUDART(cudaGetLastError());
  CUDART(cudaDeviceSynchronize());
}
#endif // !_CCCL_CUDA_COMPILER(CLANG)
