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
  stream.wait();
}

TEST_CASE("Launch configuration", "[launch]")
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
