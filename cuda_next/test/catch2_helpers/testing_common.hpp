//===----------------------------------------------------------------------===//
//
// Part of CUDA Next in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __TESTING_COMMON_H__
#define __TESTING_COMMON_H__

#include <exception>
#include <iostream>

#include <catch2/catch.hpp>
#include <cuda/next/hierarchy_dimensions.hpp>

#define CUDART(call) REQUIRE((call) == cudaSuccess)

// TODO make it work on NVC++
#ifdef __CUDA_ARCH__
#  define HOST_DEV_REQUIRE assert
#else
#  define HOST_DEV_REQUIRE REQUIRE
#endif

bool constexpr __host__ __device__ operator==(const dim3& lhs, const dim3& rhs)
{
  return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
}

template <typename Dims, typename Lambda>
void __global__ lambda_launcher(const Dims dims, const Lambda lambda)
{
  lambda(dims);
}

template <typename Comparator, unsigned int FilterArch>
bool arch_filter(const cudaDeviceProp& props)
{
  int act_arch = props.major * 10 + props.minor;
  if (Comparator()(act_arch, FilterArch))
  {
    return true;
  }
  return false;
}

static bool skip_host_exec(bool (*filter)(const cudaDeviceProp&))
{
  return false;
}

static bool skip_device_exec(bool (*filter)(const cudaDeviceProp&))
{
  cudaDeviceProp props;
  CUDART(cudaGetDeviceProperties(&props, 0));
  return filter(props);
}

template <typename Dims, typename Lambda, typename... Filters>
void test_host_dev(const Dims& dims, const Lambda& lambda, const Filters&... filters)
{
  SECTION("Host execution")
  {
    if ((... && !skip_host_exec(filters)))
    {
      // host testing
      lambda(dims);
    }
  }

  SECTION("Device execution")
  {
    // Asymmetrical but cleaner
    if ((... || skip_device_exec(filters)))
    {
      return;
    }

    cudaLaunchConfig_t config = {0};
    cudaLaunchAttribute attrs[1];
    config.attrs = &attrs[0];

    config.blockDim = dims.flatten(cuda_next::thread, cuda_next::block);
    config.gridDim  = dims.flatten(cuda_next::block, cuda_next::grid);

    if constexpr (cuda_next::has_level<cuda_next::cluster_level, decltype(dims)>)
    {
      dim3 cluster_dims                            = dims.flatten(cuda_next::block, cuda_next::cluster);
      config.attrs[config.numAttrs].id             = cudaLaunchAttributeClusterDimension;
      config.attrs[config.numAttrs].val.clusterDim = {cluster_dims.x, cluster_dims.y, cluster_dims.z};
      config.numAttrs                              = 1;
    }
    else
    {
      config.numAttrs = 0;
    }

    // device testing
    CUDART(cudaLaunchKernelEx(&config, lambda_launcher<Dims, Lambda>, dims, lambda));
    CUDART(cudaDeviceSynchronize());
  }
}

template <typename Fn, typename Tuple>
void apply_each(const Fn& fn, const Tuple& tuple)
{
  cuda::std::apply(
    [&](const auto&... elems) {
      (fn(elems), ...);
    },
    tuple);
}

#endif
