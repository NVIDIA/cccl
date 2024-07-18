//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __TESTING_COMMON_H__
#define __TESTING_COMMON_H__

#include <cuda/experimental/hierarchy.cuh>

#include <exception>
#include <iostream>
#include <sstream>

#include <catch2/catch.hpp>

namespace cudax = cuda::experimental;

#define CUDART(call) REQUIRE((call) == cudaSuccess)

inline void __device__ cudax_require_impl(
  bool condition, const char* condition_text, const char* filename, unsigned int linenum, const char* funcname)
{
  if (!condition)
  {
    // TODO do warp aggregate prints for easier readibility?
    printf("%s:%u: %s: block: [%d,%d,%d], thread: [%d,%d,%d] Condition `%s` failed.\n",
           filename,
           linenum,
           funcname,
           blockIdx.x,
           blockIdx.y,
           blockIdx.z,
           threadIdx.x,
           threadIdx.y,
           threadIdx.z,
           condition_text);
    __trap();
  }
}

// TODO make it work on NVC++
#ifdef __CUDA_ARCH__
#  define CUDAX_REQUIRE(condition) cudax_require_impl(condition, #condition, __FILE__, __LINE__, __PRETTY_FUNCTION__);
#else
#  define CUDAX_REQUIRE REQUIRE
#endif

bool constexpr __host__ __device__ operator==(const dim3& lhs, const dim3& rhs)
{
  return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
}

namespace Catch
{
template <>
struct StringMaker<dim3>
{
  static std::string convert(dim3 const& dims)
  {
    std::ostringstream oss;
    oss << "(" << dims.x << ", " << dims.y << ", " << dims.z << ")";
    return oss.str();
  }
};
} // namespace Catch

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

static bool skip_host_exec(bool (* /* filter */)(const cudaDeviceProp&))
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

    cudaLaunchConfig_t config = {};
    config.gridDim            = {0};
    cudaLaunchAttribute attrs[1];
    config.attrs = &attrs[0];

    config.blockDim = dims.extents(cudax::thread, cudax::block);
    config.gridDim  = dims.extents(cudax::block, cudax::grid);

    if constexpr (cudax::has_level<cudax::cluster_level, decltype(dims)>)
    {
      dim3 cluster_dims                            = dims.extents(cudax::block, cudax::cluster);
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
