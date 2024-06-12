//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAUNCH_LAUNCH
#define _CUDAX__LAUNCH_LAUNCH
#include <cuda_runtime.h>

#include <cuda/experimental/__launch/confiuration.cuh>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/stream_ref>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

namespace detail
{
template <typename Config, typename Kernel, class... Args>
__global__ void kernel_launcher(const Config conf, Kernel kernel_fn, Args... args)
{
  kernel_fn(conf, args...);
}

template <typename Kernel, class... Args>
__global__ void kernel_launcher_no_config(Kernel kernel_fn, Args... args)
{
  kernel_fn(args...);
}

template <typename Config, typename Kernel, typename... Args>
_CCCL_NODISCARD cudaError_t
launch_impl(::cuda::stream_ref stream, Config conf, const Kernel& kernel_fn, const Args&... args)
{
  cudaLaunchConfig_t config               = {0};
  cudaError_t status                      = cudaSuccess;
  constexpr bool has_cluster_level        = has_level<cluster_level, decltype(conf.dims)>;
  constexpr unsigned int num_attrs_needed = conf.num_attrs_needed() + has_cluster_level;
  cudaLaunchAttribute attrs[num_attrs_needed == 0 ? 1 : num_attrs_needed];
  config.attrs    = &attrs[0];
  config.numAttrs = 0;
  config.stream   = stream.get();

  status = conf.apply(config, reinterpret_cast<void*>(kernel_fn));
  if (status != cudaSuccess)
  {
    return status;
  }

  config.blockDim = conf.dims.extents(thread, block);
  config.gridDim  = conf.dims.extents(block, grid);

  if constexpr (has_cluster_level)
  {
    auto cluster_dims                            = conf.dims.extents(block, cluster);
    config.attrs[config.numAttrs].id             = cudaLaunchAttributeClusterDimension;
    config.attrs[config.numAttrs].val.clusterDim = {
      static_cast<unsigned int>(cluster_dims.x),
      static_cast<unsigned int>(cluster_dims.y),
      static_cast<unsigned int>(cluster_dims.z)};
    config.numAttrs++;
  }

  // TODO lower to cudaLaunchKernelExC?
  return cudaLaunchKernelEx(&config, kernel_fn, args...);
}
} // namespace detail

template <typename... Args, typename... Config, typename Dimensions, typename Kernel>
void launch(
  ::cuda::stream_ref stream, const kernel_config<Dimensions, Config...>& conf, const Kernel& kernel, Args... args)
{
  cudaError_t status;
  if constexpr (::cuda::std::is_invocable_v<Kernel, kernel_config<Dimensions, Config...>, Args...>)
  {
    auto launcher = detail::kernel_launcher<kernel_config<Dimensions, Config...>, Kernel, Args...>;
    status        = detail::launch_impl(stream, conf, launcher, conf, kernel, args...);
  }
  else
  {
    static_assert(::cuda::std::is_invocable_v<Kernel, Args...>);
    auto launcher = detail::kernel_launcher_no_config<Kernel, Args...>;
    status        = detail::launch_impl(stream, conf, launcher, kernel, args...);
  }
  if (status != cudaSuccess)
  {
    throw ::cuda::cuda_error(status, "Failed to launch a kernel");
  }
}

template <typename... Args, typename... Levels, typename Kernel>
void launch(::cuda::stream_ref stream, const hierarchy_dimensions<Levels...>& dims, const Kernel& kernel, Args... args)
{
  cudaError_t status;
  if constexpr (::cuda::std::is_invocable_v<Kernel, hierarchy_dimensions<Levels...>, Args...>)
  {
    auto launcher = detail::kernel_launcher<hierarchy_dimensions<Levels...>, Kernel, Args...>;
    status        = detail::launch_impl(stream, kernel_config(dims), launcher, dims, kernel, args...);
  }
  else
  {
    static_assert(::cuda::std::is_invocable_v<Kernel, Args...>);
    auto launcher = detail::kernel_launcher_no_config<Kernel, Args...>;
    status        = detail::launch_impl(stream, kernel_config(dims), launcher, kernel, args...);
  }
  if (status != cudaSuccess)
  {
    throw ::cuda::cuda_error(status, "Failed to launch a kernel");
  }
}

/* Functions accepting __global__ function pointer (needs to be instantiated or template arguments need to be passed
 * into launch template, but it will support implicit conversion of arguments) */
template <typename... ExpArgs, typename... ActArgs, typename... Config, typename Dimensions>
void launch(::cuda::stream_ref stream,
            const kernel_config<Dimensions, Config...>& conf,
            void (*kernel)(kernel_config<Dimensions, Config...>, ExpArgs...),
            ActArgs&&... actArgs)
{
  cudaError_t status = [&](ExpArgs... args) {
    return detail::launch_impl(stream, conf, kernel, conf, args...);
  }(std::forward<ActArgs>(actArgs)...);
  if (status != cudaSuccess)
  {
    throw ::cuda::cuda_error(status, "Failed to launch a kernel");
  }
}

template <typename... ExpArgs, typename... ActArgs, typename... Levels>
void launch(::cuda::stream_ref stream,
            const hierarchy_dimensions<Levels...>& dims,
            void (*kernel)(hierarchy_dimensions<Levels...>, ExpArgs...),
            ActArgs&&... actArgs)
{
  cudaError_t status = [&](ExpArgs... args) {
    return detail::launch_impl(stream, kernel_config(dims), kernel, dims, args...);
  }(std::forward<ActArgs>(actArgs)...);
  if (status != cudaSuccess)
  {
    throw ::cuda::cuda_error(status, "Failed to launch a kernel");
  }
}

template <typename... ExpArgs, typename... ActArgs, typename... Config, typename Dimensions>
void launch(::cuda::stream_ref stream,
            const kernel_config<Dimensions, Config...>& conf,
            void (*kernel)(ExpArgs...),
            ActArgs&&... actArgs)
{
  cudaError_t status = [&](ExpArgs... args) {
    return detail::launch_impl(stream, conf, kernel, args...);
  }(std::forward<ActArgs>(actArgs)...);
  if (status != cudaSuccess)
  {
    throw ::cuda::cuda_error(status, "Failed to launch a kernel");
  }
}

template <typename... ExpArgs, typename... ActArgs, typename... Levels>
void launch(::cuda::stream_ref stream,
            const hierarchy_dimensions<Levels...>& dims,
            void (*kernel)(ExpArgs...),
            ActArgs&&... actArgs)
{
  cudaError_t status = [&](ExpArgs... args) {
    return detail::launch_impl(stream, kernel_config(dims), kernel, args...);
  }(std::forward<ActArgs>(actArgs)...);
  if (status != cudaSuccess)
  {
    throw ::cuda::cuda_error(status, "Failed to launch a kernel");
  }
}

} // namespace cuda::experimental
#endif // _CCCL_STD_VER >= 2017
#endif // _CUDAX__LAUNCH_LAUNCH
