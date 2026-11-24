//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This example demonstrates how kernel functors can be launched using cuda::launch.

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/stream>

#include <cstdio>
#include <exception>

#include "common.cuh"

// This is a kernel functor, a callable object with operator() decorated with __device__ attribute. When launched, the
// object is copied to the device and operator() is invoked on the device.
struct KernelFunctor
{
  // The operator() must be decorated with __device__ attribute. It can also be a template.
  __device__ void operator()(const KernelName& kernel_name) const
  {
    say_hello(cuda::gpu_thread.index(cuda::block), kernel_name);
  }
};

// Kernel functor can contain data. However, the functor must be trivially copyable.
struct KernelFunctorWithData
{
  KernelName kernel_name_;

  __device__ void operator()() const
  {
    say_hello(cuda::gpu_thread.index(cuda::block), kernel_name_);
  }
};

// Kernel functors can also take the cuda::kernel_config objects as the first argument. That way the kernel has access
// to compile time information of the block and grid dimensions which can produce better optimized kernels.
struct KernelFunctorWithConfig
{
  template <class Config>
  __device__ void operator()(const Config& config, const KernelName& kernel_name) const
  {
    // The config can be used in hierarchy queries for better performance.
    say_hello(cuda::gpu_thread.index(cuda::block, config), kernel_name);
  }
};

// Kernel functors can provide a default config that is combined with the config passed to cuda::launch. This can be
// useful for example when a kernel functor requires cooperative launch.
struct KernelFunctorWithDefaultConfig
{
  // Kernel functor provides the default config by implementing the .default_config() method.
  auto default_config() const
  {
    // This default config only specifies that the block dimensions are 2x2. The config passed to cuda::launch must
    // provide grid dimensions, otherwise the kernel functor wouldn't be able to be launched.
    return cuda::make_config(cuda::block_dims<2, 2>());
  }

  template <class Config>
  __device__ void operator()(const Config& config, const KernelName& kernel_name) const
  {
    say_hello(cuda::gpu_thread.index(cuda::block, config), kernel_name);
  }
};

// Kernel functor can use the
struct KernelFunctorWithDynamicSmem
{
  template <class Config>
  __device__ void operator()(const Config& config, const KernelName& kernel_name) const
  {
    // Retrieve the dynamic shared memory view. Since we passed uint3[4], we will get cuda::std::span<uint3>.
    const auto smem = cuda::dynamic_shared_memory(config);

    // Get all necessary hierarchy values.
    const auto tindex = cuda::gpu_thread.index(cuda::block, config);
    const auto trank  = cuda::gpu_thread.rank(cuda::block, config);
    const auto tcount = cuda::gpu_thread.count(cuda::block, config);

    // Each thread will write it's index to the next thread's index in the shared memory.
    smem[(trank + 1) % tcount] = tindex;

    // Wait for the all threads to finish the write to shared memory.
    __syncthreads();

    // Call say hello with received previous thread's index.
    say_hello(smem[trank], kernel_name);
  }
};

int main()
try
{
  // Check we have at least one device.
  if (cuda::devices.size() == 0)
  {
    std::fprintf(stderr, "No CUDA devices found\n");
    return 1;
  }

  // We will use the first device.
  cuda::device_ref device = cuda::devices[0];

  // cuda::launch always requires a work submitter, so let's create a CUDA stream.
  cuda::stream stream{device};

  // Set block and grid dimensions to be used with the kernel config. Dimensions specified as template parameters will
  // be statically known in the kernel.
  const auto block_dims = cuda::block_dims<2, 2>();
  const auto grid_dims  = cuda::grid_dims(dim3{1});

  // Make the kernel config.
  const auto kernel_config = cuda::make_config(grid_dims, block_dims);

  // For kernels that use dynamic shared memory, we need a dynamic shared memory option to be passed in the kernel
  // config.
  const auto dyn_smem_opt = cuda::dynamic_shared_memory<uint3[]>(cuda::gpu_thread.count(cuda::block, kernel_config));

  // Make the kernel config with dynamic shared memory option.
  const auto kernel_config_with_dyn_smem = cuda::make_config(grid_dims, block_dims, dyn_smem_opt);

  // Launch the kernel functor using the kernel config.
  cuda::launch(stream, kernel_config, KernelFunctor{}, KernelName{"kernel functor"});

  // Kernel functor can also contain data.
  cuda::launch(stream, kernel_config, KernelFunctorWithData{KernelName{"kernel functor with data"}});

  // If the kernel functor in invocable with the kernel config, it's automatically passed as the first parameter by the
  // cuda::launch function.
  cuda::launch(stream, kernel_config, KernelFunctorWithConfig{}, KernelName{"kernel functor with config"});

  // When launching a kernel functor with default config, we need to pass just a partial config as the launch parameter.
  // The missing parts are supplied from the default config inside the cuda::launch function.
  cuda::launch(stream,
               cuda::make_config(grid_dims),
               KernelFunctorWithDefaultConfig{},
               KernelName{"kernel functor with default config"});

  // Launch the kernel functor that uses dynamic shared memory.
  cuda::launch(stream,
               kernel_config_with_dyn_smem,
               KernelFunctorWithDynamicSmem{},
               KernelName{"kernel functor with dynamic shared memory"});

  // Wait for all of the tasks in the stream to complete.
  stream.sync();
}
catch (const cuda::cuda_error& e)
{
  std::fprintf(stderr, "CUDA error: %s\n", e.what());
  return 1;
}
catch (const std::exception& e)
{
  std::fprintf(stderr, "Error: %s\n", e.what());
  return 1;
}
catch (...)
{
  std::fprintf(stderr, "An unknown error was encountered\n");
  return 1;
}
